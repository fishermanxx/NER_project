
import torch
from torch import nn
import torch.nn.init as I

class CRF(nn.Module):
    def __init__(self, config={}):
        '''
        :param - dict
            param['n_tags']
            param['start_idx']  int, <start> tag index for entity tag seq
            param['end_idx']   int, <end> tag index for entity tag seq
            param['use_cuda']
        '''
        super(CRF, self).__init__()
        self.config = config
        self.n_tags = self.config.get('n_tags', 3)
        self.start_idx = self.config.get('start_idx', 0)
        self.end_idx = self.config.get('end_idx', 1)
        self.use_cuda = self.config.get('use_cuda', 0)

        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))
        self.transitions.data[:, self.start_idx] = self.transitions.data[self.end_idx, :] = -10000

        self.reset_parameters()

    def reset_parameters(self):
        I.normal_(self.transitions.data, 0, 1)
        self.transitions.data[:, self.start_idx] = self.transitions.data[self.end_idx, :] = -10000

    def log_norm_score(self, logits, lens, use_cuda=None):
        '''
        求所有路径的score之和, 或者alpha, 如果batch_size=1的情况下, exp(alpha_tj)表示t时刻以tag_j结尾的所有路径score之和.
        alpha[i,j] = log( sum_j' exp( alpha_(i-1)j' + T_j'j + E_jwi ) )
        mat[j', j] = alpha_(i-1)j' + T_j'j + E_jwi
            @(alpha) exp (alpha[i, j]): (alpha)表示i时刻以tag_j结尾的所有路径score之和.  (n_tags, 1)
            @(transition) T_j'j: (transition)表示score transform from tag_j' to tag_j   (n_tags, n_tags)
            @(logits) E_jwi: (logits)表示在i时刻给定所有word之后以tag_j结尾的score.   (1, n_tags)

        :param
            @logits: (batch_size, T, n_tags), torch.tensor, self._get_lstm_features的output, 
            @lens: (batch_size), list, 具体每个句子的长度 
        :return
            @norm: (batch_size), torch.tensor, 每个句子所有可能路径score之和, 取过log
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size, T, n_tags = logits.size()
        
        ## 初始化t=0时候的alpha
        alpha = logits.data.new(batch_size, self.n_tags).fill_(-10000.0)   ##(batch_size, n_tags)
        ## 处理<start>
        alpha[:, self.start_idx] = 0

        c_lens = self._to_tensor(lens, use_cuda)
        logits_t = logits.transpose(1, 0)  #(T, batch_size, n_tags)
        
        for logit in logits_t:
            ## logit(batch_size, n_tags)
            logit_exp = logit.unsqueeze(1)  #(batch_size, 1, n_tags)
            logit_exp = logit_exp.expand(batch_size, *self.transitions.size())  #(batch_size, n_tags, n_tags)  *self.transitions.size()解压缩
            
            alpha_exp = alpha.unsqueeze(-1)  #(batch_size, n_tags, 1)
            alpha_exp = alpha_exp.expand(batch_size, *self.transitions.size())  #(batch_size, n_tags, n_tags)

            trans_exp = self.transitions.unsqueeze(0)  #(1, n_tags, n_tags)
            trans_exp = trans_exp.expand_as(alpha_exp)  #(batch_size, n_tags, n_tags)

            mat = trans_exp + logit_exp + alpha_exp
            alpha_nxt = self._log_sum_exp(mat, dim=1)  #(batch_size, 1, n_tags)
            alpha_nxt = alpha_nxt.squeeze(1)  #(batch_size, n_tags)
            
            ## 添加mask, 对于超出句子长度部分的score 不去进行计算
            mask = (c_lens > 0).float().unsqueeze(-1) #(batch_size, 1)
            mask = mask.expand_as(alpha) #(batch_size, n_tags)
            
            alpha = mask*alpha_nxt + (1-mask)*alpha  #(batch_size, n_tags)
            c_lens = c_lens - 1

        # 处理<end>, 最终从最后一个tag跳到tag_<end>的score  不能忽略，虽然分母上面都加上了，但是对于不同句子来说，每个句子最后一个tag是不一定相同的，所以分子上这最后一项会不同
        trans_end = self.transitions[:, self.end_idx].unsqueeze(0).expand_as(alpha)  #(batch_size, n_tags)
        mat = alpha + trans_end
        norm = self._log_sum_exp(mat, dim=1).squeeze(-1)  #(batch_size)
        return norm

    def path_score(self, logits, y_ent, lens, use_cuda=None):
        """
        求路径上面的总score
        :param
            @logits: (batch_size, T, n_tags), torch.tensor, self._get_lstm_features的output, 类似eject score
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别, 
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return
            @score: (batch_size), torch.tensor, the total score of the whole path for each sentence in the batch
        """
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        t_score = self._transition_score(y_ent, lens, use_cuda)
        e_score = self._ejection_score(logits, y_ent, lens, use_cuda)

        score = t_score + e_score
        return score

    def _transition_score(self, y_ent, lens, use_cuda=None):
        """
        checked
        求路径上面的transition_score之和,使用矩阵运算形式, 而不是用for循环来做
        :param
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别, 
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return
            @score: (batch_size), torch.tensor, transition_score of each path, 
        """
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size, T = y_ent.shape
        y_ent = self._to_tensor(y_ent, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)   #(batch_size)

        ## add <start> to the head of y_ent
        # labels_ext = torch.LongTensor(batch_size, T+2)
        labels_ext = y_ent.data.new(batch_size, T+2)  #(batch_size, T+2)
        labels_ext[:,0] = self.start_idx
        labels_ext[:, 1:-1] = y_ent
        
        ## add <end> to the tail of y_ent, maybe more than one <end>, from the end of sentence to max_len
        # mask = sequence_mask(lens+1, T+2).long()  #(batch_size, T+2)  len+1是因为句子开头插入了一个<start>tag
        mask = self._generate_mask(lens+1, T+2).long()  #(batch_size, T+2)  len+1是因为句子开头插入了一个<start>tag
        # print('mask')
        # print([(i, mask[0, i].item()) for i in range(len(mask[0]))])

        pad_stop = y_ent.data.new(batch_size, T+2).fill_(self.end_idx)  #(batch_size, T+2) 
        labels_ext = (1-mask)*pad_stop + mask * labels_ext  #(batch_size, T+2)

        ##计算所有的transition score--------------------------
        trn_mat = self.transitions.transpose(1, 0).unsqueeze(0).expand(batch_size, *self.transitions.size())  #(batch_size, n_tags, n_tags)

        lbl_r = labels_ext[:, 1:] #(batch_size, T+1)
        lbl_r_idx = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.n_tags)  #(batch_size, T+1, n_tags) ,不看batch_size, lbl_r_idx[t, j]表示在t时刻从tag_j 转移到的tag

        # torch.gather(mat, dim, idx), 要求mat, idx这两个矩阵的维度除了axis=dim这一维之外都要相等
        # 最终结果的维度和idx的维度一样，相当于替换idx中的每一个元素，
        # ex. mat(3, 4), dim = 0, idx(10, 4) ---> res(10, 4)
        # res[i, j] = mat[idx[i, j], j]
        trn_all = torch.gather(trn_mat, dim=1, index=lbl_r_idx)  #(batch_size, T+1, n_tags), shape=index.shape, 不看batch_size, trn_row[0][t][j]表示batch中第0笔数据在t时刻从tag_j 转移到 lbl_r[t]的transition score
        
        ##计算一条路径上的的transition score------------------
        lbl_l_idx = labels_ext[:, :-1].unsqueeze(-1) #(batch_size, T+1, 1)
        trn_scr = torch.gather(trn_all, dim=2, index=lbl_l_idx).squeeze(-1) #(batch_size, T+1)
        
        ##检查transition score 的计算是否正确，和for 循环的结果作为比较
        # check_trn_scr(trn_scr, trn_all, self.transitions, labels_ext)

        # mask = sequence_mask(lens+1, max_len=T+1).float()  ## (batch_size, T+1), 包括最后一个tag -> <end> 的转移score, 但是之后的<end>-><end> transition score 要mask掉
        mask = self._generate_mask(lens+1, max_len=T+1).float()  ## (batch_size, T+1), 包括最后一个tag -> <end> 的转移score, 但是之后的<end>-><end> transition score 要mask掉
        trn_scr = trn_scr * mask  ## (batch_size, T+1)
        
        # check_mask_seq(mask, trn_scr)
        score = trn_scr.sum(1)
        return score

    def _ejection_score(self, logits, y_ent, lens, use_cuda=None):
        """
        checked
        求路径上面的ejection_score之和,使用矩阵运算形式, 而不是用for循环来做
        :param
            @logits: (batch_size, T, n_tags), torch.tensor, self._get_lstm_features的output, 类似eject score
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别, 
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return
            @score: (batch_size), torch.tensor, the ejection_score of the whole path for each sentence in the batch
        """
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        y_ent = self._to_tensor(y_ent, use_cuda)
        lens = self._to_tensor(lens, use_cuda)
        # print('lens')
        # print(lens)

        T = y_ent.shape[1]
        y_ent_exp = y_ent.unsqueeze(-1) #(batch_size, T, 1)
        scores = torch.gather(logits, dim=2, index=y_ent_exp).squeeze(-1) #(batch_size, T)

        # mask = sequence_mask(lens, T).float()
        mask = self._generate_mask(lens, T).float()
        scores = scores * mask

        # check_mask_seq(mask, scores)

        score = scores.sum(1) #(batch_size)
        return score

    def viterbi_decode(self, logits, lens, use_cuda=None):
        '''
        checked - not sure
        返回的是所有路径中score最大的一条路径, 类似nore_score仍然用dp算法, 但是
        不需要用alpha: alpha[t, j]在t时刻以tag_j为节点的所有路径之和
        使用vit: vit[t, j]在t时刻以tag_j为节点的单路径最大score
        :param
            @logits: (batch_size, T, n_tags), torch.tensor, self._get_lstm_features的output, 类似eject score
            @lens: (batch_size), list, 具体每个句子的长度
        :return
            @paths: (batch_size, T), torch.tensor, 最佳句子路径  '<start> this is a sentence.'
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size, T, n_tags = logits.size()
        vit = logits.data.new(batch_size, self.n_tags).fill_(-10000)  #(batch_size, n_tags)
        vit[:, self.start_idx] = 0
        c_lens = self._to_tensor(lens, use_cuda) #(batch_size)

        logits_t = logits.transpose(1, 0)  #(T, batch_size, n_tags)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(-1) #(batch_size, n_tags, 1)
            vit_exp = vit_exp.expand(batch_size, self.n_tags, self.n_tags) #(batch_size, n_tags, n_tags)

            trans_exp = self.transitions.unsqueeze(0) #(1, n_tags, n_tags)
            trans_exp = trans_exp.expand_as(vit_exp) #(batch_size, n_tags, n_tags)    

            mat = vit_exp + trans_exp #(batch_size, n_tags, n_tags)
            vit_max, vit_argmax = mat.max(dim=1)  #(batch_size, n_tags), (batch_size, n_tags)  #忽略batch, 为矩阵(n_tag, n_tag)中每一列的最大值
            
            vit_nxt = vit_max + logit
            pointers.append(vit_argmax.unsqueeze(0)) ##TODO: vit_argmax结尾部分有些问题，当self.transition 不大的时候就没有问题，相当于在句子的结尾之后的一列所有的tag都是从前一个最大的tag过来才最大

            mask = (c_lens > 0).float() #(batch_size)
            mask = mask.unsqueeze(-1).expand_as(vit_nxt)  #(batch_size, n_tags)
            vit = mask*vit_nxt + (1-mask)*vit   ##如果长度超过句子原始长度，则总分数就保持不变了

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            final_trans_exp = self.transitions[:, self.end_idx].unsqueeze(0).expand_as(vit) #(batch_size, n_tags)
            vit += mask*final_trans_exp

            c_lens = c_lens - 1

            ##===================Another Try===================================
            # vit_exp = vit.unsqueeze(-1) #(batch_size, n_tags, 1)
            # vit_exp = vit_exp.expand(batch_size, self.n_tags, self.n_tags) #(batch_size, n_tags, n_tags)

            # trans_exp = self.transitions.unsqueeze(0) #(1, n_tags, n_tags)
            # trans_exp = trans_exp.expand_as(vit_exp) #(batch_size, n_tags, n_tags)

            # logit_exp = logit.unsqueeze(1) #(batch_size, 1, n_tags)
            # logit_exp = logit_exp.expand_as(vit_exp) #(batch_size, n_tags, n_tags)

            # mat = vit_exp + trans_exp + logit_exp #(batch_size, n_tags, n_tags)
            # vit_nxt, vit_argmax = mat.max(dim=1) #(batch_size, n_tags), (batch_size, n_tags)  #忽略batch, 为矩阵(n_tag, n_tag)中每一列的最大值

            # mask = (c_lens > 0).float()  #(batch_size)
            # mask = mask.unsqueeze(-1).expand_as(vit_nxt)  #(batch_size, n_tags)
            # vit_nxt = mask*vit_nxt + (1-mask)*vit  #(batch_size, n_tags)  在句子结束之后score一直保持不变，就是结尾的那个score, 之后的<end>to<end>不考虑


            # #处理最后一个单词转到tag_<end>的情况  TODO: 感觉这部分有点浪费计算资源
            # mask = (c_lens == 0).float().unsqueeze(-1).expand_as(vit)  #(batch_size, n_tags)
            # final_trans_exp = self.transitions[:, self.end_idx].unsqueeze(0).expand_as(vit) #(batch_size, n_tags)
            # vit_final = vit + final_trans_exp #(batch_size, n_tags)
            # _, vit_final_argmax = vit_final.max(dim=1) #(batch_size, n_tags)
            # vit_final_argmax = vit_final_argmax.unsqueeze(-1).expand_as(vit_argmax) #(batch_size, n_tags)
            
            # vit_argmax = mask.long()*vit_final_argmax + (1-mask.long())*vit_argmax
            # pointers.append(vit_argmax.squeeze(-1).unsqueeze(0))  #(1, batch_size, n_tags)
            # vit = mask*vit_final + (1-mask)*vit_nxt

            # c_lens = c_lens - 1

        pointers = torch.cat(pointers)  #(T, batch_size, n_tags)
        scores, idx = vit.max(dim=1)  #(batch_size)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            #argmax  (batch_size, n_tags)
            idx_exp = idx.unsqueeze(-1)  #(batch_size, 1)
            idx = torch.gather(argmax, dim=1, index=idx_exp).squeeze(-1)  #(batch_size)
            paths.insert(0, idx.unsqueeze(1))
        
        paths = torch.cat(paths, dim=1)  #(batch_size, T)
        return scores, paths[:, 1:]

    @staticmethod
    def _to_tensor(x, use_cuda=False):
        if use_cuda:
            return torch.tensor(x, dtype=torch.long).cuda()
        else:
            return torch.tensor(x, dtype=torch.long)  

    @staticmethod
    def _generate_mask(lens, max_len=None):
        '''
        返回一个mask, 遮住<pad>部分的无用信息.
        :param
            @lens: (batch_size), torch.tensor, the lengths of each sentence
            @max_len: int, the max length of the sentence - T
        :return 
            @mask: (batch_size, max_len)
        '''
        batch_size = lens.shape[0]
        if max_len is None:
            max_len = lens.max()
        ranges = torch.arange(0, max_len).long()  #(max_len)
        if lens.is_cuda:
            ranges = ranges.cuda()
        ranges = ranges.unsqueeze(0).expand(batch_size, max_len)   #(batch_size, max_len)
        lens_exp = lens.unsqueeze(1).expand_as(ranges)  #(batch_size, max_len)
        mask = ranges < lens_exp
        return mask

    @staticmethod
    def _log_sum_exp(mat, dim):
        """
        纵向求和
        :param
            @mat (n_tag, n_tag)  -- 分析以这种简单情况为例
            @mat (batch_size, n_tag, n_tag), torch.tensor
        :return
            @res (1, n_tag)
        alpha[i,j] = log( sum_j' exp( alpha_(i-1)j' + T_j'j + E_jwi ) )
        mat[j', j] = alpha_(i-1)j' + T_j'j + E_jwi
        1. exp(999)上溢问题   [1, 999, 4]
        log( exp(1)+exp(999)+exp(4) ) = log( exp(1-999)+exp(999-999)+exp(4-999) ) + 999, 找到每一列的最大值，进行操作
        """
        vmax = mat.max(dim, keepdim=True).values  #(1, n_tag) or (batch_size, 1, n_tag)
        # print('vmax shape:', vmax.shape)
        res = (mat-vmax).exp().sum(dim, keepdim=True).log() + vmax #(1, n_tag) or (batch_size, 1, n_tag)
        # print('alpha shape:', res.shape)
        return res


if __name__ == '__main__':
    crf_params = {
        'n_tags':45, 
        'start_idx':43, 
        'end_idx':44, 
        'use_cuda':0
    }
    mycrf = CRF(crf_params)