# !-*- coding:utf-8 -*-
# TODO: 
# 1. 句子经过blstm时长度的处理
# 2. ***dropout layer 的添加  
# 3. ***linear layer 之后激活层的添加 , tanh(x), relu(x)等
# 5. ***embedding layer 初始化使用glove embedding

import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

import transformers

# from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import os
import math
import json
import numpy as np

torch.manual_seed(1)


class CRF(nn.Module):
    def __init__(self, params={}):
        '''
        :param - dict
            param['n_tags']
            param['start_idx']  int, <start> tag index for entity tag seq
            param['end_idx']   int, <end> tag index for entity tag seq
            param['use_cuda']
        '''
        super(CRF, self).__init__()
        self.params = params
        self.n_tags = self.params['n_tags']
        self.start_idx = self.params['start_idx']
        self.end_idx = self.params['end_idx']
        self.use_cuda = self.params['use_cuda']

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
        lens = self._to_tensor(lens, use_cuda)
        # print('lens')
        # print(lens)

        ## add <start> to the head of y_ent
        # labels_ext = torch.LongTensor(batch_size, T+2)
        labels_ext = y_ent.data.new(batch_size, T+2)  #(batch_size, T+2)
        labels_ext[:,0] = self.start_idx
        labels_ext[:, 1:-1] = y_ent
        
        ## add <end> to the tail of y_ent, maybe more than one <end>, from the end of sentence to max_len
        # mask = sequence_mask(lens+1, T+2).long()  #(batch_size, T+2)  len+1是因为句子开头插入了一个<start>tag
        mask = self._generate_mask(lens+1, T+2, use_cuda).long()  #(batch_size, T+2)  len+1是因为句子开头插入了一个<start>tag
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
        mask = self._generate_mask(lens+1, max_len=T+1, use_cuda=use_cuda).float()  ## (batch_size, T+1), 包括最后一个tag -> <end> 的转移score, 但是之后的<end>-><end> transition score 要mask掉
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
        mask = self._generate_mask(lens, T, use_cuda).float()
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
            @paths: (batch_size, T+1), torch.tensor, 最佳句子路径
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
        
        paths = torch.cat(paths, dim=1)  #(batch_size, T+1)
        return scores, paths

    @staticmethod
    def _to_tensor(x, use_cuda=False):
        if use_cuda:
            return torch.tensor(x, dtype=torch.long).cuda()
        else:
            return torch.tensor(x, dtype=torch.long)  

    @staticmethod
    def _generate_mask(lens, max_len=None, use_cuda=False):
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
        if use_cuda:
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

class BERT_LSTM_CRF(nn.Module):
    def __init__(self, params={}, show_param=False):
        '''
        :param - dict
            param['embedding_dim']
            param['hidden_dim']
            param['n_tags']
            param['n_words']
            param['start_idx']  int, <start> tag index for entity tag seq
            param['end_idx']   int, <end> tag index for entity tag seq
            param['use_cuda']
            param['dropout_prob']
            param['lstm_layer_num']
        '''
        super(BERT_LSTM_CRF, self).__init__()
        self.params = params
        self.embedding_dim = self.params.get('embedding_dim', 768)
        self.hidden_dim = self.params['hidden_dim']
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.n_tags = self.params['n_tags']
        self.n_words = self.params['n_words']
        self.start_idx = self.params['start_idx']
        self.end_idx = self.params['end_idx']
        self.use_cuda = self.params['use_cuda']
        self.dropout_prob = self.params.get('dropout_prob', 0.05)
        self.lstm_layer_num = self.params.get('lstm_layer_num', 1)

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'embedding_dim: {self.embedding_dim}', 1)
        log(f'hidden_dim: {self.hidden_dim}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log(f'lstm_layer_num: {self.lstm_layer_num}', 1)
        log(f'dropout_prob: {self.dropout_prob}', 1)  
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        self.word_embeds = nn.Embedding(self.n_words, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, batch_first=True, num_layers=self.lstm_layer_num, dropout=self.dropout_prob, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.n_tags)
        crf_params = {'n_tags':self.n_tags, 'start_idx':self.start_idx, 'end_idx':self.end_idx, 'use_cuda':self.use_cuda}
        self.crf = CRF(crf_params)
        self.bert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

    def reset_parameters(self):        
        I.xavier_normal_(self.word_embeds.weight.data)
        self.lstm.reset_parameters()
        # stdv = 1.0 / math.sqrt(self.hidden_dim)
        # for weight in self.lstm.parameters():
        #     I.uniform_(weight, -stdv, stdv)
        I.xavier_normal_(self.hidden2tag.weight.data)
        self.crf.reset_parameters()
        
    def _get_lstm_features(self, x, lens, use_cuda=None):
        '''
        TODO: 添加关于句子长度处理的部分
        :param  
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: 每个句子的实际长度
        :return 
            @lstm_feature: (batch_size, T, n_tags) -- 类似于eject score, torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size, T = x.shape
        ##embedding layer
        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        # embeds = self.word_embeds(words_tensor)  #(batch_size, T, n_embed)

        lens = self._to_tensor(lens)
        att_mask = self._generate_mask(lens, max_len=T).cuda()
        embeds = self.bert(words_tensor, attention_mask=att_mask)[0]  #(batch_size, T, n_embed)
        

        ##LSTM layer
        if use_cuda:
            h_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2).cuda()  #(n_layer*n_dir, N, n_hid)
            c_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2).cuda()
        else:
            h_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2)
            c_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2)
        # c_0 = h_0.clone()
        hidden = (h_0, c_0)
        lstm_out, _hidden = self.lstm(embeds, hidden)   #(batch_size, T, n_dir*n_hid), (h, c)

        ##FC layer
        lstm_feature = self.hidden2tag(lstm_out) #(batch_size, T, n_tags)
        lstm_feature = torch.tanh(lstm_feature)

        return lstm_feature

    def _loss(self, x, y_ent, lens, use_cuda=None):
        '''
        loss function: neg_log_likelihood
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (batch_size), torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda

        logits = self._get_lstm_features(x, lens)
        log_norm_score = self.crf.log_norm_score(logits, lens)
        path_score = self.crf.path_score(logits, y_ent, lens)

        loss = log_norm_score - path_score
        return loss

    def _output(self, x, lens, use_cuda=None):
        '''
        return the crf decode paths
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @paths: (batch_size, T+1), torch.tensor, 最佳句子路径
            @scores: (batch_size), torch.tensor, 最佳句子路径上的得分
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        logits = self._get_lstm_features(x, lens, use_cuda)
        scores, paths = self.crf.viterbi_decode(logits, lens, use_cuda)
        return paths, scores

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        if os.path.exists(path):
            if self.use_cuda:
                self.load_state_dict(torch.load(path))
            else:
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            pass

    def train_model(self, data_loader: KGDataLoader, train_dataset=None, eval_dataset=None, hyper_param={}, use_cuda=None):
        '''
        :param
            @data_loader: (KGDataLoader),
            @result_dir: (str) path to save the trained model and extracted dictionary
            @hyper_param: (dict)
                @hyper_param['EPOCH']
                @hyper_param['batch_size']
                @hyper_param['learning_rate']
                @hyper_param['visualize_length']   #num of batches between two check points
                @hyper_param['isshuffle']
                @hyper_param['result_dir']
                @hyper_param['model_name']
        :return
            @loss_record, 
            @score_record
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        EPOCH = hyper_param.get('EPOCH', 3)
        BATCH_SIZE = hyper_param.get('batch_size', 4)
        LEARNING_RATE = hyper_param.get('learning_rate', 1e-2)
        visualize_length = hyper_param.get('visualize_length', 10)
        result_dir = hyper_param.get('result_dir', './result/')
        model_name = hyper_param.get('model_name', 'model.p')
        is_shuffle = hyper_param.get('isshuffle', True)
        DATA_TYPE = 'ent'
        

        train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset

        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        old_train_dict_path = os.path.join(result_dir, 'train_data_mat_dict.pkl')
        if os.path.exists(old_train_dict_path):
            train_data_mat_dict = data_loader.load_preprocessed_data(old_train_dict_path)
            log('Reload preprocessed data successfully~')
        else:
            train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
            data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        data_generator = Batch_Generator(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        if use_cuda:
            print('use cuda=========================')
            self.cuda()
        

        all_cnt = len(train_data_mat_dict['cha_matrix'])
        log(f'{model_name} Training start!', 0)
        loss_record = []
        score_record = []

        evel_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
        for epoch in range(EPOCH):
            self.train()
            log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
            loss = 0.0
            for cnt, data_batch in enumerate(data_generator):
                x, pos, _, _, y_ent, lens, data_list = data_batch
                optimizer.zero_grad()
                nll = self._loss(x, y_ent, lens)
                sub_loss = nll.mean()
                sub_loss.backward()
                optimizer.step()

                loss_avg = (nll/self._to_tensor(lens, use_cuda)).mean()
                loss += loss_avg
                if use_cuda:
                    loss_record.append(loss_avg.cpu().item())
                else:
                    loss_record.append(loss_avg.item())

                if (cnt+1) % visualize_length == 0:
                    loss_cur = loss / visualize_length
                    log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
                    loss = 0.0

                if cnt+1 % 100 == 0:
                    save_path = os.path.join(result_dir, model_name)
                    self.save_model(save_path)
                    print('Checkpoint saved successfully')
                #     print(data_list[0]['input'])
                #     pre_paths, pre_scores = self._output(x, lens)
                #     print('predict-path')
                #     print(pre_paths[0])
                #     print('target-path')
                #     print(y_ent[0])
            save_path = os.path.join(result_dir, model_name)
            self.save_model(save_path)
            print('Checkpoint saved successfully')

            temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=evel_param, use_cuda=use_cuda)
            score_record.append(temp_score)

        return loss_record, score_record

    @torch.no_grad()
    def predict(self, data_loader: KGDataLoader, data_set=None, hyper_param={}, use_cuda=False):
        '''
        预测出 test_data_mat_dict['y_ent_matrix']中的内容，重新填写进该matrix, 未预测之前都是0
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  ##默认4
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认None
        :return
            @result: list
                case = result[0]
                case['input']
                case['entity_list']
                    e = case['entity_list'][0]  
                    e['entity']:2016年04月08日
                    e['entity_type']:Date
                    e['entity_index']['begin']:13
                    e['entity_index']['end']:24
        '''
        BATCH_SIZE = hyper_param.get('batch_size', 64)
        ISSAVE = hyper_param.get('issave', False)
        result_dir = hyper_param.get('result_dir', './result/')
        DATA_TYPE = 'ent'

        if data_set is None:
            test_dataset = data_loader.dataset.test_dataset
        else:
            test_dataset = data_set

        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        old_test_dict_path = os.path.join(result_dir, 'test_data_mat_dict.pkl')
        if os.path.exists(old_test_dict_path):
            test_data_mat_dict = data_loader.load_preprocessed_data(old_test_dict_path)
            log('Reload preprocessed data successfully~')
        else:
            test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
            data_loader.save_preprocessed_data(old_test_dict_path, test_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        data_generator = Batch_Generator(test_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=False)
        
        if use_cuda:
            print('use cuda=========================')
            self.cuda()
        self.eval()   #disable dropout layer and the bn layer

        total_output_ent = []
        all_cnt = len(test_data_mat_dict['cha_matrix'])
        log(f'Predict start!', 0)
        for cnt, data_batch in enumerate(data_generator):
            x, pos, _, _, _, lens, _ = data_batch
            pre_paths, pred_scores = self._output(x, lens)  ##pre_paths, (batch_size, T+1), torch.tensor
            if use_cuda:
                pre_paths = pre_paths.data.cpu().numpy()[:, 1:].astype(np.int)
            else:
                pre_paths = pre_paths.data.numpy()[:, 1:].astype(np.int)
            total_output_ent.append(pre_paths)
            
            if (cnt+1) % 10 == 0:
                log(f'[PREDICT] step {(cnt+1)*BATCH_SIZE}/{all_cnt}', 1)

        ## add mask when the ent seq idx larger than sentance length
        pred_output = np.vstack(total_output_ent)   ###(N, max_length), numpy.array
        len_list = test_data_mat_dict['sentence_length']   ###(N), list
        pred_output = self._padding_mask(pred_output, len_list[:len(pred_output)])

        ## transform back to the dict form
        test_data_mat_dict['y_ent_matrix'] = pred_output
        result = data_loader.transform_back(test_data_mat_dict, data_type='ent')
        
        ## save the result
        if ISSAVE and result_dir:
            save_file = os.path.join(result_dir, 'predict.json')
            with open(save_file, 'w') as f:
                for data in result:
                    temps = json.dumps(data, ensure_ascii=False)
                    f.write(temps+'\n')
            log(f'save the predict result in {save_file}')
        return result

    @torch.no_grad()
    def eval_model(self, data_loader: KGDataLoader, data_set=None, hyper_param={}, use_cuda=False):
        '''
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  #默认64
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认./result WARNING:可能报错如果result目录不存在的话
        :return
            @precision_s, 
            @recall_s, 
            @f1_s
        '''
        def dict2str(d):
            ## 将entity 从字典形式转化为str形式方便比较
            res = d['entity']+':'+d['entity_type']+':'+str(d['entity_index']['begin'])+'-'+str(d['entity_index']['end'])
            return res

        def calculate_f1(pred_cnt, tar_cnt, correct_cnt):
            precision_s = round(correct_cnt / (pred_cnt + 1e-8), 3)
            recall_s = round(correct_cnt / (tar_cnt + 1e-8), 3)
            f1_s = round(2*precision_s*recall_s / (precision_s + recall_s + 1e-8), 3)
            return precision_s, recall_s, f1_s


        eva_data_set = data_loader.dataset.dev_dataset if data_set is None else data_set

        pred_result = self.predict(data_loader, eva_data_set, hyper_param, use_cuda) ###list(dict), 预测结果
        target = eva_data_set  ###list(dict)  AutoKGDataset, 真实结果

        pred_cnt = 0
        tar_cnt = 0
        correct_cnt = 0
        cnt_all = len(eva_data_set)
        log('Eval start')
        for idx in range(cnt_all):
            sentence = pred_result[idx]['input']
            pred_list = pred_result[idx]['entity_list']
            tar_list = target[idx]['output']['entity_list']
   
            str_pred_set = set(map(dict2str, pred_list))
            str_tar_set = set(map(dict2str, tar_list))
            common_set = str_pred_set.intersection(str_tar_set)

            pred_cnt += len(str_pred_set)
            tar_cnt += len(str_tar_set)
            correct_cnt += len(common_set)

            if (idx+1) % 1000 == 0:
                precision_s, recall_s, f1_s = calculate_f1(pred_cnt, tar_cnt, correct_cnt)
                log(f'[EVAL] step {idx+1}/{cnt_all} | precision: {precision_s} | recall: {recall_s} | f1 score: {f1_s}', 1)

        precision_s, recall_s, f1_s = calculate_f1(pred_cnt, tar_cnt, correct_cnt)
        print('='*100)
        log(f'[FINAL] | precision: {precision_s} | recall: {recall_s} | f1 score: {f1_s}', 0)
        print('='*100)
        return (precision_s, recall_s, f1_s)

    @staticmethod
    def _to_tensor(x, use_cuda=False):
        if use_cuda:
            return torch.tensor(x, dtype=torch.long).cuda()
        else:
            return torch.tensor(x, dtype=torch.long)     

    @staticmethod
    def _padding_mask(arr, lens):
        '''
        将超出句子长度部分的array mask成0
        :param
            @arr: (N, T), np.array
            @lens: (N), list
        :return 
            @arr: (N, T), np.array, arr after mask
        '''
        assert(len(arr) == len(lens)), 'len(arr) must be equal to len(lens), otherwise cannot mask'
        for idx in range(len(arr)):
            arr[idx, lens[idx]:] = 0
        return arr

    @staticmethod
    def _generate_mask(lens, max_len=None, use_cuda=False):
        '''
        返回一个mask, 遮住<pad>部分的无用信息.
        :param
            @lens: (batch_size), torch.tensor, the lengths of each sentence
            @max_len: int, the max length of the sentence - T
        :return 
            @mask: (batch_size, max_len)
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size = lens.shape[0]
        if max_len is None:
            max_len = lens.max()
        ranges = torch.arange(0, max_len).long()  #(max_len)
        if use_cuda:
            ranges = ranges.cuda()
        ranges = ranges.unsqueeze(0).expand(batch_size, max_len)   #(batch_size, max_len)
        lens_exp = lens.unsqueeze(1).expand_as(ranges)  #(batch_size, max_len)
        mask = ranges < lens_exp
        return mask