# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from dataloader3 import KGDataLoader3, Batch_Generator3
from utils import log

import transformers
import torch
from torch import nn
import torch.nn.init as I
# import torch.nn.functional as F
from torch.autograd import Variable

from utils import my_lr_lambda
from torch.optim.lr_scheduler import LambdaLR
from tricks import FocalLoss, EMA

import os
import json
import random
import numpy as np

import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

def show_time(cost, name):
    print(f'{name} -- {cost:.4f}')

def check_obj(y_obj, subids, data_list):
    def show_relation(relation):
        return f"{relation['relation']}--{relation['head']}--{relation['tail']}"
        # return f"{relation['relation']}--{relation['head']}--{relation['tail']}--{relation['tail_index']['begin']}"
    for i in range(len(data_list)):
        data_i = data_list[i]
        sentence_i = data_i['input']
        relations_i = data_i['output']['relation_list']
        r_s = list(map(show_relation, relations_i))
        print(sentence_i)
        print('origin:')
        print(r_s)

        sub_pick = subids[i]
        sub_i = (sub_pick[0], sub_pick[1])
        y_objr_list = mymodel._convert_relation_back(y_obj[i:i+1])
        case_i = [{sub_i:y_objr_list}]

        res = data_loader._obtain_sub_obj(case_i[0], sentence_i)
        res = [json.loads(i) for i in res]
        res_r_s = list(map(show_relation, res))
        print('decode:')
        print(res_r_s)
        print()

def check_sub(sub_pred, data_list):
    def show_relation(relation):
        return f"{relation['relation']}--{relation['head']}--{relation['tail']}"

    for i in range(len(x)):
        data_i = data_list[i]
        sentence_i = data_i['input']
        relations_i = data_i['output']['relation_list']
        r_s = list(map(show_relation, relations_i))
        # print(sentence_i)
        print(r_s)

        sub_pred_i = sub_pred[i]
        pred = []
        for sub in sub_pred_i:
            pred.append(sentence_i[sub[0]:sub[1]])
        print(pred)
        print()


class BERT_Hierarchical(nn.Module):
    def __init__(self, config={}, show_param=False):
        '''
        :param - dict
            param['embedding_dim']
            param['n_rels']
            param['use_cuda']
        '''
        super(BERT_Hierarchical, self).__init__()
        self.config = config
        self.embedding_dim = self.config.get('embedding_dim', 768)  ##(768)
        self.n_rels = self.config['n_rels']  ##(2)

        self.use_cuda = self.config['use_cuda']
        self.model_type = 'BERT_Hierarchical'

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log(f'embedding_dim: {self.embedding_dim}', 1)       
        log(f'n_rel_types: {self.n_rels}', 1) 
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        self.bert = transformers.BertModel.from_pretrained('bert-base-chinese')
        self.embed2sub = nn.Linear(self.embedding_dim, 2)
        self.embed2obj = nn.Linear(self.embedding_dim, 2*self.n_rels)

        # self.loss_fn = nn.BCELoss(reduction='none')
        self.weight = torch.tensor([1, 1], requires_grad=False).float()  ###(0:weight, 1: weight) (1, 10)  ##TODO:
        self.loss_fn = FocalLoss(alpha=self.weight, gamma=0, size_average=False)

    def reset_parameters(self):        
        I.xavier_normal_(self.embed2sub.weight.data)
        I.xavier_normal_(self.embed2obj.weight.data)

    def _get_bert_embedding(self, x, lens):
        use_cuda = self.use_cuda
        T = x.shape[1]

        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)
        att_mask = self._generate_mask(lens, max_len=T)
        embeds = self.bert(words_tensor, attention_mask=att_mask)[0]  #(batch_size, T, n_embed)
        return embeds

    def _get_subject_features(self, embeds, subids):
        '''
        :param   ###batch index select  -- torch.expand and torch.gather
            @embeds: bert embedding之后的word, (batch_size, T, n_embed), torch.tensor
            @subids: 选择的某一个subject的开始和结束idx (batch_size, K=2), torch.tensor
        :return 
            @sub_mean_features: (batch_size, n_embed) torch.tensor
        '''

        ### TODO:features 暂时使用subject开头和结尾两个单词的平均向量来替换，后面可以尝试整个subject的平均，但是那样的话就不能矩阵化运算了，因为长度不一样每个subject
        batch_size, T, n_embed = embeds.shape[0], embeds.shape[1], embeds.shape[2] 
        subids[:, 1] -= 1

        ##case1. the mean of start and end subject word embeddings
        subids_exp = subids.unsqueeze(2).expand(batch_size, subids.shape[1], n_embed)  ##(N, K, n_embed)
        sub_features = torch.gather(embeds, dim=1, index=subids_exp)   ##(N, K, n_embed)
        sub_mean_features = sub_features.mean(dim=1)  #(N, n_embed)

        ##case2. the mean of the total subject words embedding
        # res = []
        # for bid in range(batch_size):
        #     start, end = subids[bid, 0].item(), subids[bid, 1].item()
        #     # print(start, end)
        #     e_i = embeds[bid]
        #     # print(e_i[start:end+1].shape)
        #     test = torch.mean(e_i[start:end+1], dim=0)
        #     # print(test.shape)
        #     res.append(test)
        # sub_mean_features = torch.stack(res, dim=0)  ##(N, n_embed)
        # print(sub_mean_features.shape)

        return sub_mean_features

    def _loss(self, x, y_rel, lens):
        '''
        loss function: neg_log_likelihood
        :param
            @x: index之后的word, (batch_size, T), 每个字符按照字典对应到index, np.array
            @y_rel: list(dict), 关系对, {, 'sub2':objr_list, ..}
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (1), torch.tensor
        '''
        def random_pick(item): 
            return random.choice(list(item))

        use_cuda = self.use_cuda
        batch_size, max_length = x.shape[0], x.shape[1]
        lens_t = self._to_tensor(lens, use_cuda)

        ##y_sub target sequence 
        y_sub = self._convert_y_sub(y_rel, max_length)
        y_sub = self._to_tensor(y_sub, use_cuda).float()  ##(N, T, 2)
     
        ##batch中每个句子random pick 一个subject来做训练, y_obj target sequence
        subids = list(map(random_pick, y_rel))  ##(N, 2)， 2-start, end  

        y_obj = self._convert_y_obj(y_rel, subids, max_length, self.n_rels)   ##(N, T, 2*n_rels)
        y_obj = self._to_tensor(y_obj, use_cuda).float()

        ##根据随机sample出的subject来进行的编码，要和句子embedding重新组合，得到object decode 的输入
        subids = self._to_tensor(subids, use_cuda) ##(N, 2)
        embeds = self._get_bert_embedding(x, lens)  ###(N, T, n_embed)
        sub_features = self._get_subject_features(embeds, subids)  ##(N, n_embed)

        ## 预测出的obj sequence
        sub_features_exp = sub_features.unsqueeze(dim=1).expand(batch_size, max_length, embeds.shape[2])  ##(N, T, n_embed)
        obj_inputs = embeds + sub_features_exp
        obj_pred = torch.sigmoid(self.embed2obj(obj_inputs))  ##(N, T, 2*n_rels)
        # obj_pred = obj_pred**4  ##deal with the mismatch of positive and negative samples numbers

        ##预测出的sub sequence
        sub_pred = torch.sigmoid(self.embed2sub(embeds))  ###(N, T, 2)
        # sub_pred = sub_pred**2  ##deal with the mismatch of positive and negative samples numbers

        ## 计算loss
        mask = self._generate_mask(lens_t, max_length).float()  ##(N, T)

        sub_loss = self.loss_fn(sub_pred, y_sub)  ##(N, T, 2)
        sub_loss = sub_loss.mean(dim=2)  ##(N, T)
        sub_loss = (sub_loss*mask).sum()/mask.sum()

        obj_loss = self.loss_fn(obj_pred, y_obj)  ##(N, T, 2*n_rels)
        obj_loss = obj_loss.mean(dim=2)  ##(N, T)
        obj_loss = (obj_loss*mask).sum()/mask.sum()

        ##TODO: 尝试解决正负样本不均衡 -- 也可以用trick中的focalloss来解决
        # class_weight = torch.tensor([1, 10]).float()
        # weight = class_weight[y_sub.long()]  ##(N, T, 2)
        # sub_loss_fn = torch.nn.BCELoss(weight=weight, reduction='mean')
        # sub_loss = sub_loss_fn(sub_pred, y_sub.float())

        # weight = class_weight[y_obj.long()]  ##(N, T, 2*n_rels)
        # obj_loss_fn = torch.nn.BCELoss(weight=weight, reduction='mean')
        # obj_loss = obj_loss_fn(obj_pred, y_obj.float())
        return sub_loss + obj_loss

    def _output(self, x, lens, threshold=0.5):
        '''
        return the crf decode paths
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @[spoes], 其中spoes是一个字典，{'sub1': objr_list, 'sub2': objr_list, ...}
        '''

        use_cuda = self.use_cuda
        batch_size, max_length = x.shape[0], x.shape[1]
        assert batch_size == 1, 'for evaluation, the batch_size must be 1'
        lens_t = self._to_tensor(lens, use_cuda)

        embeds = self._get_bert_embedding(x, lens)  ###(1, T, n_embed)
        ##预测出的sub sequence
        sub_pred = torch.sigmoid(self.embed2sub(embeds))  ###(1, T, 2)
        sub_pred_int = (sub_pred>=threshold).int()  ##(1, T, 2)
        sub_list = self._convert_entity_back(sub_pred_int, lens)[0]   ##list --> [(start1, end1), (start2, end2), ...]

        spoes = {}
        for start, end in sub_list[:4]:  ##只取前五个subject
            sub_i = (start, end)

            subids = [[start, end]]
            subids = self._to_tensor(subids, use_cuda) ##(1, 2)
            sub_features = self._get_subject_features(embeds, subids)  ##(1, n_embed)
            # print('sub_features:', sub_features.shape)

            sub_features_exp = sub_features.unsqueeze(dim=1).expand(batch_size, max_length, embeds.shape[2])  ##(1, T, n_embed)
            # print('sub_features_exp:', sub_features_exp.shape)

            obj_inputs = embeds + sub_features_exp
            obj_pred = torch.sigmoid(self.embed2obj(obj_inputs))  ##(1, T, 2*n_rels)     
            obj_pred_int = (obj_pred>=threshold).int()  ##(1, T, 2*n_rels)
            # print('obj_pred_int', obj_pred_int.shape)  

            objr_list = self._convert_relation_back(obj_pred_int, lens)
            if len(objr_list) > 0:
                ##删除主语和宾语相同的情况
                new_objr_list = [objr for objr in objr_list if objr[0] != sub_i[0]]
                spoes[sub_i] = new_objr_list
            
        return [spoes]

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

    def backup_param(self):
        '''
        for restore the parameters in learning epochs
        '''
        backup_p = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                backup_p[name] = param.data.clone()
        print('back_up successfully')
        return backup_p

    def restore_param(self, backup_p):
        for name, param in self.named_parameters():
            if param.requires_grad:
                assert name in backup_p
                param.data = backup_p[name]
        print('restore successfully')

    def train_model(self, data_loader, train_dataset=None, eval_dataset=None, hyper_param={}, rebuild=False):
        '''
        :param
            @data_loader: (KGDataLoader),
            @result_dir: (str) path to save the trained model and extracted dictionary
            @hyper_param: (dict)
                @hyper_param['EPOCH']
                @hyper_param['batch_size']
                @hyper_param['learning_rate_upper']
                @hyper_param['learning_rate_bert']
                @hyper_param['bert_finetune']
                @hyper_param['visualize_length']   #num of batches between two check points
                @hyper_param['isshuffle']
                @hyper_param['result_dir']
                @hyper_param['model_name']
        :return
            @loss_record, 
            @score_record
        '''
        use_cuda = self.use_cuda
        use_ema = True
        use_grad_clip = True
        ema = EMA(self, mu=0.99) if use_ema else None

        if use_cuda:
            print('use cuda=========================')
            self.cuda()

        if use_ema:
            ema.register()
            # print('origin bias', self.embed2sub.bias)
    

        EPOCH = hyper_param.get('EPOCH', 3)
        BATCH_SIZE = hyper_param.get('batch_size', 4)
        LEARNING_RATE_upper = hyper_param.get('learning_rate_upper', 1e-3)
        LEARNING_RATE_bert = hyper_param.get('learning_rate_bert', 5e-5)
        bert_finetune = hyper_param.get('bert_finetune', True)
        visualize_length = hyper_param.get('visualize_length', 10)
        result_dir = hyper_param.get('result_dir', './result/')
        model_name = hyper_param.get('model_name', 'model.p')
        is_shuffle = hyper_param.get('isshuffle', True)
        DATA_TYPE = 'rel'
        
        train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset
        if rebuild:
            train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        else:
            old_train_dict_path = os.path.join(result_dir, 'train_data_mat_dict.pkl')
            if os.path.exists(old_train_dict_path):
                train_data_mat_dict = data_loader.load_preprocessed_data(old_train_dict_path)
                log('Reload preprocessed data successfully~')
            else:
                train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
                data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        ##TODO: 根据subject的平均数目以及句子的平均长度来设置self.weight的权重
        avg_length = data_loader.metadata_['avg_sen_len']
        avg_sub = train_data_mat_dict['total_sub']/len(train_data_mat_dict['y_rel_list'])
        weight_1 = round(avg_length/avg_sub/10, 2)
        self.weight[1] = weight_1
        print(self.weight)
        # print('total sub:', train_data_mat_dict['total_sub'])
        # print('total sentence: ', len(train_data_mat_dict['y_rel_list']))
        # print('avg length:', data_loader.metadata_['avg_sen_len'])

        data_generator = Batch_Generator3(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)

        all_param = list(self.named_parameters()) 
        bert_param = [p for n, p in all_param if 'bert' in n]
        other_param = [p for n, p in all_param if 'bert' not in n]

        if bert_finetune:
            optimizer_group_paramters = [
                {'params': other_param, 'lr': LEARNING_RATE_upper}, 
                {'params': bert_param, 'lr': LEARNING_RATE_bert}
            ]
            optimizer = torch.optim.Adam(optimizer_group_paramters)
            log(f'****BERT_finetune, learning_rate_upper: {LEARNING_RATE_upper}, learning_rate_bert: {LEARNING_RATE_bert}', 0)
        else:
            optimizer = torch.optim.Adam(other_param, lr=LEARNING_RATE_upper)
            log(f'****BERT_fix, learning_rate_upper: {LEARNING_RATE_upper}', 0)
        
        ##TODO:
        scheduler = LambdaLR(optimizer, lr_lambda=my_lr_lambda)
        # scheduler = transformers.optimization.get_cosine_schedule_with warmup(optimizer, num_warmup_steps=int(EPOCH*0.2), num_training_steps=EPOCH)
        
        all_cnt = len(train_data_mat_dict['cha_matrix'])
        log(f'{model_name} Training start!', 0)
        loss_record = []
        score_record = []
        max_score = 0

        evel_param = {'batch_size':1, 'issave':False, 'result_dir': result_dir}
        for epoch in range(EPOCH):
            self.train()

            log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
            loss = 0.0

            ##备份当前epoch训练之前的model和ema中的参数，用来回滚
            temp_param = self.backup_param()
            if use_ema:
                ema.backup_oldema()
            print('before train bias', self.embed2sub.bias)
            print('before ema bias', list(ema.shadow.values())[200], list(ema.shadow.keys())[200])
            print(optimizer.state_dict()['param_groups'][0]['lr'], optimizer.state_dict()['param_groups'][1]['lr'])

            for cnt, data_batch in enumerate(data_generator):

                x, pos, y_rel, y_ent, lens, data_list = data_batch

                loss_avg = self._loss(x, y_rel, lens)

                optimizer.zero_grad()
                loss_avg.backward()

                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()

                if use_ema:
                    ema.update()
                    # print('after update bias', self.embed2sub.bias)

                loss += loss_avg
                if use_cuda:
                    loss_record.append(loss_avg.cpu().item())
                else:
                    loss_record.append(loss_avg.item())

                if (cnt+1) % visualize_length == 0:
                    loss_cur = loss / visualize_length
                    log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
                    loss = 0.0

            if use_ema:
                ema.apply_shadow()
                # print('after apply ema shadow bias', self.embed2sub.bias)

            temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=evel_param)
            score_record.append(temp_score)
            # scheduler.step()   #TODO:
            
            if temp_score[2] > max_score:
                max_score = temp_score[2]
                save_path = os.path.join(result_dir, model_name)
                self.save_model(save_path)
                print(f'Checkpoint saved successfully, current best score is {max_score}')

                if use_ema:
                    ema.restore()
                    # print('restore bias', self.embed2sub.bias)
            elif temp_score[2] < max_score:
                ###回滚到这个epoch之前的参数
                self.restore_param(temp_param)
                ema.return_oldema()
                scheduler.step()
                print(optimizer.state_dict()['param_groups'][0]['lr'], optimizer.state_dict()['param_groups'][1]['lr'])

                if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-4:
                    print('early stop!!!')
                    break
            else:
                if use_ema:
                    ema.restore()
                
            
        log(f'the best score of the model is {max_score}')
        return loss_record, score_record

    @torch.no_grad()
    def predict(self, data_loader, data_set=None, hyper_param={}, rebuild=False):
        '''
        预测出 test_data_mat_dict['y_rel_matrix']中的内容，重新填写进该matrix, 未预测之前都是0
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  ##默认1  no need, must be one
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认None
        :return
            @result: list, len(句子个数)
                case = result[0]
                case['input']
                case['relation_list']
                    r = case['relation_list'][0]
                    r['relation']: 成立日期
                    r['head']: '百度'
                    r['tail']: '2016年04月08日'
        '''
        use_cuda = self.use_cuda
        if use_cuda:
            print('use cuda=========================')
            self.cuda()

        # BATCH_SIZE = hyper_param.get('batch_size', 64)
        BATCH_SIZE = 1
        ISSAVE = hyper_param.get('issave', False)
        result_dir = hyper_param.get('result_dir', './result/')
        DATA_TYPE = 'rel'

        test_dataset = data_loader.dataset.test_dataset if data_set is None else data_set
        if rebuild:
            test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        else:
            old_test_dict_path = os.path.join(result_dir, 'test_data_mat_dict.pkl')
            if os.path.exists(old_test_dict_path):
                test_data_mat_dict = data_loader.load_preprocessed_data(old_test_dict_path)
                log('Reload preprocessed data successfully~')
            else:
                test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
                data_loader.save_preprocessed_data(old_test_dict_path, test_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        data_generator = Batch_Generator3(test_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=False)
        self.eval()   #disable dropout layer and the bn layer

        
        all_cnt = len(test_data_mat_dict['cha_matrix'])
        log(f'Predict start!', 0)
        total_pre_rel = []
        for cnt, data_batch in enumerate(data_generator):
            x, pos, _, _, lens, data_list = data_batch
            pre_rel = self._output(x, lens, threshold=0.6)  ##TODO:
            total_pre_rel += pre_rel

            if (cnt+1) % 200 == 0:
                log(f'[PREDICT] step {(cnt+1)*BATCH_SIZE}/{all_cnt}', 1)


        test_data_mat_dict['y_rel_list'] = total_pre_rel

        result = data_loader.transform_back(test_data_mat_dict, data_type=DATA_TYPE)

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
    def eval_model(self, data_loader, data_set=None, hyper_param={}, rebuild=False):
        '''
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  #默认1
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认./result WARNING:可能报错如果result目录不存在的话
        :return
            @precision_s, 
            @recall_s, 
            @f1_s
        '''
        use_cuda = self.use_cuda
        if use_cuda:
            print('use cuda=========================')
            self.cuda()

        def dict2str(d):
            ## 将entity 从字典形式转化为str形式方便比较
            # res = d['entity']+':'+d['entity_type']+':'+str(d['entity_index']['begin'])+'-'+str(d['entity_index']['end'])
            ## 将relation 从字典形式转化为str形式方便比较
            res = d['relation']+'-'+d['head']+'-'+d['tail']
            return res

        def calculate_f1(pred_cnt, tar_cnt, correct_cnt):
            precision_s = round(correct_cnt / (pred_cnt + 1e-8), 3)
            recall_s = round(correct_cnt / (tar_cnt + 1e-8), 3)
            f1_s = round(2*precision_s*recall_s / (precision_s + recall_s + 1e-8), 3)
            return precision_s, recall_s, f1_s


        eva_data_set = data_loader.dataset.dev_dataset if data_set is None else data_set

        pred_result = self.predict(data_loader, eva_data_set, hyper_param, rebuild=rebuild) ###list(dict), 预测结果 len=n_sentence
        target = eva_data_set  ###list(dict)  AutoKGDataset, 真实结果

        pred_cnt = 0
        tar_cnt = 0
        correct_cnt = 0
        cnt_all = len(eva_data_set)
        log('Eval start')
        for idx in range(cnt_all):
            sentence = pred_result[idx]['input']
            pred_list = pred_result[idx]['relation_list']
            tar_list = target[idx]['output']['relation_list']
   
            str_pred_set = set(map(dict2str, pred_list))
            str_tar_set = set(map(dict2str, tar_list))
            common_set = str_pred_set.intersection(str_tar_set)
            # print('target:')
            # print(str_tar_set)
            # print('predict:')
            # print(str_pred_set)

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

    def _convert_relation_back(self, obj_pred_int, lens):
        ''' 
        :param
            @obj_pred_int: (1, T, 2*n_rel)  torch.tensor,batch_size must be one, and the correspond object according to the subject
            @n_rels: number of relation types
            @lens: list, (batch_size)
        :return
            @objr_list: list(tuple) the objr_list corresponding to the input subi
        '''
        objr_list = []
        assert len(obj_pred_int) == 1, 'for evaluation, the batch_size must be 1'

        for ridx in range(self.n_rels):
            obj_pred_int_case = obj_pred_int[:, :, 2*ridx:2*(ridx+1)]  ##(1, T, 2)
            obj_case_list = self._convert_entity_back(obj_pred_int_case, lens)[0]

            for obj in obj_case_list[:4]:
                obj_i = (obj[0], obj[1], ridx+1)  ##(obj_start, obj_end, relation_type)
                objr_list.append(obj_i)
        return objr_list

    @staticmethod
    def _convert_entity_back(sub_pred_int, lens):
        ''' end 是单词结尾后一个单词的idx
        :param
            @sub_pred_int, torch.tensor (N, T, 2)
            @lens, list, (N)
        :return
            sub_list: list(list), len= (N, n_entities)
        '''
        batch_size = sub_pred_int.shape[0]
        sub_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            s_list = sub_pred_int[i, :, 0]  #(T)
            e_list = sub_pred_int[i, :, 1]  #(T)
            len_i = lens[i]
            for k in range(len_i):
                if s_list[k] > 0.5:
                    for w in range(k, len_i): ###len(s_list) 会超出句子范围
                        if e_list[w] > 0.5:
                            # if k >= w+1:
                                # print('exist subject that k >= w+1')
                            sub_list[i].append((k, w+1))
                            break

        return sub_list
            
    @staticmethod
    def _convert_y_sub(y_rels, T):
        '''
        @y_rels: list(dict), 关系对, {, 'sub1':objr_list, ..}  sub1--(start, end), objr_list--(start, end, ridx)
        return:
            @y_sub_matrix: (N, T, 2), np.array
        '''
        y_sub_list = []
        for rel in y_rels:
            y_sub = np.zeros([T, 2])
            for start, end in rel.keys():
                if start < T:
                    y_sub[start, 0] = 1
                if end-1 < T:
                    y_sub[end-1, 1] = 1
            y_sub_list.append(np.expand_dims(y_sub, axis=0))
        y_sub_matrix = np.vstack(y_sub_list)
        # print(y_sub_matrix.shape)
        return y_sub_matrix

    @staticmethod
    def _convert_y_obj(y_rels, subids, T, n_rels):
        '''
        :param:
            @y_rels: list(dict), 关系对, {, 'sub1':objr_list, ..}  sub1--(start, end), objr_list--[(start, end, ridx), (start, end, ridx)]
            @subids: list(tuple) (N, 2) -- [(start, end), (start, end), ...]
            @T: max_length
            @n_rels: total number of relation types
        :return
            @y_obj_matrix: (N, T, 2*n_rels), np.array
        '''

        y_obj_list = []
        for i in range(len(y_rels)):
            objr_list = y_rels[i][subids[i]]
            y_obj = np.zeros([T, 2*n_rels])
            for objr in objr_list:
                obj_b, obj_e, ridx = objr
                if obj_b < T:
                    y_obj[obj_b, 2*(ridx-1)] = 1
                if obj_e-1 < T:
                    y_obj[obj_e-1, 2*(ridx-1)+1] = 1
            y_obj_list.append(np.expand_dims(y_obj, axis=0))
        y_obj_matrix = np.vstack(y_obj_list)  ###(N, T, 2*n_rels)

        return y_obj_matrix

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
    def _generate_mask(lens, max_len=None):
        '''
        返回一个mask, 遮住<pad>部分的无用信息.
        :param
            @lens: (batch_size), torch.tensor, the lengths of each sentence
            @max_len: int, the max length of the sentence - T
        :return 
            @mask: (batch_size, max_len)
        '''
        # use_cuda = self.use_cuda if use_cuda is None else use_cuda
        assert len(lens)>0, 'lens must be a real list'

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


if __name__ == '__main__':


    ###===========================================================
    ###试训练
    ###===========================================================
    data_set = AutoKGDataset('./data/newdata2/d10')
    train_dataset = data_set.train_dataset
    eval_dataset = data_set.dev_dataset
    # train_dataset = data_set.train_dataset
    # eval_dataset = data_set.dev_dataset

    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader3(data_set, rebuild=False, temp_dir='result/')

    model_config = {
        'embedding_dim' : 768,
        'n_rels': len(data_loader.relation_type_dict),
        'use_cuda':1,
        'dropout_prob': 0,
    }
    mymodel = BERT_Hierarchical(model_config, show_param=True)

    train_param = {
        'EPOCH': 0,         
        'batch_size': 16,    
        'learning_rate_bert': 5e-5,
        'learning_rate_upper': 1e-3, 
        'bert_finetune': True,
        'visualize_length': 20,
        'isshuffle': True,
        'result_dir': './result',
        'model_name':'model_test.p'
    }
    mymodel.train_model(data_loader, train_dataset, eval_dataset, train_param)

    eval_param = {        
        'batch_size': 1,    
        'issave': True,
        'result_dir': './result'
    }
    mymodel.predict(data_loader, eval_dataset, eval_param, rebuild=False)

    ###===========================================================
    ###试训练 -- 参数细节
    ###===========================================================    
    # train_data_mat_dict = data_loader.transform(train_dataset, istest=False, data_type='rel')
    # data_generator = Batch_Generator3(train_data_mat_dict, batch_size=4, data_type='rel', isshuffle=True)


    # def random_pick(item):
    #     return random.choice(list(item))
    # def show_relation(relation):
    #     return f"{relation['relation']}--{relation['head']}--{relation['tail']}"

    # for epoch in range(1):
    #     print('EPOCH: %d' % epoch)
    #     for data_batch in data_generator:
    #         x, pos, y_rel, y_ent, lens, data_list = data_batch

    #         ###======================for BERT-Hierarchical-MODEL only==================================
    #         # mymodel._loss(x, y_rel, lens)
    #         # for batch_idx in range(len(x)):
                
    #         #     subids = list(map(random_pick, y_rel[batch_idx:1+batch_idx]))
    #         #     print(subids)
    #         #     print(data_list[batch_idx])
    #             # output = mymodel._output(x[batch_idx:1+batch_idx], lens[batch_idx:1+batch_idx])
    #             # res = data_loader._obtain_sub_obj(output[0], data_list[0]['input'])
    #             # res = [json.loads(i) for i in res]
    #             # res_r_s = list(map(show_relation, res))
    #             # print(res_r_s)
            
    #         # test = data_loader._obtain_sub_obj(output[0], sentence)
    #         # print(test)

    #         # y_sub = mymodel._convert_y_sub(y_rel, x.shape[1])  ##(N, T, 2)
    #         # sub_pred = mymodel._convert_entity_back(y_sub)  ##(list(list)), N
    #         # # check_sub(sub_pred, data_list)

    #         # subids = list(map(random_pick, y_rel))  ##(N, 2)， 2-start, end
    #         # print(subids)
    #         # y_obj = mymodel._convert_y_obj(y_rel, subids, x.shape[1], mymodel.n_rels)  ##(N, T, 2*n_type)
    #         # check_obj(y_obj, subids, data_list)

    #         # subids = list(map(random_pick, y_rel))  ##(N, 2)， 2-start, end
    #         # subids = mymodel._to_tensor(subids, use_cuda=0) ##(N, 2)
    #         # embeds = mymodel._get_bert_embedding(x, lens)  ###(N, T, n_embed)
    #         # # print(subids.shape)
    #         # # print(embeds.shape)
    #         # sub_features = mymodel._get_subject_features(embeds, subids)
    #         # print(sub_features.shape)

    #         # break
    #     break