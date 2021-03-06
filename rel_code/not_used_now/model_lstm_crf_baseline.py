# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

from dataloader import KGDataLoader2

import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

# from utils import my_lr_lambda
# from torch.optim.lr_scheduler import LambdaLR

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from model import MODEL_TEMP
# from model_crf import CRF
from torchcrf import CRF
torch.manual_seed(1)


class BASELINE(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
        '''
        :param - dict
            *param['embedding_dim']
            *param['hidden_dim']
            param['n_ent_tags']
            param['n_rel_tags']
            param['n_rels']
            param['n_words']
            *param['start_ent_idx']  int, <start> tag index for entity tag seq
            *param['end_ent_idx']   int, <end> tag index for entity tag seq
            *param['start_rel_idx']  int, <start> tag index for entity tag seq
            *param['end_rel_idx']   int, <end> tag index for entity tag seq
            param['use_cuda']
            param['dropout_prob']
            param['lstm_layer_num']
        '''
        super(BASELINE, self).__init__()
        self.config = config

        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.hidden_dim = self.config.get('hidden_dim', 64)
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.lstm_layer_num = self.config.get('lstm_layer_num', 1)
        self.use_cuda = self.config.get('use_cuda', False)

        self.model_type = 'BASELINE'
        self.n_tags = self.config['n_rel_tags'] - 2
        self.n_words = self.config['n_words']

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log(f'embedding_dim: {self.embedding_dim}', 1)
        log(f'hidden_dim: {self.hidden_dim}', 1)
        log(f'lstm_layer_num: {self.lstm_layer_num}', 1)
        log(f'dropout_prob: {self.dropout_prob}', 1)  
        log(f'n_rel_tags: {self.n_tags}', 1)
        # log(f"crf_start_idx: {self.config['start_ent_idx']}", 1)
        # log(f"crf_end_idx: {self.config['end_ent_idx']}", 1)
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        self.word_embeds = nn.Embedding(self.n_words, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, batch_first=True, num_layers=self.lstm_layer_num, dropout=self.dropout_prob, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.n_tags)
        self.crf = CRF(self.n_tags, batch_first=True)

    def reset_parameters(self):        
        I.xavier_normal_(self.word_embeds.weight.data)
        self.lstm.reset_parameters()
        # stdv = 1.0 / math.sqrt(self.hidden_dim)
        # for weight in self.lstm.parameters():
        #     I.uniform_(weight, -stdv, stdv)
        I.xavier_normal_(self.hidden2tag.weight.data)
        self.crf.reset_parameters()
        
    def _get_lstm_features(self, x, use_cuda=None):
        '''
        :param  
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
        :return 
            @lstm_feature: (batch_size, T, n_tags) -- 类似于eject score, torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size = x.shape[0]

        ##embedding layer
        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        embeds = self.word_embeds(words_tensor)  #(batch_size, T, n_embed)

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

    def _loss(self, x, y_rel, lens, use_cuda=None):
        '''
        loss function: neg_log_likelihood
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @y_rel: (batch_size, T), np.array, index之后的rel_with_ent seq, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (1), torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]

        logits = self._get_lstm_features(x)   ##(batch_size, T, n_tags)
        tensor_y_rel = self._to_tensor(y_rel, use_cuda)

        lens = self._to_tensor(lens, use_cuda)
        len_mask = self._generate_mask(lens, max_len=T)  ##(batch_size, T)

        log_likelihood_ent = self.crf(emissions=logits, tags=tensor_y_rel, mask=len_mask, reduction='mean')

        return - log_likelihood_ent

    def _output(self, x, lens, use_cuda=None):
        '''
        return the crf decode paths
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @paths: (batch_size, T), torch.tensor, 最佳句子路径
            @scores: (batch_size), torch.tensor, 最佳句子路径上的得分
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]
        logits = self._get_lstm_features(x, use_cuda)

        lens = self._to_tensor(lens, use_cuda)
        len_mask = self._generate_mask(lens, max_len=T)  ##(batch_size, T)
    
        # paths = self.crf.decode(logits, len_mask)  ##will return with sequence of different length
        paths = self.crf.decode(logits)
        paths = self._to_tensor(paths, use_cuda)
        return paths


if __name__ == '__main__':
    # model_config = {
    #     'embedding_dim' : 128,
    #     'hidden_dim' : 64,
    #     'n_ent_tags' : 45,
    #     'n_rel_tags' : 15,
    #     'n_words' : 22118,
    #     # 'start_ent_idx': 43
    #     # 'end_ent_idx': 44
    #     # 'start_rel_idx': 13
    #     # 'end_rel_idx': 14
    #     'use_cuda':False,
    #     'dropout_prob': 0,
    #     'lstm_layer_num': 1,
    # }

    # mymodel = BASELINE(config=model_config, show_param=True) 


    ###===========================================================
    ###模型参数测试
    ###===========================================================

    ##case1:
    # all_param = list(mymodel.named_parameters()) 
    # bert_param = [(n, p) for n, p in all_param if 'bert' in n]
    # other_param = [(n, p) for n, p in all_param if 'bert' not in n]
    # print(f'all_param: {len(all_param)}')
    # print(f'bert_param: {len(bert_param)}')
    # print(f'other_param: {len(other_param)}')
    # for n, p in other_param:
    #     print(n, p.shape)

    # ##case2:
    # print('='*80)
    # all_param = list(mymodel.named_parameters())
    # crf_param = list(mymodel.crf.named_parameters())
    # lstm_param = list(mymodel.lstm.named_parameters())
    # fc_param = list(mymodel.hidden2tag.named_parameters())
    # # bert_param = list(mymodel.bert.named_parameters())
    # emb_param = list(mymodel.word_embeds.named_parameters())

    # # print(f'emb_param: {len(emb_param)}')
    # print(f'crf_param: {len(crf_param)}')
    # print(f'lstm_param: {len(lstm_param)}')
    # print(f'fc_param: {len(fc_param)}')
    # print(f'emb_param: {len(emb_param)}')
    # print('='*50)
    # print(f'all_param: {len(all_param)}')
    # # print(len(all_param))

    # other_param = crf_param + fc_param + lstm_param + emb_param
    # for n, p in other_param:
    #     print(n, p.shape)

    # no_decay = ['bias', 'reverse']
    # # choose_param = [p for n, p in lstm_param if not any(nd in n for nd in no_decay)]
    # # choose_param = [np for np in lstm_param if not any(nd in np[0] for nd in no_decay)]
    # # choose_param = [np for np in lstm_param if 'bias' in np[0]]
    # choose_param = [np for np in lstm_param if any(nd in np[0] for nd in no_decay)]
    # for name, param in lstm_param:
    #     print(name, param.shape, id(param))

    # print('='*50)
    # for name, param in choose_param:
    #     print(name, param.shape, id(param))


    ###===========================================================
    ###试训练
    ###===========================================================
    data_set = AutoKGDataset('./data/d4/')
    train_dataset = data_set.train_dataset[:200]
    eval_dataset = data_set.dev_dataset[:100]
    # train_dataset = data_set.train_dataset
    # eval_dataset = data_set.dev_dataset
    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader2(data_set, rebuild=False, temp_dir='result/')
    # print(data_loader.embedding_info_dicts['entity_type_dict'])
    print(list(data_loader.rel_seq_map_dict))

    model_config = {
        # 'embedding_dim' : 128,
        # 'hidden_dim' : 64,
        'n_ent_tags' : len(data_loader.ent_seq_map_dict),  
        'n_rel_tags' : len(data_loader.rel_seq_map_dict),  
        'n_rels' : len(data_loader.relation_type_dict),
        'n_words' : len(data_loader.character_location_dict),
        # 'start_ent_idx': 43
        # 'end_ent_idx': 44
        # 'start_rel_idx': 13
        # 'end_rel_idx': 14
        'use_cuda':False,
        'dropout_prob': 0,
        'lstm_layer_num': 1,
    }

    mymodel = BASELINE(config=model_config, show_param=True) 

    train_param = {
        'EPOCH': 1,         #45
        'batch_size': 32,    #512
        'learning_rate_bert': 5e-5,
        'learning_rate_upper': 5e-3,
        'bert_finetune': False,
        'visualize_length': 2, #10
        'isshuffle': True,
        'result_dir': './result/',
        'model_name':'model_test.p'
    }
    mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)

    eval_param = {
        'batch_size':100, 
        'issave':False, 
        'result_dir': './result/'
    }
    mymodel.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_param)
