# !-*- coding:utf-8 -*-
from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log

import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from model import MODEL_TEMP
from torchcrf import CRF

torch.manual_seed(1)

class BASELINE(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
        '''
        :param - dict
            param['n_tags']
            param['n_words']
            param['use_cuda']
            param['dropout_prob']
            param['lstm_layer_num']
        '''
        super(BASELINE, self).__init__()
        self.config = config
        # self.embedding_dim = self.config.get('embedding_dim', 64)
        self.embedding_dim = 768  #TODO:
        # self.hidden_dim = self.config.get('hidden_dim', 128*2)
        self.hidden_dim = 64  #TODO: 256, 64
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.n_tags = self.config['n_ent_tags'] - 2
        self.n_words = self.config['n_words']

        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.lstm_layer_num = self.config.get('lstm_layer_num', 1)

        self.use_cuda = self.config.get('use_cuda', False)
        self.model_type = 'BASELINE'

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
        log(f'n_ent_tags: {self.n_tags}', 1)
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

    def _loss(self, x, y_ent, lens, use_cuda=None):
        '''
        loss function: neg_log_likelihood
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (1), torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]

        logits = self._get_lstm_features(x)   ##(batch_size, T, n_tags)
        tensor_y_ent = self._to_tensor(y_ent, use_cuda)

        lens = self._to_tensor(lens, use_cuda)
        len_mask = self._generate_mask(lens, max_len=T)  ##(batch_size, T)

        log_likelihood_ent = self.crf(emissions=logits, tags=tensor_y_ent, mask=len_mask, reduction='mean')
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
    
        paths = self.crf.decode(logits)
        paths = self._to_tensor(paths, use_cuda)
        return paths

if __name__ == '__main__':
    model_params = {
        # 'embedding_dim' : 768,
        # 'hidden_dim' : 64,       
        'n_ent_tags' : len(data_loader.ent_seq_map_dict),  
        'n_rel_tags' : len(data_loader.rel_seq_map_dict),  
        'n_rels' : len(data_loader.label_location_dict)+1,
        'n_words' : len(data_loader.character_location_dict),
        'use_cuda':args.use_cuda,
        'dropout_prob': 0,
        'lstm_layer_num': 1
    }
    
    mymodel = BASELINE(config=model_config, show_param=True) 

    ###===========================================================
    ###试训练
    ###===========================================================
    # data_set = AutoKGDataset('./d1/')
    # # train_dataset = data_set.train_dataset[:20]
    # # eval_dataset = data_set.dev_dataset[:10]
    # train_dataset = data_set.train_dataset
    # eval_dataset = data_set.dev_dataset
    # os.makedirs('result', exist_ok=True)
    # data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')
    # print(data_loader.embedding_info_dicts['entity_type_dict'])

    # train_param = {
    #     'EPOCH': 1,         #45
    #     'batch_size': 4,    #512
    #     'learning_rate_bert': 5e-5,
    #     'learning_rate_upper': 5e-3,
    #     'bert_finetune': False,
    #     'visualize_length': 2, #10
    #     'isshuffle': True,
    #     'result_dir': './result/',
    #     'model_name':'model_test.p'
    # }

    # mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)

    # eval_param = {
    #     'batch_size':100, 
    #     'issave':False, 
    #     'result_dir': './result/'
    # }
    # mymodel.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_param)
