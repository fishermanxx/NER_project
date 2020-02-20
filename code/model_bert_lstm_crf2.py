# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import transformers
import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

# from utils import my_lr_lambda
# from torch.optim.lr_scheduler import LambdaLR

import os

from model import MODEL_TEMP
# from model_crf import CRF
from torchcrf import CRF

torch.manual_seed(1)



class BERT_LSTM_CRF2(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
        '''
        :param - dict
            param['embedding_dim']
            param['hidden_dim']
            param['n_ent_tags']
            param['n_rel_tags']
            param['n_words']
            param['start_ent_idx']  int, <start> tag index for entity tag seq
            param['end_ent_idx']   int, <end> tag index for entity tag seq
            param['start_rel_idx']
            param['end_rel_idx']
            param['use_cuda']
            param['dropout_prob']
            param['lstm_layer_num']
        '''
        super(BERT_LSTM_CRF2, self).__init__()
        self.config = config
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.hidden_dim = self.config.get('hidden_dim', 64)  ##TODO: 128*2
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.n_tags = self.config.get('n_ent_tags', 45) - 2
        # self.n_words = self.config.get('n_words', 10000)

        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.lstm_layer_num = self.config.get('lstm_layer_num', 1)

        self.use_cuda = self.config.get('use_cuda', 0)
        self.model_type = 'BERT_LSTM_CRF2'

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
        log(f'n_ent_tags: {self.n_tags}', 1)
        log(f"crf_start_idx: {self.config['start_ent_idx']}", 1)
        log(f"crf_end_idx: {self.config['end_ent_idx']}", 1)
        log(f'lstm_layer_num: {self.lstm_layer_num}', 1)
        log(f'dropout_prob: {self.dropout_prob}', 1)  
        log('='*80, 0)      

    def build_model(self):
        '''
        build the bert layer, lstm layer and CRF layer
        '''
        # self.word_embeds = nn.Embedding(self.n_words, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, batch_first=True, num_layers=self.lstm_layer_num, dropout=self.dropout_prob, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.n_tags)

        self.crf = CRF(self.n_tags, batch_first=True)
        self.bert = transformers.BertModel.from_pretrained('bert-base-chinese')

    def reset_parameters(self):        
        # I.xavier_normal_(self.word_embeds.weight.data)
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

        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)
        att_mask = self._generate_mask(lens, max_len=T)
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
        T = x.shape[1]

        logits = self._get_lstm_features(x, lens)   ##(batch_size, T, n_tags)
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
        logits = self._get_lstm_features(x, lens, use_cuda)

        lens = self._to_tensor(lens, use_cuda)
        len_mask = self._generate_mask(lens, max_len=T)  ##(batch_size, T)
    
        # paths = self.crf.decode(logits, len_mask)
        paths = self.crf.decode(logits)
        paths = self._to_tensor(paths, use_cuda)
        return paths

if __name__ == '__main__':
    pass