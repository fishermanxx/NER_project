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
from model_crf import CRF

torch.manual_seed(1)



class BERT_LSTM_CRF(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
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
        self.config = config
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.hidden_dim = self.config.get('hidden_dim', 64)
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.n_tags = self.config.get('n_tags', 45)
        # self.n_words = self.config.get('n_words', 10000)

        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.lstm_layer_num = self.config.get('lstm_layer_num', 1)

        self.use_cuda = self.config.get('use_cuda', 0)
        self.model_type = 'BERT_LSTM_CRF'

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'embedding_dim: {self.embedding_dim}', 1)
        log(f'hidden_dim: {self.hidden_dim}', 1)
        log(f'num_labels: {self.n_tags}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
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
        self.crf = CRF(self.config)
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

        logits = self._get_lstm_features(x, lens)
        log_norm_score = self.crf.log_norm_score(logits, lens)
        path_score = self.crf.path_score(logits, y_ent, lens)

        loss = log_norm_score - path_score
        loss = (loss/self._to_tensor(lens, use_cuda)).mean()
        return loss

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
        # self.eval()
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        logits = self._get_lstm_features(x, lens, use_cuda)
        scores, paths = self.crf.viterbi_decode(logits, lens, use_cuda)
        return paths


if __name__ == '__main__':
    model_config = {
        'embedding_dim' : 768,
        'hidden_dim' : 64,
        'n_tags' : 45,
        'n_words' : 22118,
        'start_idx': 43,  ## <start> tag index for entity tag seq
        'end_idx': 44,  ## <end> tag index for entity tag seq
        'use_cuda':True,
        'dropout_prob': 0,
        'lstm_layer_num': 1,
        'num_labels': 45
    }
    
    mymodel = BERT_LSTM_CRF(model_config, show_param=True) 

    ###===========================================================
    ###模型参数测试
    ###===========================================================

    ##case1:
    all_param = list(mymodel.named_parameters()) 
    bert_param = [(n, p) for n, p in all_param if 'bert' in n]
    other_param = [(n, p) for n, p in all_param if 'bert' not in n]
    print(f'all_param: {len(all_param)}')
    print(f'bert_param: {len(bert_param)}')
    print(f'other_param: {len(other_param)}')
    for n, p in other_param:
        print(n, p.shape)

    ##case2:
    print('='*80)
    all_param = list(mymodel.named_parameters())
    crf_param = list(mymodel.crf.named_parameters())
    lstm_param = list(mymodel.lstm.named_parameters())
    fc_param = list(mymodel.hidden2tag.named_parameters())
    bert_param = list(mymodel.bert.named_parameters())
    # emb_param = list(mymodel.word_embeds.named_parameters())

    # print(f'emb_param: {len(emb_param)}')
    print(f'crf_param: {len(crf_param)}')
    print(f'lstm_param: {len(lstm_param)}')
    print(f'fc_param: {len(fc_param)}')
    print(f'bert_param: {len(bert_param)}')
    print('='*50)
    print(f'all_param: {len(all_param)}')
    # print(len(all_param))

    other_param = crf_param + fc_param + lstm_param
    for n, p in other_param:
        print(n, p.shape)

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
