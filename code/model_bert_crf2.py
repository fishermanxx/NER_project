# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import transformers
import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

import os

from model import MODEL_TEMP
from torchcrf import CRF

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.manual_seed(1)

class BERT_CRF2(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
        '''
        :param - dict
            param['embedding_dim']
            param['n_ent_tags']
            param['use_cuda']
        '''
        super(BERT_CRF2, self).__init__()
        self.config = config
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.n_tags = self.config['n_ent_tags']-2
        self.use_cuda = self.config['use_cuda']
        self.model_type = 'BERT_CRF2'

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log(f'embedding_dim: {self.embedding_dim}', 1)       
        log(f'n_ent_tags: {self.n_tags}', 1)
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        self.hidden2tag = nn.Linear(self.embedding_dim, self.n_tags)
        self.crf = CRF(self.n_tags, batch_first=True)
        self.bert = transformers.BertModel.from_pretrained('bert-base-chinese')

    def reset_parameters(self):        
        I.xavier_normal_(self.hidden2tag.weight.data)
        self.crf.reset_parameters()
        
    def _get_features(self, x, lens, use_cuda=None):
        '''
        :param  
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: 每个句子的实际长度 (batch_size)
        :return 
            @lstm_feature: (batch_size, T, n_tags) -- 类似于eject score, torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size, T = x.shape

        ##bert layer
        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)
        att_mask = self._generate_mask(lens, max_len=T)
        embeds = self.bert(words_tensor, attention_mask=att_mask)[0]  #(batch_size, T, n_embed)
        
        ##FC layer
        feature = self.hidden2tag(embeds) #(batch_size, T, n_tags)
        feature = torch.tanh(feature)
        # print(feature.shape)
        return feature

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

        logits = self._get_features(x, lens, use_cuda)  ##(batch_size, T, n_tags)

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
            @paths: (batch_size, T+1), torch.tensor, 最佳句子路径
            @scores: (batch_size), torch.tensor, 最佳句子路径上的得分
        '''
        # self.eval()
        use_cuda = self.use_cuda if use_cuda is None else use_cuda

        logits = self._get_features(x, lens, use_cuda)
        paths = self.crf.decode(logits)
        paths = self._to_tensor(paths, use_cuda)
        return paths

if __name__ == '__main__':
    model_params = {
        'embedding_dim' : 768,
        'n_ent_tags' : 45,
        'use_cuda':False,
    }
    
    mymodel = BERT_CRF(model_params, show_param=True)

    ###===========================================================
    ###试训练
    ###===========================================================
    # data_set = AutoKGDataset('./d1/')
    # train_dataset = data_set.train_dataset[:20]
    # eval_dataset = data_set.dev_dataset[:10]
    # # # train_dataset = data_set.train_dataset
    # # # eval_dataset = data_set.dev_dataset

    # os.makedirs('result', exist_ok=True)
    # data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')

    # # print(data_loader.embedding_info_dicts['entity_type_dict'])
 
 
    # train_data_mat_dict = data_loader.transform(train_dataset)
    # data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)

    # for epoch in range(2):
    #     print('EPOCH: %d' % epoch)
    #     for data_batch in data_generator:
    #         x, pos, _, _, y_ent, lens, data_list = data_batch
    #         print(x.shape, pos.shape, y_ent.shape)    ##(batch_size, max_length)
    #         sentence = data_list[0]['input']
    #         # print([(i, sentence[i]) for i in range(len(sentence))])

    #         ###======================for BERT-MLP-MODEL only==================================
    #         mymodel._get_features(x, lens)
    #         loss = mymodel._loss(x, y_ent, lens)
    #         print(loss.shape)
    #         mymodel._output(x, lens)

    #         # print(x[0])
    #         # word_dict = data_loader.character_location_dict
    #         # rev_word_dict = data_loader.inverse_character_location_dict
    #         # print(list(word_dict.items())[1300:1350])
    #         # print(list(rev_word_dict.items())[1300:1350])
    #         # print(sentence)
    #         # print(list(rev_word_dict[i] for i in x[0]))
    #         break
    #     break