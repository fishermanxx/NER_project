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

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.manual_seed(1)

class BERT_CRF2(MODEL_TEMP):
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
        super(BERT_CRF2, self).__init__()
        self.config = config
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.n_tags = self.config['n_rel_tags']-2
        # self.n_words = self.config['n_words']
        # self.dropout_prob = self.config.get('dropout_prob', 0)

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
        log(f'n_rel_tags: {self.n_tags}', 1)
        # log(f"crf_start_idx: {self.config['start_ent_idx']}", 1)
        # log(f"crf_end_idx: {self.config['end_ent_idx']}", 1)
        # log(f'dropout_prob: {self.dropout_prob}', 1)  
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

        logits = self._get_features(x, lens, use_cuda)  ##(batch_size, T, n_tags)

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
        'hidden_dim' : 64,
        'n_tags' : 45,
        'n_words' : 22118,
        'start_idx': 43,  ## <start> tag index for entity tag seq
        'end_idx': 44,  ## <end> tag index for entity tag seq
        'use_cuda':1,
        'dropout_prob': 0,
        'lstm_layer_num': 1,
        'num_labels': 45
    }
    
    mymodel = BERT_CRF(model_params, show_param=True).cuda()


    ###===========================================================
    ###模型参数测试
    ###===========================================================
    ### case1
    all_param = list(mymodel.named_parameters()) 
    all_param2 = list(mymodel.parameters())
    bert_param = [(n, p) for n, p in all_param if 'bert' in n]
    other_param = [(n, p) for n, p in all_param if 'bert' not in n]
    print(f'all_param: {len(all_param)}')
    print(f'bert_param: {len(bert_param)}')
    print(f'other_param: {len(other_param)}')
    for n, p in other_param:
        print(n, p.shape)

    print('='*80)
    crf_param = list(mymodel.crf.named_parameters())
    crf_param2 = list(mymodel.crf.parameters())
    fc_param = list(mymodel.hidden2tag.named_parameters())
    fc_param2 = list(mymodel.hidden2tag.parameters())
    bert_param = list(mymodel.bert.named_parameters())
    bert_param2 = list(mymodel.bert.parameters())
    print(f'all_param: {len(all_param)}, {len(all_param2)}')
    print(f'crf_param: {len(crf_param)}, {len(crf_param2)}')
    print(f'fc_param: {len(fc_param)}, {len(fc_param2)}')
    print(f'bert_param: {len(bert_param)}, {len(bert_param2)}')

    other_param2 = crf_param + fc_param
    for n, p in other_param2:
        print(n, p.shape)

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