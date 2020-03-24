# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import torch
from torch import nn
from transformers import BertForTokenClassification, BertTokenizer

# from utils import my_lr_lambda
# from torch.optim.lr_scheduler import LambdaLR

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from model import MODEL_TEMP

class BERT_NER(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
        '''
        :param
            @params['n_tags']
            @params['use_cuda']
        '''
        super(BERT_NER, self).__init__()
        self.config = config
        self.num_labels = self.config.get('n_ent_tags', 45) - 2
        self.use_cuda = self.config.get('use_cuda', False)
        self.model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=self.num_labels)
        self.model_type = 'BERT_NER'

        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'n_ent_tags: {self.num_labels}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log('='*80, 0)

    def _loss(self, x, y_ent, lens):
        '''
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (1), torch.tensor
        '''
        use_cuda = self.use_cuda

        T = x.shape[1]
        input_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)  #(batch_size)
        att_mask = self._generate_mask(lens, max_len=T)  #(batch_size, T)
        labels = self._to_tensor(y_ent, use_cuda)

        res_tuple = self.model(input_tensor, attention_mask=att_mask, labels=labels)
        loss, score = res_tuple[0], res_tuple[1]
        return loss

    def _output(self, x, lens):
        '''
        return the softmax decode paths
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @paths: (batch_size, T), torch.tensor, 最佳句子路径
        '''
        use_cuda = self.use_cuda

        T = x.shape[1]
        input_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)  #(batch_size)
        att_mask = self._generate_mask(lens, max_len=T)  #(batch_size, T)

        labels = self.model(input_tensor, attention_mask=att_mask)[0]  #(batch_size, T, n_tags)
        label_max, label_argmax = labels.max(dim=2)
        paths = label_argmax
        return paths

    # def save_model(self, path: str):
    #     self.model.save_pretrained(path)

    # def load_model(self, path: str):
    #     if os.path.exists(path):
    #         # self.model.from_pretrained(path)
    #         self.model = BertForTokenClassification.from_pretrained(path)
    #         if self.use_cuda:
    #             self.model.cuda()
    #         print('reload model successfully(in_model)~')
    #     else:
    #         # print('build new model')
    #         pass

if __name__ == '__main__':
    # model_config = {
    #     'num_labels':45,
    #     'use_cuda':True   #True
    # }
    # mymodel = BERT_NER(config=model_config).cuda()
    # # mymodel.load_model('./result/')

    # ###===========================================================
    # ###模型参数测试
    # ###===========================================================
    # ##case1:
    # all_param = list(mymodel.named_parameters()) 
    # bert_param = [(n, p) for n, p in all_param if 'bert' in n]
    # other_param = [(n, p) for n, p in all_param if 'bert' not in n]
    # print(f'all_param: {len(all_param)}')
    # print(f'bert_param: {len(bert_param)}')
    # print(f'other_param: {len(other_param)}')
    # for n, p in other_param:
    #     print(n, p.shape)

    # ##case2:
    # cls_param = list(mymodel.model.classifier.named_parameters())
    # bert_param = list(mymodel.model.bert.named_parameters())
    # print(f'bert_param: {len(bert_param)}')
    # print(f'cls_param: {len(cls_param)}')
    # for n, p in cls_param:
    #     print(n, p.shape)

    ###===========================================================
    ###试训练
    ###===========================================================
    data_set = AutoKGDataset('./data/d4/')
    # train_dataset = data_set.train_dataset[:20]
    # eval_dataset = data_set.dev_dataset[:10]
    train_dataset = data_set.train_dataset
    eval_dataset = data_set.dev_dataset

    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')

    # print(data_loader.embedding_info_dicts['entity_type_dict'])
    model_params = {
        # 'embedding_dim' : 768,
        # 'hidden_dim' : 64,       
        'n_ent_tags' : len(data_loader.ent_seq_map_dict),  
        'n_rel_tags' : len(data_loader.rel_seq_map_dict),  
        'n_rels' : len(data_loader.label_location_dict)+1,
        'n_words' : len(data_loader.character_location_dict),
        'start_ent_idx': data_loader.ent_seq_map_dict[data_loader.START_TAG],  ## <start> tag index for entity tag seq
        'end_ent_idx': data_loader.ent_seq_map_dict[data_loader.END_TAG],  ## <end> tag index for entity tag seq
        'start_rel_idx': data_loader.rel_seq_map_dict[data_loader.START_TAG],  ## <start> tag index for relation tag seq
        'end_rel_idx': data_loader.rel_seq_map_dict[data_loader.END_TAG],  ## <end> tag index for relation tag seq
        'use_cuda':1,
        'dropout_prob': 0,
        'lstm_layer_num': 1  ##TODO:
        # 'num_labels': len(data_loader.ent_seq_map_dict)
    }

    train_param = {
        'EPOCH': 10,         #45  TODO:15
        'batch_size': 16,    #512   TODO:64
        'learning_rate_bert': 5e-5,
        'learning_rate_upper': 1e-3,  #TODO:
        'bert_finetune': True,
        'visualize_length': 20, #10
        'isshuffle': True,
        'model_name':'model_test.p',
        'result_dir': './result/'
    }
    mymodel = BERT_NER(model_params, show_param=True)
    mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)

    hyper_param = {
        'batch_size': 100,
        'issave': True,
        'result_dir': './result/'
    }
    # model.predict(data_loader, data_set=eval_dataset, hyper_param=predict_param)
    mymodel.load_model('./result/model_test.p')
    mymodel.eval_model(data_loader, eval_dataset, hyper_param, rebuild=True)

    # train_data_mat_dict = data_loader.transform(train_dataset)
    # data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)


    # for epoch in range(EPOCH):
    #     print('EPOCH: %d' % epoch)
    #     for data_batch in data_generator:
    #         x, pos, _, _, y_ent, lens, data_list = data_batch
    #         print(x.shape, pos.shape, y_ent.shape)    ##(batch_size, max_length)
    #         sentence = data_list[0]['input']
    #         # print([(i, sentence[i]) for i in range(len(sentence))])

    #         ###======================for BERT-MLP-MODEL only==================================
    #         mymodel._loss(x, y_ent, lens, use_cuda=False)
    #         mymodel._output(x, lens, use_cuda=False)

    #         print(x[0])
    #         word_dict = data_loader.character_location_dict
    #         rev_word_dict = data_loader.inverse_character_location_dict
    #         print(list(word_dict.items())[1300:1350])
    #         print(list(rev_word_dict.items())[1300:1350])
    #         print(sentence)
    #         print(list(rev_word_dict[i] for i in x[0]))
    #     break
    # break
