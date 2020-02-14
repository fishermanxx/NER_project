# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import torch
from torch import nn
from transformers import BertForTokenClassification, BertTokenizer

from utils import my_lr_lambda
from torch.optim.lr_scheduler import LambdaLR

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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
        self.num_labels = self.config.get('n_tags', 45)
        self.use_cuda = self.config.get('use_cuda', False)
        self.model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=self.num_labels)
        self.model_type = 'BERT_NER'

        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'num_labels: {self.num_labels}', 1)
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

    # def train_model(self, data_loader: KGDataLoader, hyper_param={}, train_dataset=None, eval_dataset=None):
    #     '''
    #     :param
    #         @hyper_param['learning_rate_upper']
    #         @hyper_param['learning_rate_bert']
    #         @hyper_param['bert_finetune']
    #         @hyper_param['EPOCH']
    #         @hyper_param['batch_size']
    #         @hyper_param['visualize_length']
    #         @hyper_param['result_dir']
    #         @hyper_param['isshuffle']
    #         @hyper_param['model_name']
    #     '''
    #     LEARNING_RATE_bert = hyper_param.get('learning_rate_bert', 5e-5)
    #     LEARNING_RATE_upper = hyper_param.get('learning_rate_upper', 1e-3)

    #     bert_finetune = hyper_param.get('bert_finetune', True)
    #     EPOCH = hyper_param.get('EPOCH', 3)
    #     BATCH_SIZE = hyper_param.get('batch_size', 4)
    #     use_cuda = self.use_cuda
    #     visualize_length = hyper_param.get('visualize_length', 2)
    #     result_dir = hyper_param.get('result_dir', './result/')
    #     is_shuffle = hyper_param.get('isshuffle', True)
    #     model_name = hyper_param.get('model_name', 'model.p')
    #     DATA_TYPE = 'ent'

    #     if use_cuda:
    #         self.model.cuda()

    #     train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset
    #     # train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
    #     ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
    #     old_train_dict_path = os.path.join(result_dir, 'train_data_mat_dict.pkl')
    #     if os.path.exists(old_train_dict_path):
    #         train_data_mat_dict = data_loader.load_preprocessed_data(old_train_dict_path)
    #         log('Reload preprocessed data successfully~')
    #     else:
    #         train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
    #         data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
    #     ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
    #     data_generator = Batch_Generator(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)

    #     cls_param = self.model.classifier.parameters()
    #     bert_param = self.model.bert.parameters()

    #     if bert_finetune:
    #         optimizer_group_paramters = [
    #             {'params': cls_param, 'lr': LEARNING_RATE_upper}, 
    #             {'params': bert_param, 'lr': LEARNING_RATE_bert}
    #         ]
    #         optimizer = torch.optim.Adam(optimizer_group_paramters)
    #         log(f'****BERT_finetune, learning_rate_upper: {LEARNING_RATE_upper}, learning_rate_bert: {LEARNING_RATE_bert}', 0)
    #     else:
    #         optimizer = torch.optim.Adam(cls_param, lr=LEARNING_RATE_upper)
    #         log(f'****BERT_fix, learning_rate_upper: {LEARNING_RATE_upper}', 0)

    #     ##TODO:
    #     scheduler = LambdaLR(optimizer, lr_lambda=my_lr_lambda)
    #     # scheduler = transformers.optimization.get_cosine_schedule_with warmup(optimizer, num_warmup_steps=int(EPOCH*0.2), num_training_steps=EPOCH)

    #     all_cnt = len(train_data_mat_dict['cha_matrix'])
    #     log(f'{model_name} Training start!', 0)
    #     loss_record = []
    #     score_record = []
    #     max_score = 0

    #     eval_hyper_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
    #     for epoch in range(EPOCH):
    #         log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
    #         self.model.train()
                
    #         loss = 0.0
    #         for cnt, data_batch in enumerate(data_generator):
    #             x, pos, _, _, y_ent, lens, data_list = data_batch  

    #             loss_avg = self._loss(x, y_ent, lens)
    #             optimizer.zero_grad()
    #             loss_avg.backward()
    #             optimizer.step()

    #             loss += loss_avg
    #             if use_cuda:
    #                 loss_record.append(loss_avg.cpu().item())
    #             else:
    #                 loss_record.append(loss_avg.item())

    #             if (cnt+1) % visualize_length == 0:
    #                 loss_cur = loss / visualize_length
    #                 log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
    #                 loss = 0.0

    #                 # self.model.eval()
    #                 # print(data_list[0]['input'])
    #                 # pre_paths = self._output(x, lens)
    #                 # print('predict-path')
    #                 # print(pre_paths[0])
    #                 # print('target-path')
    #                 # print(y_ent[0])
    #                 # self.model.train()

    #         temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_hyper_param)
    #         score_record.append(temp_score)
    #         scheduler.step()

    #         if temp_score[2] > max_score:
    #             max_score = temp_score[2]
    #             save_path = os.path.join(result_dir, model_name)
    #             self.save_model(save_path)
    #             print(f'Checkpoint saved successfully, current best socre is {max_score}')
    #         # break
    #     log(f'the best score of the model is {max_score}')
    #     return loss_record, score_record


if __name__ == '__main__':
    model_param = {
        'num_labels':45,
        'use_cuda':True   #True
    }
    mymodel = BERT_NER(params=model_param)
    # mymodel.load_model('./result/')

    data_set = AutoKGDataset('./d1/')
    train_dataset = data_set.train_dataset[:20]
    eval_dataset = data_set.dev_dataset[:10]
    # train_dataset = data_set.train_dataset
    # eval_dataset = data_set.dev_dataset

    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')

    print(data_loader.embedding_info_dicts['entity_type_dict'])

    train_param = {
        'learning_rate':5e-5,
        'EPOCH':3,  #30
        'batch_size':64,  #64
        'visualize_length':20,  #10
        'result_dir': './result/',
        'isshuffle': True
    }
    mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)


    # predict_param = {
    #     'result_dir':'./result/',
    #     'issave':False,
    #     'batch_size':64
    # }
    # # model.predict(data_loader, data_set=eval_dataset, hyper_param=predict_param)
    # model.eval_model(data_loader, data_set=eval_dataset, hyper_param=predict_param)


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
