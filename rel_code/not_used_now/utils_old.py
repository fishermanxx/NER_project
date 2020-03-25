import os
import json
import pickle
from collections import defaultdict
import jieba
import jieba.posseg as pseg
import numpy as np
import random

import sklearn
from sklearn.utils import shuffle
import torch
import torch.nn as nn

from dataset import AutoKGDataset


def my_lr_lambda(epoch):
    return 1/(1+0.05*epoch)

def show_result(datalist, checklist):
    for i in checklist:
        data = datalist[i]
        print(data['input'])
        data = data.get('output', data)
        for e in data['entity_list']:
            print(e['entity_type'], '-->', e['entity'])

def log(entry, nesting_level=0):
    space = '-'*(4*nesting_level)
    print(f"{space}{entry}")

def show_metadata(metadata):
    '''
    :param
        (AutoKGDataset)--dataset.metadata_
        @self.metadata_: dict
            self.metadata_['char_size']
            self.metadata_['char_set']
            self.metadata_['entity_size']
            self.metadata_['entity_set']
            self.metadata_['relation_size']
            self.metadata_['relation_set']
            self.metadata_['max_sen_len']
            self.metadata_['avg_sen_len']
            self.metadata_['train_num']
            self.metadata_['test_num']
    '''
    print('===================metadata of the dataset======================')
    print('char size:', metadata['char_size'])
    print('entity size:', metadata['entity_size'])
    print('relation size:', metadata['relation_size'])
    print('train num:', metadata['train_num'])
    print('test num:', metadata['test_num'])
    print('================================================================')
    print()

def show_dict_info(dataloader):
    '''
    :param
        dataloader: KGDataLoader类
    '''
    print('===================info about the data======================')
    print('entity type dict length:', len(dataloader.embedding_info_dicts['entity_type_dict']))
    print('entity seq dict length:', len(dataloader.embedding_info_dicts['ent_seq_map_dict']))
    print('relation type dict length:', len(dataloader.embedding_info_dicts['relation_type_dict']))  ##TODO: change
    print('relation seq dict length:', len(dataloader.embedding_info_dicts['rel_seq_map_dict']))
    print('character location dict length:', len(dataloader.embedding_info_dicts['character_location_dict']))
    print('pos location dict length:', len(dataloader.embedding_info_dicts['pos_location_dict']))
    print('============================================================')
    print()

def load_bert_pretrained_dict():
    vocab_list = []
    with open('./data/vocab.txt', encoding='utf-8') as f:
        for l in f.readlines():
            vocab_list.append(l.strip())

    vocab_dict = {}
    for idx, word in enumerate(vocab_list):
        vocab_dict[word]=idx
    # print(len(vocab_dict))
    return vocab_dict

class BaseLoader:
    def __init__(self):
        self.START_TAG = '<start>'
        self.END_TAG= '<end>'
        self.UNK_TAG = '[UNK]'
        self.PAD_TAG = '[PAD]'

    def _inverse_dict(self, d):
        '''
        反转字典
        '''
        return dict(zip(d.values(), d.keys()))

    def save_preprocessed_data(self, filename, data):
        with open(filename, 'wb') as f:
            p_str = pickle.dump(data, f)
        return p_str

    def load_preprocessed_data(self, filename):
        with open(filename, 'rb') as f:
            processed_dict = pickle.load(f)
        return processed_dict

    def _get_size_word_set(self, data, size=None, add_unk=True):
        '''
        获得文本语料中按照频率从多到少排序的前size个单词的set
        :param 
            @data: list[list]
        :return 
            @word_dict: set
        '''
        word_count = defaultdict(int)
        for item in data:
            for word in item:
                word_count[word] += 1

        word_list = list(word_count.items())
        word_list = sorted(word_list, key=lambda x: x[1], reverse=True)

        if size is not None and len(word_list) > size:
            small_word_list = word_list[:size]
        else:
            small_word_list = word_list

        word_dict = set()
        for word, _ in small_word_list:
            word_dict.add(word)
        if add_unk:
            word_dict.add(self.UNK_TAG)
        return word_dict

    def _generate_word_dict(self, data_dict):
        '''
        :param 
            @data_dict: set, self._get_size_word_set的output
        :return 
            @dic: word2idx
            0 没有对应的， 作为<PAD>使用
        '''
        word_location_dict = {self.PAD_TAG: 0}
        for row_idx, word in enumerate(data_dict):
            word_location_dict[word] = row_idx + 1
        # word_location_dict[self.START_TAG] = len(word_location_dict)
        # word_location_dict[self.END_TAG] = len(word_location_dict)
        return word_location_dict 

    def _tokenizer(self, sentence):
        words = pseg.cut(sentence)
        tokens = []
        poss = []
        for w in words:
            tokens.append(w.word)
            poss.append(w.flag)
        return tokens, poss


class Batch_Generator(nn.Module):
    def __init__(self, data_dict, batch_size=16, data_type='ent', isshuffle=True):
        '''
        :param
            @data_dict: dict --- KGDataLoader.transform()的返回值
                data_dict['cha_matrix']: ## 字符编码序列
                data_dict['y_ent_matrix']: ## 命名实体编码序列
                data_dict['y_rel_matrix']
                data_dict['relation_type_list']
                data_dict['pos_matrix']: ## POS编码序列
                data_dict['sentence_length']: ##句子长度序列
                data_dict['data_list']: ## 原始数据序列 - 增加postag信息
            @batch_size: 
            @data_type: {'ent', 'rel', 'ent_rel'}
        '''
        self.x = data_dict['cha_matrix']
        self.number = len(self.x)
        self.relation = data_dict.get('relation_type_list', [None]*self.number)
        self.y_rel = data_dict.get('y_rel_matrix', [None]*self.number)
        self.y_ent = data_dict.get('y_ent_matrix', [None]*self.number)
        self.sentence_length = data_dict['sentence_length']
        self.pos = data_dict['pos_matrix']
        self.data_list = data_dict['data_list']
        self.batch_size = batch_size
        self.current = 0
        self.isshuffle = isshuffle
        self.data_type = data_type

    def __iter__(self):
        return self

    def __next__(self):
        '''
        :return
            @x: (batch_size, max_length)
            @pos: (batch_size, max_length)
            @relation, (batch_size, 1)
            @y_rel, (batch_size, max_length)
            @y_ent: (batch_size, max_length)
            @sentence_length: (batch_size) 
            @data_list: (batch_size)
        '''
        if self.current >= len(self.x):
            self.current = 0
            if self.isshuffle:
                self.x, self.pos, self.relation, self.y_rel, self.y_ent, self.sentence_length, self.data_list = sklearn.utils.shuffle(
                    self.x,
                    self.pos,
                    self.relation,
                    self.y_rel,
                    self.y_ent,
                    self.sentence_length,
                    self.data_list
                )
            raise StopIteration
        else:
            old_current = self.current
            to_ = self.current + self.batch_size
            self.current = to_
            if self.data_type == 'ent':
                return self.x[old_current:to_, :], \
                       self.pos[old_current:to_, :], \
                       None, \
                       None, \
                       self.y_ent[old_current:to_, :], \
                       self.sentence_length[old_current:to_], \
                       self.data_list[old_current:to_]

            elif self.data_type == 'rel':
                return self.x[old_current:to_, :], \
                       self.pos[old_current:to_, :], \
                       np.expand_dims(self.relation[old_current:to_], 1), \
                       self.y_rel[old_current:to_, :], \
                       None, \
                       self.sentence_length[old_current:to_], \
                       self.data_list[old_current:to_]
            
            else:
                return self.x[old_current:to_, :], \
                       self.pos[old_current:to_, :], \
                       np.expand_dims(self.relation[old_current:to_], 1), \
                       self.y_rel[old_current:to_, :], \
                       self.y_ent[old_current:to_, :], \
                       self.sentence_length[old_current:to_], \
                       self.data_list[old_current:to_]


if __name__ == '__main__':

    # load_bert_pretrained_dict()
    result_dir = './result/'
    data_set = AutoKGDataset('./data/d4/')
    train_dataset = data_set.train_dataset[:10]

    import os
    os.makedirs(result_dir, exist_ok=True)

    data_loader = KGDataLoader(data_set, rebuild=False, temp_dir=result_dir)
    show_dict_info(data_loader)
    
    # train_data_mat_dict = data_loader.transform_rel(train_dataset, istest=False, ratio=0)
    train_data_mat_dict = data_loader.transform(train_dataset, istest=False, data_type='rel', ratio=0)
    data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='rel', isshuffle=True)
    # data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)
    
    pred = data_loader.transform_back(train_data_mat_dict, data_type='rel')
    for i in range(len(train_dataset)):
        ori_data = train_dataset[i]
        pre_data = pred[i]

        print('origin sentence:')
        print(ori_data['input'])
        print('decode sentence')
        print(pre_data['input'])

        def str_relation_fn(item):
            return item['relation']+'--'+item['head']+'--'+item['tail']

        print('origin relation')
        ori_relations_str = list(map(str_relation_fn, ori_data['output']['relation_list']))
        print(ori_relations_str)
        print('decode relation')
        decode_relations_str = list(map(str_relation_fn, pre_data['relation_list']))
        print(decode_relations_str)
        print('='*80)
        print()
        # break

    '''
    for epoch in range(1):
        print('='*100)
        print('epoch %d' % (epoch))
        for data_batch in data_generator:
            x, pos, relation, y_rel, y_ent, sentence_length, data_list = data_batch
            print(x.shape, pos.shape, y_rel.shape)  ##x, pos, y_rel, y_ent (batch_size, max_seq_length)
            print(sentence_length)  ##relation(batch_size, 1), sentence_length, data_list  (batch_size)
            print(relation.shape, relation[:, 0])
            for i in range(x.shape[0]):
                ori_data = data_list[i]

                ## check x -- 句子字符层面上的编码======================
                ori_sentence = ori_data['input']
                x_i = x[i][:sentence_length[i]]
                x_decode = [data_loader.inverse_character_location_dict[w] for w in x_i]
                x_decode = ''.join(x_decode)
                print('orgin sentence:', ori_sentence)
                print('decode sentence:', x_decode)

                ## check y_rel and relation -- relation======================
                ori_relations = ori_data['output']['relation_list']
                def str_relation(item):
                    return item['relation']+'--'+item['head']+'--'+item['tail']
                ori_relations_str = list(map(str_relation, ori_relations))
                print('origin relations')
                print(ori_relations_str)

                relation_i = relation[i][0]
                rel_type_decode = data_loader.inverse_relation_type_dict[relation_i]
                sub_decode = data_loader._obtain_sub_obj(y_rel[i], ori_sentence, entity_type='sub')
                obj_decode = data_loader._obtain_sub_obj(y_rel[i], ori_sentence, entity_type='obj')
                print('decode relations')
                print(f"{rel_type_decode}--[{sub_decode[0]['entity']}]--[{obj_decode[0]['entity']}]")  

                # relation_i = relation[i][0]
                # rel_type_decode = data_loader.inverse_relation_type_dict[relation_i]
                # # print('decode relation type: ', rel_type_decode)
                # relation_seq_i = y_rel[i, :sentence_length[i]]
                # print('decode y_rel:')
                # print(relation_seq_i)
                # def rel_decode(y_rel, sentence):
                #     obj, sub = '', ''
                #     obj_s, obj_e, sub_s, sub_e = -1, -1, -1, -1
                #     for idx, y_rel_i in enumerate(y_rel):
                #         if obj_s < 0 and y_rel_i == 1:
                #             obj_s = idx
                #         if obj_e < 0 and y_rel_i == 3:
                #             obj_e = idx
                #         if sub_s < 0 and y_rel_i == 4:
                #             sub_s = idx
                #         if sub_e < 0 and y_rel_i == 6:
                #             sub_e = idx
                #     if obj_s >= 0 and obj_e > 0 and obj_s < obj_e:
                #         obj = sentence[obj_s:obj_e+1]
                #     if sub_s >= 0 and sub_e > 0 and sub_s < sub_e:
                #         sub = sentence[sub_s:sub_e+1]    
                #     return sub, obj
                # sub, obj = rel_decode(relation_seq_i, ori_sentence)  
                # print(f'decode relations')
                # print(f'{rel_type_decode}--[{sub}]--[{obj}]')              
                # print('='*80)

                ## check y_ent -- entity============================
                # ori_sentence = ori_data['input']
                # ori_entitys = ori_data['output']['entity_list']
                # print('ori_entitys', ori_entitys)
                # ent_i = y_ent[i][:sentence_length[i]]
                # ent_decode = data_loader._obtain_entity(ent_i, ori_sentence)
                # print('ent_decode', ent_decode)
                # print(type(ent_decode), type(list(ent_decode)[0]))

                ## check pos============================
                # ori_sentence = ori_data['input']
                # ori_pos = pseg.cut(ori_sentence)
                # wlist, plist = [], []
                # for wp in ori_pos:
                #     wlist.append(wp.word)
                #     plist.append(wp.flag)
                # pos_i = pos[i][:sentence_length[i]]
                # pos_decode = [data_loader.inverse_pos_location_dict[p] for p in pos_i]
                # print(wlist[:5])
                # print(plist[:5])
                # pos_check = list(zip(list(ori_sentence), pos_decode))
                # print(pos_check)

                break
            break
            print('='*80)
        break
        print()
        '''

        