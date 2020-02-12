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
    print('relation seq dict length:', len(dataloader.embedding_info_dicts['rel_seq_map_dict']))
    print('character location dict length:', len(dataloader.embedding_info_dicts['character_location_dict']))
    print('pos location dict length:', len(dataloader.embedding_info_dicts['pos_location_dict']))
    print('============================================================')
    print()

def load_bert_pretrained_dict():
    vocab_list = []
    with open('./d1/vocab.txt', encoding='utf-8') as f:
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


class KGDataLoader(BaseLoader):
    def __init__(self, dataset: AutoKGDataset, rebuild=True, temp_dir = None):
        '''
        :param 
            @rebuild: 是否重新建立各类字典
            @istest: 是否是测试集
            @dataset: 通过AutoKGDataset得到的数据集
        '''
        super(KGDataLoader, self).__init__()
        self.dataset = dataset
        self.temp_dir = temp_dir
        self.metadata_ = dataset.metadata_
        self.sentence_max_len = max(100, min(self.metadata_['avg_sen_len'], 100))
        
        self.joint_embedding_info_dicts_path = os.path.join(temp_dir, "joint_embedding_info_dict.pkl")

        if (not rebuild) and os.path.exists(self.joint_embedding_info_dicts_path):
            self.embedding_info_dicts = self.load_preprocessed_data(
                self.joint_embedding_info_dicts_path
            ) 
        else:
            self.embedding_info_dicts = self._preprocess_data(self.dataset.all_train_dataset)

        self.ent_seq_map_dict = self.embedding_info_dicts['ent_seq_map_dict'] ## 实体序列字典
        self.inverse_ent_seq_map_dict = self._inverse_dict(self.ent_seq_map_dict)
        self.rel_seq_map_dict = self.embedding_info_dicts['rel_seq_map_dict'] ## 关系序列字典
        self.inverse_rel_seq_map_dict = self._inverse_dict(self.rel_seq_map_dict)

        self.entity_type_dict = self.embedding_info_dicts['entity_type_dict'] ## 实体类别字典
        self.inverse_entity_type_dict = self._inverse_dict(self.entity_type_dict)

        # self.character_location_dict = self.embedding_info_dicts['character_location_dict']  #TODO:choose the pretrained dict
        self.character_location_dict = load_bert_pretrained_dict()

        self.inverse_character_location_dict = self._inverse_dict(self.character_location_dict)
        self.pos_location_dict = self.embedding_info_dicts['pos_location_dict']
        self.inverse_pos_location_dict = self._inverse_dict(self.pos_location_dict)

    def _preprocess_data(self, data):
        '''
        建立各种所必须字典
        :param
            @data: list
                sample = data[0]  字典形式
                sample['input'] - sentence, 
                sample['output']['entity_list']: entity_list
                sample['output']['relation_list']: relation_list
            entity = sample['output']['relation_list'][0]
                entity['entity_type']: entity_type
                entity['entity']: real entity body
                entity['entity_index']['begin']: entity_index_begin
                entity['entity_index']['end']: entity_index_end
        :return
            @embedding_info_dicts: dict
                -embedding_info_dicts['character_location_dict']: ## 字符层面字典 - {"北":1, "京":2}
                -embedding_info_dicts['pos_location_dict']: ## POS字典 - {"b":1, "ag":2}
                -embedding_info_dicts['ent_seq_map_dict']: ##实体序列字典 - {"B_0":1, "I_0": 2, "E_0", 3, "B_1":4}
                -embedding_info_dicts['rel_seq_map_dict']:
                -embedding_info_dicts['entity_type_dict']: ##实体类别编号字典 - {"Time": 0, "Number": 1, "书籍": 2}
        '''

        print('start to proprocessing data...')
        tokenized_lists = []   ### word_list
        character_lists = []   ### char_list
        pos_lists = []         ### label_list

        # 将所有句子按照word以及char进行分离
        for cnt, item in enumerate(data):
            tokens, poss = self._tokenizer(item['input'])
            tokenized_lists.append(tokens)  
            pos_lists.append(poss)
            character_lists.append(list(item['input']))
            cnt += 1 
            if cnt % 300 == 0:
                log("PreProcess %.3f \r" % (cnt / len(data)))
        
        character_set = self._get_size_word_set(character_lists, size=None)   ### 字符层面上的一个set, 数目不大 --“北”, “京”
        character_location_dict = self._generate_word_dict(character_set)
        print('character location dict done...')

        pos_set = self._get_size_word_set(pos_lists, size=None)   ### 关于pos的一个set, 是在token层面，即词汇层面--“北京”
        pos_location_dict = self._generate_word_dict(pos_set)
        print('pos location dict done...')

        ent_seq_map_dict = {'ELSE': 0}
        entity_set = self.dataset.metadata_['entity_set']
        entity_type_dict = {}
        for index, each_entity in enumerate(entity_set):
            ## (0, Time) B_0:1, I_0: 2, E_0: 3
            ## (1, Number) B_1:4, I_1: 5, E_1: 6
            ent_seq_map_dict["B_{}".format(index)] = 3*index + 1
            ent_seq_map_dict["I_{}".format(index)] = 3*index + 2
            ent_seq_map_dict["E_{}".format(index)] = 3*index + 3
            entity_type_dict[each_entity] = index   ##{Time: 0, Number: 1}
        ## CRF需要，在tag中添加START_TAG以及END_TAG
        ent_seq_map_dict[self.START_TAG] = len(ent_seq_map_dict)
        ent_seq_map_dict[self.END_TAG] = len(ent_seq_map_dict)
        print('ent seq map dict done...')

        ##TODO: 作用有待研究。。。。？？？  rel:: relation???
        rel_seq_map_dict = {
            "ELSE": 0,
            "OBJ_B": 1,
            "OBJ_I": 2,
            "OBJ_E": 3,
            "SUB_B": 4,
            "SUB_I": 5,
            "SUB_E": 6
        }

        embedding_info_dicts = {
            "character_location_dict": character_location_dict,   ## 字符层面字典 - {"北":1, "京":2}
            "pos_location_dict": pos_location_dict,   ## POS字典 - {"b":1, "ag":2}
            "ent_seq_map_dict": ent_seq_map_dict,  ##实体序列字典 - {"B_0":1, "I_0": 2, "E_0", 3, "B_1":4}
            "rel_seq_map_dict": rel_seq_map_dict,
            "entity_type_dict": entity_type_dict  ##实体类别编号字典 - {"Time": 0, "Number": 1, "书籍": 2}
        }
        self.save_preprocessed_data(self.joint_embedding_info_dicts_path, embedding_info_dicts)
        print('preprocessing finished, save successfully~')
        return embedding_info_dicts

    def transform(self, data, istest=False, data_type='ent', radio=0.5):
        ## TODO: 先用只抽取实体的，关系方面先不考虑
        '''
        将文本数据矩阵化
        :param
            @data: list  (type: AutoKGDataset)
            @istest: 数据是训练集还是测试集
            @data_type:
                -- 'rel': relation
                -- 'ent': entity
                -- 'ent_rel': entity and relation
            @radio:
        
        :return
            @return_dict: dict
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                return_dict['y_ent_matrix']: ##(N, T), np.array, 命名实体编码序列  len = self.sentence_max_len
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['data_list']: ##(N), list 原始数据序列 - 增加postag信息
        '''
        if data_type == None:
            data_type = self.metadata_.get('data_type', 'ent_rel')  ## 默认类型'ent_rel'
        if data_type == 'rel':
            return self.transform_rel(data, istest, radio)
        elif data_type == 'ent':
            return self.transform_ent(data, istest)
        else:  #'ent_rel'
            return self.transform_ent_rel(data, istest, radio)
            # return self.transform_ent(data, istest)   

    def transform_ent(self, data, istest=False):
        '''
        将文本数据(目标为实体识别)矩阵化
        :param
            @data: list  (type:AutoKGDataset)  check on dataset.py
            @istest: 数据是训练集还是测试集, 如果istest=True, y_ent_matrix中都为0
        :return
            @return_dict: dict
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                (test-0)return_dict['y_ent_matrix']: ##(N, T), np.array, 命名实体编码序列  len = self.sentence_max_len
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['data_list']: ##(N), list 原始数据序列 - 增加postag信息
        '''
        character_location_dict = self.character_location_dict
        pos_location_dict = self.pos_location_dict
        sentence_max_len = self.sentence_max_len

        char_matrix_list = []   ## 字符编码序列
        sentence_length_list = []  ##句子长度序列
        pos_matrix_list = []  ## POS编码序列
        y_ent_matrix_list = []  ## 命名实体编码序列
        data_list = []  ## 原始数据序列 - 增加postag信息

        for row_idx, d in enumerate(data):
            input_text = d['input']   ## sentence
            tokens, poss = self._tokenizer(input_text)  ##分词，获取词性

            postag = []
            for token, pos in zip(tokens, poss):
                postag.append({'word': token, 'pos': pos})
            d['postag'] = postag

            ##获取文本的字符序列 --- char_list
            sentence_length = 0
            char_list = np.zeros((sentence_max_len))   ##warning: 0对应<PAD>
            for col_idx, cha in enumerate(input_text):
                if col_idx < sentence_max_len:
                    if cha in character_location_dict:
                        char_list[col_idx] = character_location_dict[cha]
                    else:
                        char_list[col_idx] = character_location_dict[self.UNK_TAG]
                    sentence_length += 1

            ##获取文本的词性序列 --- pos_list
            pos_list = np.zeros((sentence_max_len))
            last_word_loc = 0
            for item in d['postag']:
                word = item['word']
                pos = item['pos']
                word_start = input_text.find(word, last_word_loc)
                word_len = len(word)
                pos_list[word_start:min(word_start+word_len, sentence_max_len)] = pos_location_dict[pos]
                last_word_loc = word_start + word_len

            ##构造实体标注序列 --- y_ent_list
            y_ent_list = np.zeros([sentence_max_len])
            if not istest:
                for entity_dict in d['output']['entity_list']:
                    entity_type = entity_dict['entity_type']
                    entity_begin = entity_dict['entity_index']['begin']
                    entity_end = entity_dict['entity_index']['end']

                    y_ent_list[entity_begin:entity_end] = self.ent_seq_map_dict[
                        'I_{}'.format(self.entity_type_dict[entity_type])]
                    if entity_end < sentence_max_len:
                        y_ent_list[entity_end - 1] = self.ent_seq_map_dict[
                            'E_{}'.format(self.entity_type_dict[entity_type])]
                    if entity_begin < sentence_max_len:
                        y_ent_list[entity_begin] = self.ent_seq_map_dict[
                            'B_{}'.format(self.entity_type_dict[entity_type])]

            y_ent_matrix_list.append(y_ent_list)  #test: all_zero
            char_matrix_list.append(char_list)  ##test: sentence info
            sentence_length_list.append(sentence_length)  ##test: sentence info
            pos_matrix_list.append(pos_list)  ##test: sentence info
            data_list.append(d)  ##test: 只有input, 以及tag的信息

            if(row_idx % 300 == 0):
                log("Process %.3f \r" % (row_idx / len(data)))

        ## turn all list(list) to matrix
        char_matrix = np.vstack(char_matrix_list)
        y_ent_matrix = np.vstack(y_ent_matrix_list)
        pos_matrix = np.vstack(pos_matrix_list)
        return_dict = {
            'cha_matrix': char_matrix,
            'y_ent_matrix': y_ent_matrix,
            'pos_matrix': pos_matrix,
            'sentence_length': sentence_length_list,
            'data_list': data_list
        }
        return return_dict

    def transform_rel(self, data, istest=False, radio=0.5):
        pass

    def transform_ent_rel(self, data, istest=False, radio=0.5):
        pass

    def transform_back(self, result, data_type='ent'):
        '''
        :param
            @result: dict -- self.transform()的输出
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                return_dict['y_ent_matrix']: ##(N, T), np.array, 命名实体编码序列  len = self.sentence_max_len
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['data_list']: ##(N), list 原始数据序列 - 增加postag信息
            @data_type: {'ent', 'rel', 'ent_rel'}
        :return
            @ans: list
                case = ans[0]
                case['input']
                case['entity_list']
                    e = case['entity_list'][0] 
                    e['entity']:2016年04月08日
                    e['entity_type']:Date
                    e['entity_index']['begin']:13
                    e['entity_index']['end']:24
        '''
        if data_type == None:
            data_type = self.metadata_.get('data_type', 'ent_rel')
        ans = []
        sentence_ore_sub_dict = defaultdict(list)
        sample_number = len(result['data_list'])

        relation = None
        sub_list = None
        obj_list = None
        entity_set = None

        for idx in range(sample_number):
            sentence = result['data_list'][idx]['input']
            if data_type == 'ent_rel' or data_type == 'rel':
                ## TODO: 添加对于关系抽取的处理
                pass
            if data_type == 'ent_rel' or data_type == 'ent':
                ## 对命名实体的处理
                y_ent = result['y_ent_matrix'][idx]
                entity_set = self._obtain_entity(y_ent, sentence)
            sentence_ore_sub_dict[sentence].append([relation, sub_list, obj_list, entity_set])

        for sentence in sentence_ore_sub_dict.keys():
            ans_d = {}
            ans_d["input"] = sentence
            relation_list = set()
            all_eneity_set = set()
            for relation, sub_list, obj_list, entity_set in sentence_ore_sub_dict[sentence]:
                if data_type == "ent_rel" or data_type == "ent":
                    all_eneity_set = all_eneity_set.union(set(entity_set))
                if data_type == "ent_rel" or data_type == "rel":
                    for sub in sub_list:
                        for obj in obj_list:
                            relation_list.add(
                                json.dumps({"relation": relation, "head": sub["entity"], "tail": obj["entity"]},
                                           ensure_ascii=False))
            if data_type == "ent_rel" or data_type == "rel":
                ans_d["relation_list"] = [json.loads(s) for s in relation_list]
            if data_type == "ent_rel" or data_type == "ent":
                ans_d["entity_list"] = list(json.loads(i) for i in all_eneity_set)  ##将string全部转化成字典形式保存
            ans.append(ans_d)
        return ans

    def _obtain_entity(self, y_ent, sentence):
        '''
        :param
            @y_ent: encode之后的ent句子, np.array  (T)
            @sentence: 原始句子  
        :return
            entity_set: set,  里面元素是String, 看上去是字典格式的，需要json.load()读取
                case = entity_set[0]
                case['entity']
                case['entity_type']
                case['entity_index']
        '''
        entity_set = set()
        sentence_length = min(len(sentence), self.sentence_max_len)
        for entity_type, entity_type_index in self.entity_type_dict.items():
            B_label = "B_{}".format(entity_type_index)
            I_label = "I_{}".format(entity_type_index)
            E_label = "E_{}".format(entity_type_index)
            tmp_ent = None
            for idx in range(sentence_length):
                value = int(y_ent[idx])
                if self.inverse_ent_seq_map_dict[value] == B_label:
                    tmp_ent = {"entity": sentence[idx], 
                               "entity_index": {"begin": idx, "end": idx + 1}, 
                               "entity_type": entity_type
                               }

                elif self.inverse_ent_seq_map_dict[value] == I_label:
                    if tmp_ent != None:
                        tmp_ent["entity"] += sentence[idx]
                elif self.inverse_ent_seq_map_dict[value] == E_label:
                    if tmp_ent != None:
                        tmp_ent["entity"] += sentence[idx]
                        tmp_ent["entity_index"]["end"] = idx + 1
                        entity_set.add(json.dumps({"entity": tmp_ent["entity"], 
                                                   "entity_type": tmp_ent["entity_type"], 
                                                   "entity_index": 
                                                       {"begin": tmp_ent["entity_index"]["begin"], 
                                                        "end": tmp_ent["entity_index"]["end"]}
                                                },ensure_ascii=False))
                        tmp_ent = None
                else:
                    tmp_ent = None
        return entity_set


class Batch_Generator(nn.Module):
    def __init__(self, data_dict, batch_size=16, data_type='ent', isshuffle=True):
        '''
        :param
            @data_dict: dict --- KGDataLoader.transform()的返回值
                data_dict['cha_matrix']: ## 字符编码序列
                data_dict['y_ent_matrix']: ## 命名实体编码序列
                data_dict['pos_matrix']: ## POS编码序列
                data_dict['sentence_length']: ##句子长度序列
                data_dict['data_list']: ## 原始数据序列 - 增加postag信息
            @batch_size: 
            @data_type: {'ent', 'rel', 'ent_rel'}
        '''
        self.x = data_dict['cha_matrix']
        self.number = len(self.x)
        self.relation = data_dict.get('relation_matrix', [None]*self.number)
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
            @relation, 
            @y_rel, 
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

    load_bert_pretrained_dict()
    # data_set = AutoKGDataset('/Users/work-xx-pc/Desktop/xx-work/bilstm_crf_test/d1/')
    # train_dataset = data_set.train_dataset[:10]

    # data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='/Users/work-xx-pc/Desktop/xx-work/bilstm_crf_test/result/')
    # show_dict_info(data_loader)
    
    # train_data_mat_dict = data_loader.transform(train_dataset)
    # data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)

    # for epoch in range(3):
    #     print('='*100)
    #     print('epoch %d' % (epoch))
    #     for data_batch in data_generator:
    #         x, pos, _, _, y_ent, sentence_length, data_list = data_batch
    #         print(x.shape)
    #         print(sentence_length)

    #         for i in range(x.shape[0]):
    #             ori_data = data_list[i]

    #             ## check x -- 句子字符层面上的编码======================
    #             ori_sentence = ori_data['input']
    #             x_i = x[i][:sentence_length[i]]
    #             x_decode = [data_loader.inverse_character_location_dict[w] for w in x_i]
    #             x_decode = ''.join(x_decode)
    #             print('orgin sentence:', ori_sentence)
    #             print('decode sentence:', x_decode)

    #             ## check y_ent -- entity============================
    #             # ori_sentence = ori_data['input']
    #             # ori_entitys = ori_data['output']['entity_list']
    #             # print('ori_entitys', ori_entitys)
    #             # ent_i = y_ent[i][:sentence_length[i]]
    #             # ent_decode = data_loader._obtain_entity(ent_i, ori_sentence)
    #             # print('ent_decode', ent_decode)
    #             # print(type(ent_decode), type(list(ent_decode)[0]))

    #             ## check pos============================
    #             # ori_sentence = ori_data['input']
    #             # ori_pos = pseg.cut(ori_sentence)
    #             # wlist, plist = [], []
    #             # for wp in ori_pos:
    #             #     wlist.append(wp.word)
    #             #     plist.append(wp.flag)
    #             # pos_i = pos[i][:sentence_length[i]]
    #             # pos_decode = [data_loader.inverse_pos_location_dict[p] for p in pos_i]
    #             # print(wlist[:5])
    #             # print(plist[:5])
    #             # pos_check = list(zip(list(ori_sentence), pos_decode))
    #             # print(pos_check)

    #             break



        