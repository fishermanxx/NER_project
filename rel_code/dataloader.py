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


from utils import BaseLoader, log, load_bert_pretrained_dict, Batch_Generator, show_dict_info
from dataset import AutoKGDataset


class KGDataLoader2(BaseLoader):
    def __init__(self, dataset: AutoKGDataset, rebuild=True, temp_dir=None):
        '''
        :param 
            @rebuild: 是否重新建立各类字典
            @istest: 是否是测试集
            @dataset: 通过AutoKGDataset得到的数据集
        '''
        super(KGDataLoader2, self).__init__()
        self.dataset = dataset
        self.temp_dir = temp_dir
        self.metadata_ = dataset.metadata_
        self.sentence_max_len = max(100, min(self.metadata_['avg_sen_len'], 100))  ##TODO:
        
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

        self.relation_type_dict = self.embedding_info_dicts['relation_type_dict']  ## 关系类型字典
        self.inverse_relation_type_dict = self._inverse_dict(self.relation_type_dict)

        # self.character_location_dict = self.embedding_info_dicts['character_location_dict']  #TODO:choose the pretrained dict
        self.character_location_dict = load_bert_pretrained_dict()  ## input字符序列字典
        self.inverse_character_location_dict = self._inverse_dict(self.character_location_dict)

        self.pos_location_dict = self.embedding_info_dicts['pos_location_dict']  ## 词性序列字典
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
                -embedding_info_dicts['relation_type_dict']:  ## 关系字典 - {'rel1': 1, 'rel2': 2}  等价于rel_type_dictTODO:
                -embedding_info_dicts['ent_seq_map_dict']: ##实体序列字典 - {"B_0":1, "I_0": 2, "E_0", 3, "B_1":4}
                -embedding_info_dicts['rel_seq_map_dict']:
                -embedding_info_dicts['entity_type_dict']: ##实体类别编号字典 - {"Time": 0, "Number": 1, "书籍": 2}
        '''

        print('start to proprocessing data...')
        tokenized_lists = []   ### word_list
        character_lists = []   ### char_list
        pos_lists = []         ### 
        label_lists = []

        # 将所有句子按照word以及char进行分离
        for cnt, item in enumerate(data):
            tokens, poss = self._tokenizer(item['input'])
            tokenized_lists.append(tokens)  
            pos_lists.append(poss)
            character_lists.append(list(item['input']))
            if cnt % 300 == 0:
                log("PreProcess %.3f \r" % (cnt / len(data)))
        
        #character_location_dict
        character_set = self._get_size_word_set(character_lists, size=None)   ### 字符层面上的一个set, 数目不大 --“北”, “京”
        character_location_dict = self._generate_word_dict(character_set)
        print('character location dict done...')

        #pos_location_dict
        pos_set = self._get_size_word_set(pos_lists, size=None)   ### 关于pos的一个set, 是在token层面，即词汇层面--“北京”
        pos_location_dict = self._generate_word_dict(pos_set)
        print('pos location dict done...')

        #entity_type_dict, ent_seq_map_dict
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

        ## (relation_type_dict)
        rel_set = self.dataset.metadata_['relation_set']
        relation_type_dict = self._generate_word_dict(rel_set)
        relation_type_dict.pop(self.PAD_TAG)
        # print('rel_set:', list(rel_set))

        ## rel_seq_map_dict
        rel_seq_map_dict = {'ELSE':0}
        for index, each_rel in enumerate(rel_set):
            rel_seq_map_dict[f"B_{each_rel}_1"] = 6*index + 1
            rel_seq_map_dict[f"I_{each_rel}_1"] = 6*index + 2
            rel_seq_map_dict[f"E_{each_rel}_1"] = 6*index + 3
            rel_seq_map_dict[f"B_{each_rel}_2"] = 6*index + 4
            rel_seq_map_dict[f"I_{each_rel}_2"] = 6*index + 5
            rel_seq_map_dict[f"E_{each_rel}_2"] = 6*index + 6
        rel_seq_map_dict[self.START_TAG] = len(rel_seq_map_dict)
        rel_seq_map_dict[self.END_TAG] = len(rel_seq_map_dict)

        # rel_seq_map_dict = {
        #     "ELSE": 0,
        #     "OBJ_B": 1,
        #     "OBJ_I": 2,
        #     "OBJ_E": 3,
        #     "SUB_B": 4,
        #     "SUB_I": 5,
        #     "SUB_E": 6
        # }
        # rel_seq_map_dict[self.START_TAG] = len(rel_seq_map_dict)
        # rel_seq_map_dict[self.END_TAG] = len(rel_seq_map_dict)

        embedding_info_dicts = {
            "character_location_dict": character_location_dict,   ## 字符层面字典 - {"北":1, "京":2}
            "pos_location_dict": pos_location_dict,   ## POS字典 - {"b":1, "ag":2}
            "relation_type_dict": relation_type_dict,  ## 关系字典 - {'rel1': 1, 'rel2': 2}
            "ent_seq_map_dict": ent_seq_map_dict,  ##实体序列字典 - {"B_0":1, "I_0": 2, "E_0", 3, "B_1":4}
            "rel_seq_map_dict": rel_seq_map_dict,  ##关系序列字典 - {"B_rel1_1":1, "I_rel1_1":2, "E_rel1_1":3}
            "entity_type_dict": entity_type_dict  ##实体类别编号字典 - {"Time": 0, "Number": 1, "书籍": 2}
        }
        self.save_preprocessed_data(self.joint_embedding_info_dicts_path, embedding_info_dicts)
        print('preprocessing finished, save successfully~')
        return embedding_info_dicts

    def transform(self, data, istest=False, data_type='ent', ratio=0.5):
        '''
        将文本数据矩阵化
        :param
            @data: list  (type: AutoKGDataset)
            @istest: 数据是训练集还是测试集
            @data_type:
                -- 'rel': relation
                -- 'ent': entity
                -- 'ent_rel': entity and relation
            @ratio: 忽略句子中未出现的关系的概率, 不要是0, 那样就没有负样本了, 也不要是1这样可能负样本过多
        
        :return
            @return_dict: dict
                *** N不是len(data), N = len(data)*n_relation_type(当ratio=1)  当含有relation的时候
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                (test-0)return_dict['y_ent_matrix']: ##(N, T), np.array, 命名实体编码序列  len = self.sentence_max_len
                (test-0)return_dict['y_rel_matrix']: ##(N, T), np.array, 关系抽取编码序列  len = self.sentence_max_len
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['relation_type_list']: ##(N), list, 每一个训练case的relation_type, 对应relation_type_dict
                return_dict['data_list']: ##(N), list 原始数据序列 - 增加postag信息
        '''
        if data_type == None:
            data_type = self.metadata_.get('data_type', 'ent_rel')  ## 默认类型'ent_rel'
        if data_type == 'rel':
            return self.transform_rel(data, istest, ratio)
        elif data_type == 'ent':
            return self.transform_ent(data, istest)
        else:  #'ent_rel'
            return self.transform_ent_rel(data, istest, ratio)
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

            ##获取文本的字符序列 --- char_list, sentence_length_list
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

            char_matrix_list.append(char_list)  ##test: sentence info
            sentence_length_list.append(sentence_length)  ##test: sentence info
            pos_matrix_list.append(pos_list)  ##test: sentence info
            y_ent_matrix_list.append(y_ent_list)  #test: all_zero
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

    def transform_rel(self, data, istest=False, ratio=0.5):
        '''
        将文本数据(目标为实体识别)矩阵化
        :param
            @data: list  (type:AutoKGDataset)  check on dataset.py
            @istest: 数据是训练集还是测试集, 如果istest=True, y_ent_matrix中都为0
        :return
            @return_dict: dict
                *** N不是len(data), N = len(data)*n_relation_type(当ratio=1)
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                (test-0)return_dict['y_rel_matrix']: ##(N, T), np.array, 关系抽取编码序列  len = self.sentence_max_len
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['relation_type_list']: ##(N), list, 每一个训练case的relation_type, 对应relation_type_dict
                return_dict['data_list']: ##(N), list 原始数据序列 - 增加postag信息
        '''
        character_location_dict = self.character_location_dict
        pos_location_dict = self.pos_location_dict
        relation_type_dict = self.relation_type_dict
        rel_seq_map_dict = self.rel_seq_map_dict
        sentence_max_len = self.sentence_max_len

        char_matrix_list = []   ## 1. 字符编码序列
        sentence_length_list = []  ##2. 句子长度序列
        pos_matrix_list = []  ## 3. POS编码序列
        relation_type_list = []  ## 4.  一个list，记录每个case的关系类型,即relation_type，通过relation_type_dict来一一对应, not useful
        y_rel_matrix_list = []  ## 5. 关系抽取编码序列, 记录每个case的两个主体(OBJ, SUB)的位置， 如 001111100022222000
        data_list = []  ## 6. 原始数据序列 - 增加postag信息

        for row_idx, d in enumerate(data):

            input_text = d['input']   ## sentence
            tokens, poss = self._tokenizer(input_text)  ##分词，获取词性
            postag = []
            for token, pos in zip(tokens, poss):
                postag.append({'word': token, 'pos': pos})
            d['postag'] = postag

            ##获取文本的字符序列 --- 1. char_list， 2. sentence_length_list
            sentence_length = 0
            char_list = np.zeros((sentence_max_len))   ##warning: 0对应<PAD>
            for col_idx, cha in enumerate(input_text):
                if col_idx < sentence_max_len:
                    if cha in character_location_dict:
                        char_list[col_idx] = character_location_dict[cha]
                    else:
                        char_list[col_idx] = character_location_dict[self.UNK_TAG]
                    sentence_length += 1

            ##获取文本的词性序列 --- 3. pos_list
            pos_list = np.zeros((sentence_max_len))
            last_word_loc = 0
            for item in d['postag']:
                word = item['word']
                pos = item['pos']
                word_start = input_text.find(word, last_word_loc)
                word_len = len(word)
                pos_list[word_start:min(word_start+word_len, sentence_max_len)] = pos_location_dict[pos]
                last_word_loc = word_start + word_len

            ##关系抽取标注序列 --- 4, relation_type_list, 5. y_rel_list
            y_rel_list = np.zeros([sentence_max_len])
            relation = 'ELSE'
            if not istest:
                for rel_dict in d['output']['relation_list']:
                    relation = rel_dict['relation']
                    rel_e1_begin = rel_dict['head_index']['begin']
                    rel_e1_end = rel_dict['head_index']['end']
                    rel_e2_begin = rel_dict['tail_index']['begin']
                    rel_e2_end = rel_dict['tail_index']['end']
                    # print(relation, rel_e1_begin, rel_e1_end, rel_e2_begin, rel_e2_end)
                    e1_tag = f'{relation}_1'
                    e2_tag = f'{relation}_2'
                    # print(e1_tag, e2_tag)
                    if rel_e1_end < sentence_max_len:
                        y_rel_list[rel_e1_begin:rel_e1_end] = self.rel_seq_map_dict[f'I_{e1_tag}']
                        y_rel_list[rel_e1_begin] = self.rel_seq_map_dict[f'B_{e1_tag}']
                        y_rel_list[rel_e1_end-1] = self.rel_seq_map_dict[f'E_{e1_tag}']
                    else:
                        continue
                    if rel_e2_end < sentence_max_len:
                        y_rel_list[rel_e2_begin:rel_e2_end] = self.rel_seq_map_dict[f'I_{e2_tag}']
                        y_rel_list[rel_e2_begin] = self.rel_seq_map_dict[f'B_{e2_tag}']
                        y_rel_list[rel_e2_end-1] = self.rel_seq_map_dict[f'E_{e2_tag}']         

            y_rel_matrix_list.append(y_rel_list)  ##test: all_zero
            char_matrix_list.append(char_list)  ##test: sentence info
            pos_matrix_list.append(pos_list)  ##test: sentence info
            sentence_length_list.append(sentence_length)  ##test: sentence info
            # relation_type_list.append(relation_type_dict[relation])  ##
            relation_type_list.append(relation)  ##
            data_list.append(d)  ##test: 只有input, 以及tag的信息

            if (row_idx % 300 == 0):
                log("Process %.3f \r" % (row_idx / len(data)))


        # print('length of data:', len(data))
        # print('legnth after transform:', len(data_list))

        ## turn all list(list) to matrix
        char_matrix = np.vstack(char_matrix_list)
        y_rel_matrix = np.vstack(y_rel_matrix_list)
        pos_matrix = np.vstack(pos_matrix_list)
        return_dict = {
            'cha_matrix': char_matrix,
            'y_rel_matrix': y_rel_matrix,
            'pos_matrix': pos_matrix,
            'sentence_length': sentence_length_list,
            'relation_type_list': relation_type_list,
            'data_list': data_list
        }
        return return_dict

    def transform_ent_rel(self, data, istest=False, ratio=0.5):
        pass

    def transform_back(self, result, data_type='ent'):
        '''
        :param
            @result: dict -- self.transform()的输出
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                return_dict['y_ent_matrix']: ##(N, T), np.array, 命名实体编码序列  len = self.sentence_max_len
                return_dict['y_rel_matrix']: ##(N, T), np.array, 关系抽取编码序列 len = self.sentence_max_len
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['relation_type_list'] ##(N,), list, 关系类型
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
                case['relation_list']
                    r = case['relation_list'][0]
                    r['relation']: 成立日期
                    r['head']: '百度'
                    r['tail']: '2016年04月08日'
        '''
        if data_type == None:
            data_type = self.metadata_.get('data_type', 'ent_rel')
        ans = []
        sentence_ore_sub_dict = defaultdict(list)
        sample_number = len(result['data_list'])
        # print('sample_number:', sample_number)

        relation = None
        sub_list = None
        obj_list = None
        entity_set = None

        for idx in range(sample_number):
            sentence = result['data_list'][idx]['input']
            # if sentence in sentence_ore_sub_dict:
            #     print(f'WARNING:repeat sentence_{idx}: {sentence}')

            if data_type == 'ent_rel' or data_type == 'rel':
                ## 对关系抽取的处理
                y_rel = result['y_rel_matrix'][idx]
                # relation = self.inverse_relation_type_dict[result['relation_type_list'][idx]]
                relations = self._obtain_sub_obj(y_rel, sentence)   

            if data_type == 'ent_rel' or data_type == 'ent':
                ## 对命名实体的处理 
                y_ent = result['y_ent_matrix'][idx]
                entity_set = self._obtain_entity(y_ent, sentence)
            sentence_ore_sub_dict[sentence] = [relations, entity_set]
            # break

        # print('total sentence_ore_sub_dict length:', len(sentence_ore_sub_dict))

        for sentence in sentence_ore_sub_dict.keys():
            ans_d = {}
            ans_d["input"] = sentence
            relation_list = set()
            all_eneity_set = set()
            relations, entity_set = sentence_ore_sub_dict[sentence]

            if data_type == "ent_rel" or data_type == "ent":
                all_eneity_set = all_eneity_set.union(set(entity_set))
            if data_type == "ent_rel" or data_type == "rel":
                for rel_type, pairs in relations.items():
                    for sub in pairs[0]:
                        for obj in pairs[1]:
                            relation_list.add(
                                json.dumps({"relation": rel_type, "head": sub, "tail": obj},
                                        ensure_ascii=False)
                            )
            # print(list(relation_list))

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

    def _obtain_sub_obj(self, y_rel, sentence):
        '''
        :param
            @y_rel: encode之后的relation seq, np.array  (T)
            @sentence: 原始句子  
        :return
            temp: dict,  里面key是关系类型， value是[[sub_list],[obj_list]]
        '''
        temp = {}  ## to save the temp relation type exist in the sentence
        triplets = []
        for idx, tag in enumerate(y_rel):
            if tag == self.rel_seq_map_dict['ELSE']:
                continue
            pos, relation_type, role = self.inverse_rel_seq_map_dict[tag].split('_')
            # print(pos, relation_type, role, tag)
            if pos == 'B':
                if relation_type not in temp:
                    temp[relation_type] = [[], []]
                temp[relation_type][int(role)-1].append(sentence[idx])  ##存储entity开始的字符
                # temp[relation_type][int(role)-1].append([idx])  ##存储entity开始的idx
            if pos == 'I' or pos == 'E':
                if relation_type in temp and len(temp[relation_type][int(role)-1]) > 0:
                    temp[relation_type][int(role)-1][-1]+=sentence[idx]  ##往后添加字符
                    # temp[relation_type][int(role)-1][-1].append(idx)  ##往后添加idx

        # for rel, v in temp.items():
        #     print(rel)
        #     print(v[0])
        #     print(v[1])
        return temp        
            
        




if __name__ == '__main__':

    # load_bert_pretrained_dict()
    result_dir = './result/'
    data_set = AutoKGDataset('./data/d4/')
    train_dataset = data_set.train_dataset[:200]

    import os
    os.makedirs(result_dir, exist_ok=True)

    data_loader = KGDataLoader2(data_set, rebuild=False, temp_dir=result_dir)
    show_dict_info(data_loader)
    
    # train_data_mat_dict = data_loader.transform_rel(train_dataset, istest=False, ratio=0)
    train_data_mat_dict = data_loader.transform(train_dataset, istest=False, data_type='rel', ratio=0)
    data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='rel', isshuffle=True)
    # # data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)
    
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
        break


    # def str_relation_fn(item):
    #     return item['relation']+'--'+item['head']+'--'+item['tail']

    # for epoch in range(2):
    #     print('EPOCH: %d' % epoch)
    #     for data_batch in data_generator:
    #         x, pos, relation, y_rel, _, lens, data_list = data_batch
    #         print(x.shape, pos.shape, y_rel.shape)    ##(batch_size, max_length)
    #         print(relation.shape, relation[:, 0])  ##(batch_size, 1)


    #         for i in range(x.shape[0]):
    #             sentence = data_list[i]['input']
    #             print(sentence)
    #             rel_list = data_list[i]['output']['relation_list']
    #             # print([(i, sentence[i]) for i in range(len(sentence))])
    #             for rel in rel_list:
    #                 print(str_relation_fn(rel))

    #             y_rel_i = y_rel[i]
    #             rel_pred = [data_loader.inverse_rel_seq_map_dict[i] for i in y_rel_i]
    #             print(y_rel_i)
    #             print(rel_pred)
    #             break

    #         break
    #     break