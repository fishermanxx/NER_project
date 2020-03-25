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


from utils import BaseLoader, log, load_bert_pretrained_dict
from dataset import AutoKGDataset

def show_dict_info(dataloader):
    '''
    :param
        dataloader: KGDataLoader类
    '''
    print('===================info about the data======================')
    print('entity type dict length:', len(dataloader.embedding_info_dicts['entity_type_dict']))
    print('entity seq dict length:', len(dataloader.embedding_info_dicts['ent_seq_map_dict']))
    print('relation type dict length:', len(dataloader.embedding_info_dicts['relation_type_dict']))  ##TODO: change
    # print('subject seq dict length:', len(dataloader.embedding_info_dicts['sub_seq_map_dict']))
    # print('object seq dict length:', len(dataloader.embedding_info_dicts['objr_seq_map_dict']))
    print('character location dict length:', len(dataloader.embedding_info_dicts['character_location_dict']))
    print('pos location dict length:', len(dataloader.embedding_info_dicts['pos_location_dict']))
    print('============================================================')
    print()

class KGDataLoader3(BaseLoader):
    def __init__(self, dataset: AutoKGDataset, rebuild=True, temp_dir=None):
        '''
        :param 
            @rebuild: 是否重新建立各类字典
            @istest: 是否是测试集
            @dataset: 通过AutoKGDataset得到的数据集
        '''
        super(KGDataLoader3, self).__init__()
        self.dataset = dataset
        self.temp_dir = temp_dir
        self.metadata_ = dataset.metadata_
        # self.sentence_max_len = max(200, min(self.metadata_['avg_sen_len'], 200))  ##TODO:
        self.sentence_max_len = self.metadata_['mode_sen_len']  ##TODO:
        
        self.joint_embedding_info_dicts_path = os.path.join(temp_dir, "joint_embedding_info_dict.pkl")
        if (not rebuild) and os.path.exists(self.joint_embedding_info_dicts_path):
            self.embedding_info_dicts = self.load_preprocessed_data(
                self.joint_embedding_info_dicts_path
            ) 
        else:
            self.embedding_info_dicts = self._preprocess_data(self.dataset.all_train_dataset)

        self.ent_seq_map_dict = self.embedding_info_dicts['ent_seq_map_dict'] ## 实体序列字典
        self.inverse_ent_seq_map_dict = self._inverse_dict(self.ent_seq_map_dict)

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
                -embedding_info_dicts['sub_seq_map_dict']: ##subject主语序列字典
                -embedding_info_dicts['objr_seq_map_dict']: ##object_relation 渭宾短语序列字典
                -embedding_info_dicts['entity_type_dict']: ##实体类别编号字典 - {"Time": 0, "Number": 1, "书籍": 2}
        '''

        print('start to proprocessing data...')
        tokenized_lists = []   ### word_list
        character_lists = []   ### char_list
        pos_lists = []         ### 
        label_lists = []

        # 将所有句子按照word以及char进行分离
        for cnt, item in enumerate(data):
            ##TODO: not used pos information
            # tokens, poss = self._tokenizer(item['input'])
            # tokenized_lists.append(tokens)  
            # pos_lists.append(poss)
            character_lists.append(list(item['input']))
            if cnt % 300 == 0:
                log("PreProcess %.3f \r" % (cnt / len(data)))
        
        #character_location_dict
        character_set = self._get_size_word_set(character_lists, size=None)   ### 字符层面上的一个set, 数目不大 --“北”, “京”
        character_location_dict = self._generate_word_dict(character_set)
        print('character location dict done...')

        ###pos_location_dict TODO: not used
        pos_location_dict = {}
        # pos_set = self._get_size_word_set(pos_lists, size=None)   ### 关于pos的一个set, 是在token层面，即词汇层面--“北京”
        # pos_location_dict = self._generate_word_dict(pos_set)
        # print('pos location dict done...')

        #entity_type_dict, ent_seq_map_dict
        ent_seq_map_dict = {'ELSE': 0}
        entity_set = self.dataset.metadata_['entity_set']
        entity_type_dict = {}
        for index, each_entity in enumerate(entity_set):
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

        embedding_info_dicts = {
            "character_location_dict": character_location_dict,   ## 字符层面字典 - {"北":1, "京":2}
            "pos_location_dict": pos_location_dict,   ## POS字典 - {"b":1, "ag":2}
            "relation_type_dict": relation_type_dict,  ## 关系字典 - {'rel1': 1, 'rel2': 2}
            "entity_type_dict": entity_type_dict,  ##实体类别编号字典 - {"Time": 0, "Number": 1, "书籍": 2}
            "ent_seq_map_dict": ent_seq_map_dict,  ##实体序列字典 - {"B_0":1, "I_0": 2, "E_0", 3, "B_1":4}
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
            return self.transform_rel(data, istest)
        elif data_type == 'ent':
            return self.transform_ent(data, istest)
        else:  #'ent_rel'
            return self.transform_ent_rel(data, istest)
            # return self.transform_ent(data, istest)   

    def transform_ent(self, data, istest=False):
        pass

    def transform_rel(self, data, istest=False):
        '''
        将文本数据(目标为实体识别)矩阵化
        :param
            @data: list  (type:AutoKGDataset)  check on dataset.py
            @istest: 数据是训练集还是测试集, 如果istest=True, y_ent_matrix中都为0
        :return
            @return_dict: dict
                *** N不是len(data), N = len(data)*n_relation_type(当ratio=1)
                return_dict['cha_matrix']: ##(N, T), np.array, 字符编码序列  len = self.sentence_max_len
                return_dict['y_rel_list']: ##(N), list, 每个case是字典
                return_dict['pos_matrix']: ##(N, T), np.array, POS编码序列  len = self.sentence_max_len
                return_dict['sentence_length']: ##(N), list, 句子长度序列
                return_dict['data_list']: ##(N), list 原始数据序列 - 增加postag信息
        '''
        character_location_dict = self.character_location_dict
        pos_location_dict = self.pos_location_dict
        relation_type_dict = self.relation_type_dict
        sentence_max_len = self.sentence_max_len
        n_rel_types = len(relation_type_dict)

        char_matrix_list = []   ## 1. 字符编码序列
        sentence_length_list = []  ##2. 句子长度序列
        pos_matrix_list = []  ## 3. POS编码序列
        y_rel_list = [] ## list, 每个case是字典，内容如下{'sub1':[obj1, obj2,...], 'sub2':[obj1, obj2,...]}, 其中sub1形式为(sidx, eidx), obj形式为(sidx, eidx, ridx)
        data_list = []  ## 6. 原始数据序列 - 增加postag信息

        sub_count = 0
        for row_idx, d in enumerate(data):

            input_text = d['input']   ## sentence

            ##TODO: not used
            # tokens, poss = self._tokenizer(input_text)  ##分词，获取词性
            # postag = []
            # for token, pos in zip(tokens, poss):
            #     postag.append({'word': token, 'pos': pos})
            # d['postag'] = postag

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

            ##获取文本的词性序列 --- pos_list TODO: not used
            pos_list = []
            # pos_list = np.zeros((sentence_max_len))
            # last_word_loc = 0
            # for item in d['postag']:
            #     word = item['word']
            #     pos = item['pos']
            #     word_start = input_text.find(word, last_word_loc)
            #     word_len = len(word)
            #     pos_list[word_start:min(word_start+word_len, sentence_max_len)] = pos_location_dict[pos]
            #     last_word_loc = word_start + word_len

            spoes = {}
            
            if not istest:
                for rel_dict in d['output']['relation_list']:
                    relation = rel_dict['relation']
                    ridx = relation_type_dict[relation]
                    # print(relation, ridx)
                    sub_begin = rel_dict['head_index']['begin']
                    sub_end = rel_dict['head_index']['end']
                    obj_begin = rel_dict['tail_index']['begin']
                    obj_end = rel_dict['tail_index']['end']
                    
                    # sub_begin = max(0, sub_begin)
                    # obj_begin = max(0, obj_begin)
                    if sub_begin < 0 or obj_begin < 0:
                        continue
                    sub = (sub_begin, sub_end)
                    obj = (obj_begin, obj_end, ridx)

                    if sub_end > sentence_max_len or obj_end > sentence_max_len or sub_end <= sub_begin or obj_end <= obj_begin:
                        continue

                    if sub not in spoes:
                        spoes[sub] = [obj]
                    else:
                        spoes[sub].append(obj)
                if len(spoes) == 0:
                    continue
            y_rel_list.append(spoes)
            char_matrix_list.append(char_list)  ##test: sentence info
            pos_matrix_list.append(pos_list)  ##test: sentence info
            sentence_length_list.append(sentence_length)  ##test: sentence info
            data_list.append(d)  ##test: 只有input, 以及tag的信息
            sub_count += len(spoes)


            if (row_idx+1) % 300 == 0:
                log("Process %.3f \r" % (row_idx / len(data)))

        ## turn all list(list) to matrix
        char_matrix = np.vstack(char_matrix_list)  ##(N, T)
        pos_matrix = np.vstack(pos_matrix_list)
        return_dict = {
            'cha_matrix': char_matrix,
            'pos_matrix': pos_matrix,
            'sentence_length': sentence_length_list,
            'y_rel_list': y_rel_list,
            'data_list': data_list,
            'total_sub': sub_count   ##TODO:
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
                return_dict['y_rel_list']: ##(N), list(dict), 关系抽取编码序列  {'sub1':objr_list, 'sub2':objr_list, ..}
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

        entity_set = None

        for idx in range(sample_number):
            sentence = result['data_list'][idx]['input']
            # if sentence in sentence_ore_sub_dict:
            #     print(f'WARNING:repeat sentence_{idx}: {sentence}')

            if data_type == 'ent_rel' or data_type == 'rel':
                ## 对关系抽取的处理
                y_rel = result['y_rel_list'][idx]
                relations = self._obtain_sub_obj(y_rel, sentence)   

            if data_type == 'ent_rel' or data_type == 'ent':
                ## 对命名实体的处理 
                y_ent = result['y_ent_matrix'][idx]
                entity_set = self._obtain_entity(y_ent, sentence)
            sentence_ore_sub_dict[sentence] = [relations, entity_set]
            
        for sentence in sentence_ore_sub_dict.keys():
            ans_d = {}
            ans_d["input"] = sentence
            relation_list = set()
            all_eneity_set = set()
            relations, entity_set = sentence_ore_sub_dict[sentence]

            if data_type == "ent_rel" or data_type == "ent":
                all_eneity_set = all_eneity_set.union(set(entity_set))
            if data_type == "ent_rel" or data_type == "rel":
                for r in relations:
                    relation_list.add(r)
            
            if data_type == "ent_rel" or data_type == "rel":
                ans_d["relation_list"] = [json.loads(s) for s in relation_list]
            if data_type == "ent_rel" or data_type == "ent":
                ans_d["entity_list"] = list(json.loads(i) for i in all_eneity_set)  ##将string全部转化成字典形式保存
            ans.append(ans_d)

        return ans

    def _obtain_entity(self, y_ent, sentence):
        pass

    def _obtain_sub_obj(self, y_rel, sentence):
        '''
        :param
            @y_rel: encode之后的relation seq, dict: {'sub1':objr_list, 'sub2':objr_list, ..}
            @sentence: 原始句子  
        :return
            temp: list, {"relation": rel_type, "head": sub, "tail": obj}
        '''
        relations = []
        for sub, objr_list in y_rel.items():
            sub_s = sentence[sub[0]:sub[1]]
            for objr in objr_list:
                obj_s = sentence[objr[0]:objr[1]]
                r_s = self.inverse_relation_type_dict[objr[2]]
                if len(sub_s)>0 and len(obj_s)>0:
                    relations.append(
                        json.dumps({"relation": r_s, "head": sub_s, "tail": obj_s}, ensure_ascii=False)
                    )
                else:
                    print(f"sub-{sub_s}-{sub[0]}-{sub[1]}, obj_s-{obj_s}-{objr[0]}-{objr[1]}, sentence-{sentence[:10]}, len-{len(sentence)}")
        return relations

class Batch_Generator3(nn.Module):
    def __init__(self, data_dict, batch_size=16, data_type='ent', isshuffle=True):
        '''
        :param
            @data_dict: dict --- KGDataLoader.transform()的返回值
                data_dict['cha_matrix']: ## 字符编码序列
                data_dict['y_ent_matrix']: ## 命名实体编码序列
                data_dict['y_sub_matrix']: 
                data_dict['y_obj_matrix']: 
                data_dict['relation_type_list']
                data_dict['pos_matrix']: ## POS编码序列
                data_dict['sentence_length']: ##句子长度序列
                data_dict['data_list']: ## 原始数据序列 - 增加postag信息
            @batch_size: 
            @data_type: {'ent', 'rel', 'ent_rel'}
        '''
        self.x = data_dict['cha_matrix']
        self.number = len(self.x)
        self.y_rel = data_dict.get('y_rel_list', [None]*self.number)
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
            @y_rel: (batch_size,)  list(dict)
            @y_ent: (batch_size, max_length)
            @sentence_length: (batch_size) 
            @data_list: (batch_size)
        '''
        if self.current >= len(self.x):
            self.current = 0
            if self.isshuffle:
                self.x, self.pos, self.y_rel, self.y_ent, self.sentence_length, self.data_list = sklearn.utils.shuffle(
                    self.x,
                    self.pos,
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
                       self.y_ent[old_current:to_, :], \
                       self.sentence_length[old_current:to_], \
                       self.data_list[old_current:to_]

            elif self.data_type == 'rel':
                return self.x[old_current:to_, :], \
                       self.pos[old_current:to_, :], \
                       self.y_rel[old_current:to_], \
                       None, \
                       self.sentence_length[old_current:to_], \
                       self.data_list[old_current:to_]
            
            else:
                return self.x[old_current:to_, :], \
                       self.pos[old_current:to_, :], \
                       self.y_rel[old_current:to_], \
                       self.y_ent[old_current:to_, :], \
                       self.sentence_length[old_current:to_], \
                       self.data_list[old_current:to_]

if __name__ == '__main__':

    # load_bert_pretrained_dict()
    result_dir = './result/'
    data_set = AutoKGDataset('./data/d4/')
    train_dataset = data_set.train_dataset[:50]

    import os
    os.makedirs(result_dir, exist_ok=True)

    data_loader = KGDataLoader3(data_set, rebuild=False, temp_dir=result_dir)
    show_dict_info(data_loader)
    print(list(data_loader.relation_type_dict.items()))
    
    # train_data_mat_dict = data_loader.transform_rel(train_dataset, istest=False, ratio=0)
    train_data_mat_dict = data_loader.transform(train_dataset, istest=False, data_type='rel')

    data_generator = Batch_Generator3(train_data_mat_dict, batch_size=4, data_type='rel', isshuffle=True)
    # # data_generator = Batch_Generator3(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)

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


