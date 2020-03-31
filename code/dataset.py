import os
import json
import codecs
from sklearn.model_selection import train_test_split

from glob import glob as ls
from common import get_logger

VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

## data example
"""
    train_dataset, test_dataset:
    [
        {
            "input":"联磺甲氧苄啶片,通用名称：联磺甲氧苄啶片....", # input sentence
            "output":{
                "entity_list":[
                    {
                        "entity_type": "药物成分",
                        "entity": "磺胺甲唑",
                        "entity_index": {"begin": 115, "end": 119}
                    },
                    {"entity_type": "病症",
                        "entity": "肠道感染",
                            "entity_index": {"begin": 201, "end": 205}
                    },
                    ....
                ],
                "relation_list":[
                    {
                        "relation": "包含成分",
                            "head": "联磺甲氧苄啶片",
                            "head_index": {"begin": 0, "end": 7},
                            "tail": "磺胺甲唑",
                            "tail_index": {"begin": 115, "end": 119}
                    },
                    ......
                ]
            }
        },
    ]
"""

class AutoKGDataset:
    def __init__(self, dataset_dir, random_state=1):
        '''
        :param
            @dataset_dir: the path of the dataset
        :return
            @self.dataset_name
            @self.dataset_dir
            @self.all_train_dataset
            @self.train_dataset     ***
            @self.dev_dataset       ***
            @self.test_dataset      ***
            @self.metadata_: dict   ***
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
        self.dataset_name_ = dataset_dir
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self._read_metadata(
            os.path.join(dataset_dir, "info.json")
        )
        self.all_train_dataset = self._read_dataset(
            os.path.join(self.dataset_dir_, 'train.data')
        )
        self.train_dataset, self.dev_dataset = train_test_split(self.all_train_dataset, test_size=0.1, random_state=random_state)
        self.dev_dataset = self.check_repeat_sentence(self.dev_dataset)   ##remove the repeat sentence

        self.test_dataset = self._read_dataset(
            os.path.join(self.dataset_dir_, 'test.data')
        )
        self.test_dataset = self.check_repeat_sentence(self.test_dataset)  ##remove the repeat sentence

        self._generate_metadata()

    def _generate_metadata(self):
        '''
        :return
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
        chars = set()
        rels = set()
        ens = set()
        sen_len = 0
        sen_cnt = 0
        max_sen_len = 0
        for idx, sample in enumerate(self.all_train_dataset):
            '''
            sample['input']:  sentence
            sample['output']['entity_list']: entity_list
            sample['output']['relation_list']: relation_list
            entity['entity_type']: entity_type
            entity['entity']: real entity body
            entity['entity_index']['begin']: entity_index_begin
            entity['entity_index']['end']: entity_index_end
            '''
            sen_cnt += 1
            max_sen_len = max(max_sen_len, len(sample['input']))
            sen_len += len(sample['input'])
            for c in sample['input']:
                chars.add(c)
            if self.metadata_['data_type'] == 'ent' or self.metadata_['data_type'] == 'ent_rel':
                for e in sample['output']['entity_list']:
                    ens.add(e['entity_type'])
            if self.metadata_['data_type'] == 'rel' or self.metadata_['data_type'] == 'ent_rel':
                for r in sample['output']['relation_list']:
                    rels.add(r['relation'])
        
        self.metadata_['char_size'] = len(chars)
        self.metadata_['char_set'] = chars
        self.metadata_['entity_size'] = len(ens)
        self.metadata_['entity_set'] = ens
        self.metadata_['relation_size'] = len(rels)
        self.metadata_['relation_set'] = rels

        self.metadata_['max_sen_len'] = max_sen_len
        self.metadata_['avg_sen_len'] = int(sen_len / sen_cnt)
        self.metadata_['mode_sen_len'] = self.count_length(threshold=98)

        self.metadata_['train_num'] = len(self.all_train_dataset)
        self.metadata_['test_num'] = len(self.test_dataset)

    def get_metadata(self):
        return self.metadata_   
             
    def count_length(self, threshold=95):
        s_len = [0]*31
        for idx, sample in enumerate(self.all_train_dataset):
            sen_len_id = len(sample['input'])//10
            if sen_len_id > 30:
                s_len[30] += 1
            else:
                s_len[sen_len_id] += 1

        total_n = sum(s_len)
        ratio_arr = []
        cur = 0
        res, res_ratio = -1, -1
        for i in range(len(s_len)):
            cur += s_len[i]
            temp = round(100*cur/total_n, 2)
            ratio_arr.append(temp)
            if temp > threshold and res < 0:
                res = 10*(i+1)
                res_ratio = temp
        
        # print(list(enumerate(ratio_arr)))
        # print(res, res_ratio)
        return res

    @staticmethod
    def _read_metadata(metadata_path):
        return json.load(open(metadata_path))

    @staticmethod
    def _read_dataset(dataset_path):
        data = []
        with codecs.open(dataset_path, 'r', 'utf-8') as fout:
            for line in fout:
                data.append(json.loads(line))
        # print(len(data))
        # data = self.check_repeat_sentence(data)   #TODO: remove the repeat sentence in training dataset
        return data

    @staticmethod
    def check_repeat_sentence(dataset):
        new_dataset = []
        seen_sentence = set()
        cnt = 0
        for item in dataset:
            if item['input'] in seen_sentence:
                # print(f"remove repeat sentence: {item['input']}")
                cnt += 1
                continue
            seen_sentence.add(item['input'])
            new_dataset.append(item)
        print(f'remove repeat sentence {cnt}')
        return new_dataset


def inventory_data(input_dir):
    '''
    :return 
        @training_names: (list) - all datasets in the input directory in alphabetical order
    '''
    training_names = ls(os.path.join(input_dir, '*.data'))
    training_names = [name.split('/')[-1] for name in training_names]
    
    if len(training_names) == 0:
        LOGGER.warning('WARNING: Inventory data - No data file found')
    return sorted(training_names)


### not used here
def get_dataset(args):
    datanames = inventory_data(args.dataset_dir)
    datanames = [x for x in datanames if x.endswith('.data')]
    if len(datanames) != 1:
        raise ValueError("{} datasets found in dataset_dir={}!\n"
                        "Please put only ONE dataset under dataset_dir.".format(len(datanames), args.dataset_dir))

    basename = datanames[0]
    dataset_name = basename[:-5]
    dataset = AutoKGDataset(os.path.join(args.dataset_dir, basename))
    return dataset, dataset_name


if __name__ == '__main__':
    dataset = AutoKGDataset('./d1/', random_state=1)
    train, dev = dataset.train_dataset, dataset.dev_dataset
    test = dataset.test_dataset

    names = inventory_data('./d1')
    # print(names)

    from utils import show_metadata
    meta_info = dataset.get_metadata()
    show_metadata(meta_info)