import os
import json
import pickle
from collections import defaultdict
import numpy as np
import random

from dataset import AutoKGDataset

def my_lr_lambda(epoch):
    # return 1/(1+0.05*epoch)
    return 0.5**(epoch)

def log(entry, nesting_level=0):
    space = '-'*(4*nesting_level)
    print(f"{space}{entry}")

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
