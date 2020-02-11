# !-*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

import transformers

# from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import os
import math
import json
import numpy as np

torch.manual_seed(1)

class BERT_MLP(nn.Module):
    def __init__(self, params={}, show_param=False):
        '''
        :param - dict
            param['embedding_dim']
            param['hidden_dim']
            param['n_tags']
            param['n_words']
            param['start_idx']  int, <start> tag index for entity tag seq
            param['end_idx']   int, <end> tag index for entity tag seq
            param['use_cuda']
            param['dropout_prob']
            param['lstm_layer_num']
        '''
        super(BERT_MLP, self).__init__()
        self.params = params
        self.embedding_dim = self.params.get('embedding_dim', 768)
        # self.hidden_dim = self.params['hidden_dim']
        # assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.n_tags = self.params.get('n_tags', 45)
        # self.n_words = self.params.get('n_words', 0)
        # self.start_idx = self.params['start_idx']
        # self.end_idx = self.params['end_idx']
        self.use_cuda = self.params.get('use_cuda', 0)
        # self.dropout_prob = self.params.get('dropout_prob', 0.05)
        # self.lstm_layer_num = self.params.get('lstm_layer_num', 1)

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'embedding_dim: {self.embedding_dim}', 1)
        # log(f'hidden_dim: {self.hidden_dim}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        # log(f'lstm_layer_num: {self.lstm_layer_num}', 1)
        # log(f'dropout_prob: {self.dropout_prob}', 1)  
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        # self.word_embeds = nn.Embedding(self.n_words, self.embedding_dim)
        # self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, batch_first=True, num_layers=self.lstm_layer_num, dropout=self.dropout_prob, bidirectional=True)
        self.hidden2tag = nn.Linear(self.embedding_dim, self.n_tags)
        # crf_params = {'n_tags':self.n_tags, 'start_idx':self.start_idx, 'end_idx':self.end_idx, 'use_cuda':self.use_cuda}
        # self.crf = CRF(crf_params)

        # self.bert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bert = transformers.BertModel.from_pretrained('bert-base-chinese')

    def reset_parameters(self):        
        # I.xavier_normal_(self.word_embeds.weight.data)
        # self.lstm.reset_parameters()
        # stdv = 1.0 / math.sqrt(self.hidden_dim)
        # for weight in self.lstm.parameters():
        #     I.uniform_(weight, -stdv, stdv)
        I.xavier_normal_(self.hidden2tag.weight.data)
        # self.crf.reset_parameters()

    def _loss(self, x, y_ent, lens, use_cuda=None):
        '''
        loss function: neg_log_likelihood
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (batch_size), torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]

        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)
        att_mask = self._generate_mask(lens, max_len=T)
        embeds = self.bert(words_tensor, attention_mask=att_mask)[0]  #(batch_size, T, n_embed)
        # print(embeds.shape)
        labels = self.hidden2tag(embeds)  #(batch_size, T, n_tags)
        labels = labels.transpose(2, 1)  #(batch_size, n_tags, T)
        # print(labels.shape)

        targets = self._to_tensor(y_ent, use_cuda)  ##(batch_size, T)
        loss_fn = nn.CrossEntropyLoss()  ###labels need to be [N, C, d1, d2, ...] and targets [N, d1, d2, ...]
        loss = loss_fn(labels, targets)

        # print(loss.shape)
        # print(loss)
        
        # logits = self._get_lstm_features(x, lens)
        # log_norm_score = self.crf.log_norm_score(logits, lens)
        # path_score = self.crf.path_score(logits, y_ent, lens)
        # loss = log_norm_score - path_score
        return loss

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
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]

        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        lens = self._to_tensor(lens, use_cuda)
        att_mask = self._generate_mask(lens, max_len=T)

        embeds = self.bert(words_tensor, attention_mask=att_mask)[0]  #(batch_size, T, n_embed)
        # print(embeds.shape)
        labels = self.hidden2tag(embeds)  #(batch_size, T, n_tags)
        # print(labels.shape)
        label_max, label_argmax = labels.max(dim=2)

        # print(label_argmax.shape)
        paths = label_argmax
        return paths

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        if os.path.exists(path):
            if self.use_cuda:
                self.load_state_dict(torch.load(path))
            else:
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            pass


    def train_model(self, data_loader: KGDataLoader, train_dataset=None, eval_dataset=None, hyper_param={}, use_cuda=None):
        '''
        :param
            @data_loader: (KGDataLoader),
            @result_dir: (str) path to save the trained model and extracted dictionary
            @hyper_param: (dict)
                @hyper_param['EPOCH']
                @hyper_param['batch_size']
                @hyper_param['learning_rate']
                @hyper_param['visualize_length']   #num of batches between two check points
                @hyper_param['isshuffle']
                @hyper_param['result_dir']
                @hyper_param['model_name']
        :return
            @loss_record, 
            @score_record
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        EPOCH = hyper_param.get('EPOCH', 3)
        BATCH_SIZE = hyper_param.get('batch_size', 4)
        LEARNING_RATE = hyper_param.get('learning_rate', 1e-2)
        visualize_length = hyper_param.get('visualize_length', 10)
        result_dir = hyper_param.get('result_dir', './result/')
        model_name = hyper_param.get('model_name', 'model.p')
        is_shuffle = hyper_param.get('isshuffle', True)
        DATA_TYPE = 'ent'
        

        train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset

        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        old_train_dict_path = os.path.join(result_dir, 'train_data_mat_dict.pkl')
        if os.path.exists(old_train_dict_path):
            train_data_mat_dict = data_loader.load_preprocessed_data(old_train_dict_path)
            log('Reload preprocessed data successfully~')
        else:
            train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
            data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        data_generator = Batch_Generator(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        if use_cuda:
            print('use cuda=========================')
            self.cuda()
        

        all_cnt = len(train_data_mat_dict['cha_matrix'])
        log(f'{model_name} Training start!', 0)
        loss_record = []
        score_record = []

        eval_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
        for epoch in range(EPOCH):
            self.train()
            log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
            loss = 0.0
            for cnt, data_batch in enumerate(data_generator):
                x, pos, _, _, y_ent, lens, data_list = data_batch
                optimizer.zero_grad()
                # nll = self._loss(x, y_ent, lens)
                # sub_loss = nll.mean()
                sub_loss = self._loss(x, y_ent, lens)
                loss_avg = sub_loss
                loss_avg.backward()
                optimizer.step()

                # loss_avg = (nll/self._to_tensor(lens, use_cuda)).mean()
                loss += loss_avg
                if use_cuda:
                    loss_record.append(loss_avg.cpu().item())
                else:
                    loss_record.append(loss_avg.item())

                if (cnt+1) % visualize_length == 0:
                    loss_cur = loss / visualize_length
                    log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
                    loss = 0.0

                if cnt+1 % 100 == 0:
                    save_path = os.path.join(result_dir, model_name)
                    self.save_model(save_path)
                    print('Checkpoint saved successfully')

                #     print(data_list[0]['input'])
                #     pre_paths, pre_scores = self._output(x, lens)
                #     print('predict-path')
                #     print(pre_paths[0])
                #     print('target-path')
                #     print(y_ent[0])
            save_path = os.path.join(result_dir, model_name)
            self.save_model(save_path)
            print('Checkpoint saved successfully')

            temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_param, use_cuda=use_cuda)
            score_record.append(temp_score)
        return loss_record, score_record


    @torch.no_grad()
    def predict(self, data_loader: KGDataLoader, data_set=None, hyper_param={}, use_cuda=False):
        '''
        预测出 test_data_mat_dict['y_ent_matrix']中的内容，重新填写进该matrix, 未预测之前都是0
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  ##默认4
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认./result/
        :return
            @result: list
                case = result[0]
                case['input']
                case['entity_list']
                    e = case['entity_list'][0]  
                    e['entity']:2016年04月08日
                    e['entity_type']:Date
                    e['entity_index']['begin']:13
                    e['entity_index']['end']:24
        '''
        BATCH_SIZE = hyper_param.get('batch_size', 64)
        ISSAVE = hyper_param.get('issave', False)
        result_dir = hyper_param.get('result_dir', './result/')
        DATA_TYPE = 'ent'

        if data_set is None:
            test_dataset = data_loader.dataset.test_dataset
        else:
            test_dataset = data_set

        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        old_test_dict_path = os.path.join(result_dir, 'test_data_mat_dict.pkl')
        if os.path.exists(old_test_dict_path):
            test_data_mat_dict = data_loader.load_preprocessed_data(old_test_dict_path)
            log('Reload preprocessed data successfully~')
        else:
            test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
            data_loader.save_preprocessed_data(old_test_dict_path, test_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        data_generator = Batch_Generator(test_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=False)
        
        if use_cuda:
            print('use cuda=========================')
            self.cuda()
        self.eval()   #disable dropout layer and the bn layer

        total_output_ent = []
        all_cnt = len(test_data_mat_dict['cha_matrix'])
        log(f'Predict start!', 0)
        for cnt, data_batch in enumerate(data_generator):
            x, pos, _, _, _, lens, _ = data_batch
            pre_paths = self._output(x, lens)  ##pre_paths, (batch_size, T), torch.tensor  
            if use_cuda:
                pre_paths = pre_paths.data.cpu().numpy().astype(np.int)
            else:
                pre_paths = pre_paths.data.numpy().astype(np.int)
            total_output_ent.append(pre_paths)
            
            if (cnt+1) % 10 == 0:
                log(f'[PREDICT] step {(cnt+1)*BATCH_SIZE}/{all_cnt}', 1)

        ## add mask when the ent seq idx larger than sentance length
        pred_output = np.vstack(total_output_ent)   ###(N, max_length), numpy.array
        len_list = test_data_mat_dict['sentence_length']   ###(N), list
        pred_output = self._padding_mask(pred_output, len_list[:len(pred_output)])

        ## transform back to the dict form
        test_data_mat_dict['y_ent_matrix'] = pred_output
        result = data_loader.transform_back(test_data_mat_dict, data_type='ent')
        
        ## save the result
        if ISSAVE and result_dir:
            save_file = os.path.join(result_dir, 'predict.json')
            with open(save_file, 'w') as f:
                for data in result:
                    temps = json.dumps(data, ensure_ascii=False)
                    f.write(temps+'\n')
            log(f'save the predict result in {save_file}')
        return result

    @torch.no_grad()
    def eval_model(self, data_loader: KGDataLoader, data_set=None, hyper_param={}, use_cuda=False):
        '''
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  #默认64
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认./result WARNING:可能报错如果result目录不存在的话
        :return
            @precision_s, 
            @recall_s, 
            @f1_s
        '''
        def dict2str(d):
            ## 将entity 从字典形式转化为str形式方便比较
            res = d['entity']+':'+d['entity_type']+':'+str(d['entity_index']['begin'])+'-'+str(d['entity_index']['end'])
            return res

        def calculate_f1(pred_cnt, tar_cnt, correct_cnt):
            precision_s = round(correct_cnt / (pred_cnt + 1e-8), 3)
            recall_s = round(correct_cnt / (tar_cnt + 1e-8), 3)
            f1_s = round(2*precision_s*recall_s / (precision_s + recall_s + 1e-8), 3)
            return precision_s, recall_s, f1_s


        eva_data_set = data_loader.dataset.dev_dataset if data_set is None else data_set

        pred_result = self.predict(data_loader, eva_data_set, hyper_param, use_cuda) ###list(dict), 预测结果
        target = eva_data_set  ###list(dict)  AutoKGDataset, 真实结果

        pred_cnt = 0
        tar_cnt = 0
        correct_cnt = 0
        cnt_all = len(eva_data_set)
        log('Eval start')
        for idx in range(cnt_all):
            sentence = pred_result[idx]['input']
            pred_list = pred_result[idx]['entity_list']
            tar_list = target[idx]['output']['entity_list']
   
            str_pred_set = set(map(dict2str, pred_list))
            str_tar_set = set(map(dict2str, tar_list))
            common_set = str_pred_set.intersection(str_tar_set)

            pred_cnt += len(str_pred_set)
            tar_cnt += len(str_tar_set)
            correct_cnt += len(common_set)

            if (idx+1) % 1000 == 0:
                precision_s, recall_s, f1_s = calculate_f1(pred_cnt, tar_cnt, correct_cnt)
                log(f'[EVAL] step {idx+1}/{cnt_all} | precision: {precision_s} | recall: {recall_s} | f1 score: {f1_s}', 1)

        precision_s, recall_s, f1_s = calculate_f1(pred_cnt, tar_cnt, correct_cnt)
        print('='*100)
        log(f'[FINAL] | precision: {precision_s} | recall: {recall_s} | f1 score: {f1_s}', 0)
        print('='*100)
        return (precision_s, recall_s, f1_s)

    @staticmethod
    def _to_tensor(x, use_cuda=False):
        if use_cuda:
            return torch.tensor(x, dtype=torch.long).cuda()
        else:
            return torch.tensor(x, dtype=torch.long)     

    @staticmethod
    def _padding_mask(arr, lens):
        '''
        将超出句子长度部分的array mask成0
        :param
            @arr: (N, T), np.array
            @lens: (N), list
        :return 
            @arr: (N, T), np.array, arr after mask
        '''
        assert(len(arr) == len(lens)), 'len(arr) must be equal to len(lens), otherwise cannot mask'
        for idx in range(len(arr)):
            arr[idx, lens[idx]:] = 0
        return arr

    @staticmethod
    def _generate_mask(lens, max_len=None, use_cuda=False):
        '''
        返回一个mask, 遮住<pad>部分的无用信息.
        :param
            @lens: (batch_size), torch.tensor, the lengths of each sentence
            @max_len: int, the max length of the sentence - T
        :return 
            @mask: (batch_size, max_len)
        '''
        # use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size = lens.shape[0]
        if max_len is None:
            max_len = lens.max()
        ranges = torch.arange(0, max_len).long()  #(max_len)
        if lens.is_cuda:
            ranges = ranges.cuda()
        ranges = ranges.unsqueeze(0).expand(batch_size, max_len)   #(batch_size, max_len)
        lens_exp = lens.unsqueeze(1).expand_as(ranges)  #(batch_size, max_len)
        mask = ranges < lens_exp
        return mask

if __name__ == '__main__':
    model_param = {
        'num_labels':45,
        'use_cuda':True   #True
    }
    model = BERT_NER(params=model_param)
    # model.load_model('./result/')

    data_set = AutoKGDataset('./d1/')
    # train_dataset = data_set.train_dataset[:20]
    # eval_dataset = data_set.dev_dataset[:10]
    train_dataset = data_set.train_dataset
    eval_dataset = data_set.dev_dataset

    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')

    print(data_loader.embedding_info_dicts['entity_type_dict'])

    train_param = {
        'learning_rate':5e-5,
        'EPOCH':5,  #30
        'batch_size':64,  #64
        'visualize_length':20,  #10
        'result_dir': './result/',
        'isshuffle': True
    }
    model.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)


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
