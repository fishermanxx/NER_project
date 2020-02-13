from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from torch import nn
import json
import numpy as np
from transformers import BertForTokenClassification, BertTokenizer

from torch.optim.lr_scheduler import LambdaLR
from utils import my_lr_lambda

class BERT_NER(nn.Module):
    def __init__(self, params={}, show_param=False):
        '''
        :param
            @params['n_tags']
            @params['use_cuda']
        '''
        super(BERT_NER, self).__init__()
        self.num_labels = params.get('n_tags', 45)
        self.use_cuda = params.get('use_cuda', False)
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

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    # def save_model(self, path: str):
    #     self.model.save_pretrained(path)

    def load_model(self, path: str):
        if os.path.exists(path):
            if self.use_cuda:
                self.load_state_dict(torch.load(path))
            else:
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            pass

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

    def train_model(self, data_loader: KGDataLoader, hyper_param={}, train_dataset=None, eval_dataset=None):
        '''
        :param
            @hyper_param['learning_rate_upper']
            @hyper_param['learning_rate_bert']
            @hyper_param['bert_finetune']
            @hyper_param['EPOCH']
            @hyper_param['batch_size']
            @hyper_param['visualize_length']
            @hyper_param['result_dir']
            @hyper_param['isshuffle']
            @hyper_param['model_name']
        '''
        LEARNING_RATE_bert = hyper_param.get('learning_rate_bert', 5e-5)
        LEARNING_RATE_upper = hyper_param.get('learning_rate_upper', 1e-3)

        bert_finetune = hyper_param.get('bert_finetune', True)
        EPOCH = hyper_param.get('EPOCH', 3)
        BATCH_SIZE = hyper_param.get('batch_size', 4)
        use_cuda = self.use_cuda
        visualize_length = hyper_param.get('visualize_length', 2)
        result_dir = hyper_param.get('result_dir', './result/')
        is_shuffle = hyper_param.get('isshuffle', True)
        model_name = hyper_param.get('model_name', 'model.p')
        DATA_TYPE = 'ent'

        if use_cuda:
            self.model.cuda()

        train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset
        # train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
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

        cls_param = self.model.classifier.parameters()
        bert_param = self.model.bert.parameters()

        if bert_finetune:
            optimizer_group_paramters = [
                {'params': cls_param, 'lr': LEARNING_RATE_upper}, 
                {'params': bert_param, 'lr': LEARNING_RATE_bert}
            ]
            optimizer = torch.optim.Adam(optimizer_group_paramters)
            log(f'****BERT_finetune, learning_rate_upper: {LEARNING_RATE_upper}, learning_rate_bert: {LEARNING_RATE_bert}', 0)
        else:
            optimizer = torch.optim.Adam(cls_param, lr=LEARNING_RATE_upper)
            log(f'****BERT_fix, learning_rate_upper: {LEARNING_RATE_upper}', 0)

        ##TODO:
        scheduler = LambdaLR(optimizer, lr_lambda=my_lr_lambda)
        # scheduler = transformers.optimization.get_cosine_schedule_with warmup(optimizer, num_warmup_steps=int(EPOCH*0.2), num_training_steps=EPOCH)

        all_cnt = len(train_data_mat_dict['cha_matrix'])
        log(f'{model_name} Training start!', 0)
        loss_record = []
        score_record = []
        max_score = 0

        eval_hyper_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
        for epoch in range(EPOCH):
            log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
            self.model.train()
                
            loss = 0.0
            for cnt, data_batch in enumerate(data_generator):
                x, pos, _, _, y_ent, lens, data_list = data_batch  

                loss_avg = self._loss(x, y_ent, lens)
                optimizer.zero_grad()
                loss_avg.backward()
                optimizer.step()

                loss += loss_avg
                if use_cuda:
                    loss_record.append(loss_avg.cpu().item())
                else:
                    loss_record.append(loss_avg.item())

                if (cnt+1) % visualize_length == 0:
                    loss_cur = loss / visualize_length
                    log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
                    loss = 0.0

                    # self.model.eval()
                    # print(data_list[0]['input'])
                    # pre_paths = self._output(x, lens)
                    # print('predict-path')
                    # print(pre_paths[0])
                    # print('target-path')
                    # print(y_ent[0])
                    # self.model.train()

                # if cnt+1 % 100 == 0:
                #     self.save_model(result_dir)
                #     print('Checkpoint saved successfully')
                # break

            temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_hyper_param)
            score_record.append(temp_score)
            scheduler.step()

            if temp_score[2] > max_score:
                max_score = temp_score[2]
                save_path = os.path.join(result_dir, model_name)
                self.save_model(save_path)
                print(f'Checkpoint saved successfully, current best socre is {max_score}')
            # break
        log(f'the best score of the model is {max_score}')
        return loss_record, score_record

    @torch.no_grad()
    def predict(self, data_loader: KGDataLoader, data_set=None, hyper_param={}):
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
        use_cuda = self.use_cuda
        if use_cuda:
            self.model.cuda()

        test_dataset = data_loader.dataset.test_dataset if data_set is None else data_set
        # test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
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
        
        self.model.eval()   #disable dropout layer and the bn layer

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
    def eval_model(self, data_loader: KGDataLoader, data_set=None, hyper_param={}):
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

        use_cuda = self.use_cuda
        if use_cuda:
            self.model.cuda()
        eva_data_set = data_loader.dataset.dev_dataset if data_set is None else data_set
        pred_result = self.predict(data_loader, eva_data_set, hyper_param) ###list(dict), 预测结果
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
    def _generate_mask(lens, max_len=None):
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


if __name__ == '__main__':
    model_param = {
        'num_labels':45,
        'use_cuda':True   #True
    }
    mymodel = BERT_NER(params=model_param)
    mymodel.load_model('./result/')

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
