import torch
from torch import nn

import os
import json
import numpy as np

from utils import log, show_result
from utils import KGDataLoader, Batch_Generator

from utils import my_lr_lambda
from torch.optim.lr_scheduler import LambdaLR

import time

def show_time(cost, name):
    print(f'{name} -- {cost:.4f}')
    # pass

class MODEL_TEMP(nn.Module):
    def __init__(self, config={}, show_param=False):
        '''
        :param - config
        '''
        super(MODEL_TEMP, self).__init__()
        pass

    def show_model_param(self):
        print('this is a model template')

    def build_model(self):
        '''
        build the layers that the model need
        '''
        pass
        
    def reset_parameters(self):        
        '''
        reset your networks' parameters
        '''
        pass
 
    def _loss(self):
        '''
        loss function: return the batch_loss
        :return 
            @loss: (1), torch.tensor
        '''
        pass

    def _output(self):
        '''
        return the model predict result
        :return 
            @paths: (batch_size, T), torch.tensor, 最佳句子路径
        '''
        pass

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


    def train_model(self, data_loader, train_dataset=None, eval_dataset=None, hyper_param={}, use_cuda=None, rebuild=False):
        '''
        :param
            @data_loader: (KGDataLoader),
            @result_dir: (str) path to save the trained model and extracted dictionary
            @hyper_param: (dict)
                @hyper_param['EPOCH']
                @hyper_param['batch_size']
                @hyper_param['learning_rate_upper']
                @hyper_param['learning_rate_bert']
                @hyper_param['bert_finetune']
                @hyper_param['visualize_length']   #num of batches between two check points
                @hyper_param['isshuffle']
                @hyper_param['result_dir']
                @hyper_param['model_name']
        :return
            @loss_record, 
            @score_record
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        if use_cuda:
            print('use cuda=========================')
            self.cuda()

        EPOCH = hyper_param.get('EPOCH', 3)
        BATCH_SIZE = hyper_param.get('batch_size', 4)
        LEARNING_RATE_upper = hyper_param.get('learning_rate_upper', 1e-3)
        LEARNING_RATE_bert = hyper_param.get('learning_rate_bert', 5e-5)
        bert_finetune = hyper_param.get('bert_finetune', True)
        visualize_length = hyper_param.get('visualize_length', 10)
        result_dir = hyper_param.get('result_dir', './result/')
        model_name = hyper_param.get('model_name', 'model.p')
        is_shuffle = hyper_param.get('isshuffle', True)
        DATA_TYPE = 'rel'
        
        train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset
        if rebuild:
            train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        else:
            old_train_dict_path = os.path.join(result_dir, 'train_data_mat_dict.pkl')
            if os.path.exists(old_train_dict_path):
                train_data_mat_dict = data_loader.load_preprocessed_data(old_train_dict_path)
                log('Reload preprocessed data successfully~')
            else:
                train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
                data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        data_generator = Batch_Generator(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)

        all_param = list(self.named_parameters()) 
        bert_param = [p for n, p in all_param if 'bert' in n]
        other_param = [p for n, p in all_param if 'bert' not in n]

        if bert_finetune:
            optimizer_group_paramters = [
                {'params': other_param, 'lr': LEARNING_RATE_upper}, 
                {'params': bert_param, 'lr': LEARNING_RATE_bert}
            ]
            optimizer = torch.optim.Adam(optimizer_group_paramters)
            log(f'****BERT_finetune, learning_rate_upper: {LEARNING_RATE_upper}, learning_rate_bert: {LEARNING_RATE_bert}', 0)
        else:
            optimizer = torch.optim.Adam(other_param, lr=LEARNING_RATE_upper)
            log(f'****BERT_fix, learning_rate_upper: {LEARNING_RATE_upper}', 0)
        
        ##TODO:
        scheduler = LambdaLR(optimizer, lr_lambda=my_lr_lambda)
        # scheduler = transformers.optimization.get_cosine_schedule_with warmup(optimizer, num_warmup_steps=int(EPOCH*0.2), num_training_steps=EPOCH)
        

        all_cnt = len(train_data_mat_dict['cha_matrix'])
        log(f'{model_name} Training start!', 0)
        loss_record = []
        score_record = []
        max_score = 0

        evel_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
        for epoch in range(EPOCH):
            self.train()

            log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
            loss = 0.0
            for cnt, data_batch in enumerate(data_generator):
                ep_start_time = time.time()

                x, pos, relation, y_rel, y_ent, lens, data_list = data_batch
                
                loss_avg = self._loss(x, y_rel, lens)

                # loss_time = time.time()
                # show_time(loss_time-ep_start_time, 'cal loss time')

                optimizer.zero_grad()
                loss_avg.backward()

                # back_time = time.time()
                # show_time(back_time-loss_time, 'back prop time')

                optimizer.step()

                # show_time(time.time()-back_time, 'opt step time')
                # show_time(time.time()-ep_start_time, 'total cost time')
                # print('-'*80)


                loss += loss_avg
                if use_cuda:
                    loss_record.append(loss_avg.cpu().item())
                else:
                    loss_record.append(loss_avg.item())

                if (cnt+1) % visualize_length == 0:
                    loss_cur = loss / visualize_length
                    log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
                    loss = 0.0

                    show_time(time.time()-ep_start_time, 'total time cost per epoch')
                    print('='*80)

                    # self.eval()
                    # print(data_list[0]['input'])
                    # pre_paths, pre_scores = self._output(x, lens)
                    # print('predict-path')
                    # print(pre_paths[0])
                    # print('target-path')
                    # print(y_rel[0])
                    # self.train()        

            temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=evel_param, use_cuda=use_cuda)
            score_record.append(temp_score)
            scheduler.step()   #TODO:
            
            if temp_score[2] > max_score:
                max_score = temp_score[2]
                save_path = os.path.join(result_dir, model_name)
                self.save_model(save_path)
                print(f'Checkpoint saved successfully, current best socre is {max_score}')
        log(f'the best score of the model is {max_score}')
        return loss_record, score_record

    @torch.no_grad()
    def predict(self, data_loader, data_set=None, hyper_param={}, use_cuda=None, rebuild=False):
        '''
        预测出 test_data_mat_dict['y_rel_matrix']中的内容，重新填写进该matrix, 未预测之前都是0
        :param
            @data_loader: (KGDataLoader),
            @hyper_param: (dict)
                @hyper_param['batch_size']  ##默认4
                @hyper_param['issave']  ##默认False
                @hyper_param['result_dir']  ##默认None
        :return
            @result: list, len(句子个数)
                case = result[0]
                case['input']
                case['relation_list']
                    r = case['relation_list'][0]
                    r['relation']: 成立日期
                    r['head']: '百度'
                    r['tail']: '2016年04月08日'
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        if use_cuda:
            print('use cuda=========================')
            self.cuda()

        BATCH_SIZE = hyper_param.get('batch_size', 64)
        ISSAVE = hyper_param.get('issave', False)
        result_dir = hyper_param.get('result_dir', './result/')
        DATA_TYPE = 'rel'

        
        test_dataset = data_loader.dataset.test_dataset if data_set is None else data_set
        if rebuild:
            test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        else:
            old_test_dict_path = os.path.join(result_dir, 'test_data_mat_dict.pkl')
            if os.path.exists(old_test_dict_path):
                test_data_mat_dict = data_loader.load_preprocessed_data(old_test_dict_path)
                log('Reload preprocessed data successfully~')
            else:
                test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE)
                data_loader.save_preprocessed_data(old_test_dict_path, test_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        data_generator = Batch_Generator(test_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=False)
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
        test_data_mat_dict['y_rel_matrix'] = pred_output

        result = data_loader.transform_back(test_data_mat_dict, data_type=DATA_TYPE)

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
    def eval_model(self, data_loader, data_set=None, hyper_param={}, use_cuda=None, rebuild=False):
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
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        if use_cuda:
            print('use cuda=========================')
            self.cuda()

        def dict2str(d):
            ## 将entity 从字典形式转化为str形式方便比较
            # res = d['entity']+':'+d['entity_type']+':'+str(d['entity_index']['begin'])+'-'+str(d['entity_index']['end'])
            ## 将relation 从字典形式转化为str形式方便比较
            res = d['relation']+'-'+d['head']+'-'+d['tail']
            return res

        def calculate_f1(pred_cnt, tar_cnt, correct_cnt):
            precision_s = round(correct_cnt / (pred_cnt + 1e-8), 3)
            recall_s = round(correct_cnt / (tar_cnt + 1e-8), 3)
            f1_s = round(2*precision_s*recall_s / (precision_s + recall_s + 1e-8), 3)
            return precision_s, recall_s, f1_s


        eva_data_set = data_loader.dataset.dev_dataset if data_set is None else data_set

        pred_result = self.predict(data_loader, eva_data_set, hyper_param, use_cuda, rebuild=rebuild) ###list(dict), 预测结果 len=n_sentence
        target = eva_data_set  ###list(dict)  AutoKGDataset, 真实结果

        pred_cnt = 0
        tar_cnt = 0
        correct_cnt = 0
        cnt_all = len(eva_data_set)
        log('Eval start')
        for idx in range(cnt_all):
            sentence = pred_result[idx]['input']
            pred_list = pred_result[idx]['relation_list']
            tar_list = target[idx]['output']['relation_list']
   
            str_pred_set = set(map(dict2str, pred_list))
            str_tar_set = set(map(dict2str, tar_list))
            common_set = str_pred_set.intersection(str_tar_set)
            # print('target:')
            # print(str_tar_set)
            # print('predict:')
            # print(str_pred_set)

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


if __name__ == '__main__':
    model = MODEL_TEMP()