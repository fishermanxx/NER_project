# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result, show_metadata

import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

import numpy as np
import json
from utils import my_lr_lambda
from torch.optim.lr_scheduler import LambdaLR

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from model import MODEL_TEMP
from model_crf import CRF
torch.manual_seed(1)


class REL_BLSTM_CRF(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
        '''
        :param - dict
            param['embedding_dim']
            param['hidden_dim']
            ***param['n_ent_tags']
            param['n_rel_tags']
            param['n_rels']
            param['n_words']
            param['start_idx']  int, <start> tag index for entity tag seq
            param['end_idx']   int, <end> tag index for entity tag seq
            param['use_cuda']
            param['dropout_prob']
            param['lstm_layer_num']
        '''
        super(REL_BLSTM_CRF, self).__init__()
        self.config = config
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 64)
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'

        self.n_tags = self.config.get('n_rel_tags', 8)
        self.n_rels = self.config.get('n_rels', 9)
        self.n_words = self.config.get('n_words', 10000)

        self.dropout_prob = self.config.get('dropout_prob', 0)
        self.lstm_layer_num = self.config.get('lstm_layer_num', 1)

        self.use_cuda = self.config.get('use_cuda', False)
        self.model_type = 'REL_BLSTM_CRF'

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'embedding_dim: {self.embedding_dim}', 1)
        log(f'hidden_dim: {self.hidden_dim}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log(f'lstm_layer_num: {self.lstm_layer_num}', 1)
        log(f'dropout_prob: {self.dropout_prob}', 1)  
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        self.word_embeds = nn.Embedding(self.n_words, self.embedding_dim)
        self.rel_embeds = nn.Embedding(self.n_rels, self.embedding_dim)
        self.embed2hidden = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim, 
            hidden_size = self.hidden_dim//2, 
            batch_first=True, 
            num_layers=self.lstm_layer_num, 
            dropout=self.dropout_prob, 
            bidirectional=True
        )
        self.hidden2tag = nn.Linear(self.hidden_dim, self.n_tags)
        
        crf_config = {
            'n_tags': self.n_tags, 
            'start_idx': self.config['start_rel_idx'],
            'end_idx': self.config['end_rel_idx'],
            'use_cuda': self.use_cuda    
        }
        self.crf = CRF(crf_config)
        self.relu_layer = nn.ReLU()

    def reset_parameters(self):        
        I.xavier_normal_(self.word_embeds.weight.data)
        I.xavier_normal_(self.rel_embeds.weight.data)
        self.lstm.reset_parameters()
        # stdv = 1.0 / math.sqrt(self.hidden_dim)
        # for weight in self.lstm.parameters():
        #     I.uniform_(weight, -stdv, stdv)
        I.xavier_normal_(self.embed2hidden.weight.data)
        I.xavier_normal_(self.hidden2tag.weight.data)
        self.crf.reset_parameters()
        
    def _get_lstm_features(self, x, relation_type, use_cuda=None):
        '''
        :param  
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @relation_type: index之后的relation_type， (batch_size, 1), np.array
        :return 
            @lstm_feature: (batch_size, T, n_tags) -- 类似于eject score, torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size, T = x.shape[0], x.shape[1]

        ##embedding layer
        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        # print('words_tensor_shape', words_tensor.shape)
        word_input_embeds = self.word_embeds(words_tensor)  #(batch_size, T, n_embed)
        # print('word_input_embeds_shape', word_input_embeds.shape)
        reltype_tensor = self._to_tensor(relation_type, use_cuda)  #(batch_size, 1)
        # print('reltype_tensor', reltype_tensor.shape)
        reltype_input_embeds = self.rel_embeds(reltype_tensor)  #(batch_size, 1, n_embed)
        # print('reltype_input_embeds', reltype_input_embeds.shape)
        reltype_input_embeds = reltype_input_embeds.repeat(1, T, 1)  #(batch_size, T, n_embed)
        # print('reltype_input_embeds2', reltype_input_embeds.shape)
 
        input_embeds_all = torch.cat([word_input_embeds, reltype_input_embeds], -1)  #(batch_size, T, n_embed*2)
        # print('input_embeds_all.shape', input_embeds_all.shape)  
        embeds = self.embed2hidden(input_embeds_all)  #(batch_size, T, n_embeds)
        # print('embeds.shape', embeds.shape)

        # ##LSTM layer
        if use_cuda:
            h_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2).cuda()  #(n_layer*n_dir, N, n_hid)
            c_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2).cuda()
        else:
            h_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2)
            c_0 = torch.randn(2*self.lstm_layer_num, batch_size, self.hidden_dim//2)
        # c_0 = h_0.clone()
        hidden = (h_0, c_0)
        lstm_out, _hidden = self.lstm(embeds, hidden)   #(batch_size, T, n_dir*n_hid), (h, c)

        ##FC layer
        lstm_feature = self.hidden2tag(lstm_out)  #(batch_size, T, n_tags)
        lstm_feature = torch.tanh(lstm_feature)
        # print(lstm_feature.shape)

        return lstm_feature

    def _loss(self, x, relation_type, y_rel, lens, use_cuda=None):
        '''
        loss function: neg_log_likelihood
        :param
            @x: (batch_size, T), np.array, index之后的word, 每个字符按照字典对应到index, 
            @relation_type: (batch_size, 1), np.array, 关系类别
            @y_rel: (batch_size, T), np.array, index之后的关系序列, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (batch_size), torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda

        logits = self._get_lstm_features(x, relation_type, use_cuda)
        log_norm_score = self.crf.log_norm_score(logits, lens)
        path_score = self.crf.path_score(logits, y_rel, lens)

        loss = log_norm_score - path_score
        loss = (loss/self._to_tensor(lens, use_cuda).float()).mean()
        return loss

    def _output(self, x, relation_type, lens, use_cuda=None):
        '''
        return the crf decode paths
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @relation_type: (batch_size, 1), np.array, 关系类别
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @paths: (batch_size, T), torch.tensor, 最佳句子路径
            @scores: (batch_size), torch.tensor, 最佳句子路径上的得分
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        logits = self._get_lstm_features(x, relation_type, use_cuda)
        scores, paths = self.crf.viterbi_decode(logits, lens, use_cuda)
        return paths

    def train_model(self, data_loader: KGDataLoader, train_dataset=None, eval_dataset=None, hyper_param={}, use_cuda=None, rebuild=False):
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
        LEARNING_RATE_upper = hyper_param.get('learning_rate_upper', 1e-2)
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
                # train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
                train_data_mat_dict = data_loader.transform(train_dataset, istest=False, data_type=DATA_TYPE, ratio=0)
                data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
        data_generator = Batch_Generator(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)

        print('train_data_set_length:', len(train_dataset))
        print('train_data_mat_dict_length:', train_data_mat_dict['cha_matrix'].shape)

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
        
        # ##TODO:
        scheduler = LambdaLR(optimizer, lr_lambda=my_lr_lambda)
        # # scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(EPOCH*0.2), num_training_steps=EPOCH)
        

        all_cnt = len(train_data_mat_dict['cha_matrix'])
        log(f'{model_name} Training start!', 0)
        loss_record = []
        score_record = []
        max_score = -1

        evel_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
        for epoch in range(EPOCH):
            self.train()

            log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
            loss = 0.0
            for cnt, data_batch in enumerate(data_generator):
                x, pos, reltype, y_rel, y_ent, lens, data_list = data_batch
                
                loss_avg = self._loss(x, reltype, y_rel, lens)
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

                    # self.eval()
                    # print(data_list[0]['input'])
                    # pre_paths = self._output(x, reltype, lens)
                    # print('predict-path')
                    # print(pre_paths[0])
                    # print('target-path')
                    # print(y_rel[0])
                    # self.train()        

            temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=evel_param, use_cuda=use_cuda)
            score_record.append(temp_score)
            scheduler.step()
            
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
        预测出 test_data_mat_dict['y_ent_matrix']中的内容，重新填写进该matrix, 未预测之前都是0
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

        BATCH_SIZE = hyper_param.get('batch_size', 100)
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
                test_data_mat_dict = data_loader.transform(test_dataset, istest=True, data_type=DATA_TYPE, ratio=0)
                data_loader.save_preprocessed_data(old_test_dict_path, test_data_mat_dict)
        ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*

        print('test_dataset_length:', len(test_dataset))
        print('test_data_mat_dict_length:', test_data_mat_dict['cha_matrix'].shape)
        data_generator = Batch_Generator(test_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=False)
        
        self.eval()   #disable dropout layer and the bn layer

        total_output_rel = []
        all_cnt = len(test_data_mat_dict['cha_matrix'])
        log(f'Predict start!', 0)
        for cnt, data_batch in enumerate(data_generator):
            x, pos, reltype, y_rel, y_ent, lens, data_list = data_batch
            pre_paths = self._output(x, reltype, lens)  ##pre_paths, (batch_size, T), torch.tensor
            if use_cuda:
                pre_paths = pre_paths.data.cpu().numpy().astype(np.int)
            else:
                pre_paths = pre_paths.data.numpy().astype(np.int)
            total_output_rel.append(pre_paths)
            
            if (cnt+1) % 10 == 0:
                log(f'[PREDICT] step {(cnt+1)*BATCH_SIZE}/{all_cnt}', 1)

        ## add mask when the ent seq idx larger than sentance length
        pred_output = np.vstack(total_output_rel)   ###(N, max_length), numpy.array
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
        print('final predict length:', len(result))
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

if __name__ == '__main__':
    model_config = {
        'embedding_dim' : 128,
        'hidden_dim' : 64,
        'n_tags' : 9,
        'n_rels' : 7+1,
        'n_words' : 22118,
        'start_idx': 7,  ## <start> tag index for entity tag seq
        'end_idx': 8,  ## <end> tag index for entity tag seq
        'use_cuda':False,
        'dropout_prob': 0,
        'lstm_layer_num': 1,
        'num_labels': 45
    }
    
    mymodel = REL_BLSTM_CRF(config=model_config, show_param=True) 

    ###===========================================================
    ###模型参数测试
    ###===========================================================
    ##pass


    ###===========================================================
    ###试训练 -- train_part
    ###===========================================================
    data_set = AutoKGDataset('./d1/')
    train_dataset = data_set.train_dataset[:20]
    eval_dataset = data_set.dev_dataset[:10]
    # train_dataset = data_set.train_dataset
    # eval_dataset = data_set.dev_dataset
    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')
    # print(data_loader.embedding_info_dicts['entity_type_dict'])
    
    print(data_loader.embedding_info_dicts['label_location_dict'])
    show_metadata(data_loader.metadata_)

    print('start_tags:', data_loader.rel_seq_map_dict[data_loader.START_TAG])
    print('end_tags:', data_loader.rel_seq_map_dict[data_loader.END_TAG])

    train_param = {
        'EPOCH': 1,         #45
        'batch_size': 4,    #512
        'learning_rate_bert': 5e-5,
        'learning_rate_upper': 5e-3,
        'bert_finetune': False,
        'visualize_length': 2, #10
        'isshuffle': True,
        'result_dir': './result/',
        'model_name':'model_test.p'
    }
    mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)

    # eval_param = {
    #     'batch_size':100, 
    #     'issave':True, 
    #     'result_dir': './result/'
    # }
    # # res = mymodel.predict(data_loader, data_set=eval_dataset, hyper_param=eval_param)
    # # print('predict data length', len(res))
    # mymodel.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_param)

    ###===========================================================
    ###试训练 -- detail_part
    ###=========================================================== 

    # train_data_mat_dict = data_loader.transform_rel(train_dataset, istest=False, ratio=0)
    # data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='rel', isshuffle=True)

    # for epoch in range(2):
    #     print('EPOCH: %d' % epoch)
    #     for data_batch in data_generator:
    #         x, pos, reltype, y_rel, y_ent, lens, data_list = data_batch
    #         print(x.shape, pos.shape, y_rel.shape)    ##(batch_size, max_length)
    #         # print(reltype.shape, reltype[:, 0])
    #         sentence = data_list[0]['input']
    #         # print([(i, sentence[i]) for i in range(len(sentence))])

    #         ###======================for REL_BLSTM_CRF MODEL only==================================
    #         mymodel._get_lstm_features(x, reltype)
    #         # loss = mymodel._loss(x, y_ent, lens)
    #         # print(loss.shape)
    #         # mymodel._output(x, lens)

    #         # print(x[0])
    #         # word_dict = data_loader.character_location_dict
    #         # rev_word_dict = data_loader.inverse_character_location_dict
    #         # print(list(word_dict.items())[1300:1350])
    #         # print(list(rev_word_dict.items())[1300:1350])
    #         # print(sentence)
    #         # print(list(rev_word_dict[i] for i in x[0]))
    #         break
    #     break
