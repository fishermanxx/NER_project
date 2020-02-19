# !-*- coding:utf-8 -*-

from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import log, show_result

import torch
from torch import nn
import torch.nn.init as I
from torch.autograd import Variable

# from utils import my_lr_lambda
# from torch.optim.lr_scheduler import LambdaLR

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from model import MODEL_TEMP
# from model_crf import CRF
from torchcrf import CRF
torch.manual_seed(1)


class BASELINE(MODEL_TEMP):
    def __init__(self, config={}, show_param=False):
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
        super(BASELINE, self).__init__()
        self.config = config
        # self.embedding_dim = self.config.get('embedding_dim', 64)
        self.embedding_dim = 64  #TODO:
        # self.hidden_dim = self.config.get('hidden_dim', 128*2)
        self.hidden_dim = 128*2  #TODO:
        assert self.hidden_dim % 2 == 0, 'hidden_dim for BLSTM must be even'
        self.n_tags = self.config.get('n_ent_tags', 45) - 2
        self.n_words = self.config.get('n_words', 10000)

        self.dropout_prob = self.config.get('dropout_prob', 0)
        # self.lstm_layer_num = self.config.get('lstm_layer_num', 4)
        self.lstm_layer_num = 4  #TODO:

        self.use_cuda = self.config.get('use_cuda', False)
        self.model_type = 'BASELINE'

        self.build_model()
        self.reset_parameters()
        if show_param:
            self.show_model_param()

    def show_model_param(self):
        log('='*80, 0)
        log(f'model_type: {self.model_type}', 1)
        log(f'use_cuda: {self.use_cuda}', 1)
        log(f'embedding_dim: {self.embedding_dim}', 1)
        log(f'hidden_dim: {self.hidden_dim}', 1)
        log(f'lstm_layer_num: {self.lstm_layer_num}', 1)
        log(f'dropout_prob: {self.dropout_prob}', 1)  
        log(f'n_ent_tags: {self.n_tags}', 1)
        log(f"crf_start_idx: {self.config['start_ent_idx']}", 1)
        log(f"crf_end_idx: {self.config['end_ent_idx']}", 1)
        log('='*80, 0)      

    def build_model(self):
        '''
        build the embedding layer, lstm layer and CRF layer
        '''
        self.word_embeds = nn.Embedding(self.n_words, self.embedding_dim)
        # self.pos_embeds = nn.Embedding(self.)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, batch_first=True, num_layers=self.lstm_layer_num, dropout=self.dropout_prob, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.n_tags)
        self.crf = CRF(self.n_tags, batch_first=True)


    def reset_parameters(self):        
        I.xavier_normal_(self.word_embeds.weight.data)
        self.lstm.reset_parameters()
        # stdv = 1.0 / math.sqrt(self.hidden_dim)
        # for weight in self.lstm.parameters():
        #     I.uniform_(weight, -stdv, stdv)
        I.xavier_normal_(self.hidden2tag.weight.data)
        self.crf.reset_parameters()
        
    def _get_lstm_features(self, x, use_cuda=None):
        '''
        :param  
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
        :return 
            @lstm_feature: (batch_size, T, n_tags) -- 类似于eject score, torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        batch_size = x.shape[0]

        ##embedding layer
        words_tensor = self._to_tensor(x, use_cuda)  #(batch_size, T)
        embeds = self.word_embeds(words_tensor)  #(batch_size, T, n_embed)

        ##LSTM layer
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
        lstm_feature = self.hidden2tag(lstm_out) #(batch_size, T, n_tags)
        lstm_feature = torch.tanh(lstm_feature)

        return lstm_feature

    def _loss(self, x, y_ent, lens, use_cuda=None):
        '''
        loss function: neg_log_likelihood
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @y_ent: (batch_size, T), np.array, index之后的entity seq, 字符级别,
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @loss: (1), torch.tensor
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]

        logits = self._get_lstm_features(x)   ##(batch_size, T, n_tags)
        tensor_y_ent = self._to_tensor(y_ent, use_cuda)

        lens = self._to_tensor(lens, use_cuda)
        len_mask = self._generate_mask(lens, max_len=T)  ##(batch_size, T)

        log_likelihood_ent = self.crf(emissions=logits, tags=tensor_y_ent, mask=len_mask, reduction='mean')

        # log_norm_score = self.crf.log_norm_score(logits, lens)
        # path_score = self.crf.path_score(logits, y_ent, lens)

        # loss = log_norm_score - path_score
        # loss = (loss/self._to_tensor(lens, use_cuda)).mean()
        return - log_likelihood_ent

    def _output(self, x, lens, use_cuda=None):
        '''
        return the crf decode paths
        :param
            @x: index之后的word, 每个字符按照字典对应到index, (batch_size, T), np.array
            @lens: (batch_size), list, 具体每个句子的长度, 
        :return 
            @paths: (batch_size, T), torch.tensor, 最佳句子路径
            @scores: (batch_size), torch.tensor, 最佳句子路径上的得分
        '''
        use_cuda = self.use_cuda if use_cuda is None else use_cuda
        T = x.shape[1]
        logits = self._get_lstm_features(x, use_cuda)

        lens = self._to_tensor(lens, use_cuda)
        len_mask = self._generate_mask(lens, max_len=T)  ##(batch_size, T)
    
        # paths = self.crf.decode(logits, len_mask)
        paths = self.crf.decode(logits)
        paths = self._to_tensor(paths, use_cuda)
        return paths

    # def train_model(self, data_loader: KGDataLoader, train_dataset=None, eval_dataset=None, hyper_param={}, use_cuda=None):
    #     '''
    #     :param
    #         @data_loader: (KGDataLoader),
    #         @result_dir: (str) path to save the trained model and extracted dictionary
    #         @hyper_param: (dict)
    #             @hyper_param['EPOCH']
    #             @hyper_param['batch_size']
    #             @hyper_param['learning_rate']
    #             @hyper_param['visualize_length']   #num of batches between two check points
    #             @hyper_param['isshuffle']
    #             @hyper_param['result_dir']
    #             @hyper_param['model_name']
    #     :return
    #         @loss_record, 
    #         @score_record
    #     '''
    #     use_cuda = self.use_cuda if use_cuda is None else use_cuda
    #     if use_cuda:
    #         print('use cuda=========================')
    #         self.cuda()

    #     EPOCH = hyper_param.get('EPOCH', 3)
    #     BATCH_SIZE = hyper_param.get('batch_size', 4)
    #     LEARNING_RATE = hyper_param.get('learning_rate', 1e-2)
    #     visualize_length = hyper_param.get('visualize_length', 10)
    #     result_dir = hyper_param.get('result_dir', './result/')
    #     model_name = hyper_param.get('model_name', 'model.p')
    #     is_shuffle = hyper_param.get('isshuffle', True)
    #     DATA_TYPE = 'ent'
        

    #     train_dataset = data_loader.dataset.train_dataset if train_dataset is None else train_dataset
    #     ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
    #     old_train_dict_path = os.path.join(result_dir, 'train_data_mat_dict.pkl')
    #     if os.path.exists(old_train_dict_path):
    #         train_data_mat_dict = data_loader.load_preprocessed_data(old_train_dict_path)
    #         log('Reload preprocessed data successfully~')
    #     else:
    #         train_data_mat_dict = data_loader.transform(train_dataset, data_type=DATA_TYPE)
    #         data_loader.save_preprocessed_data(old_train_dict_path, train_data_mat_dict)
    #     ## 保存预处理的文本，这样调参的时候可以直接读取，节约时间   *WARNING*
    #     data_generator = Batch_Generator(train_data_mat_dict, batch_size=BATCH_SIZE, data_type=DATA_TYPE, isshuffle=is_shuffle)
        
    #     optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    #     all_cnt = len(train_data_mat_dict['cha_matrix'])
    #     log(f'{model_name} Training start!', 0)
    #     loss_record = []
    #     score_record = []
    #     max_score = 0

    #     evel_param = {'batch_size':100, 'issave':False, 'result_dir': result_dir}
    #     for epoch in range(EPOCH):
    #         self.train()

    #         log(f'EPOCH: {epoch+1}/{EPOCH}', 0)
    #         loss = 0.0
    #         for cnt, data_batch in enumerate(data_generator):
    #             x, pos, _, _, y_ent, lens, data_list = data_batch
                
    #             loss_avg = self._loss(x, y_ent, lens)
    #             optimizer.zero_grad()
    #             loss_avg.backward()
    #             optimizer.step()

    #             loss += loss_avg
    #             if use_cuda:
    #                 loss_record.append(loss_avg.cpu().item())
    #             else:
    #                 loss_record.append(loss_avg.item())

    #             if (cnt+1) % visualize_length == 0:
    #                 loss_cur = loss / visualize_length
    #                 log(f'[TRAIN] step: {(cnt+1)*BATCH_SIZE}/{all_cnt} | loss: {loss_cur:.4f}', 1)
    #                 loss = 0.0

    #                 # self.eval()
    #                 # print(data_list[0]['input'])
    #                 # pre_paths, pre_scores = self._output(x, lens)
    #                 # print('predict-path')
    #                 # print(pre_paths[0])
    #                 # print('target-path')
    #                 # print(y_ent[0])
    #                 # self.train()        

    #         temp_score = self.eval_model(data_loader, data_set=eval_dataset, hyper_param=evel_param, use_cuda=use_cuda)
    #         score_record.append(temp_score)
    #         # scheduler.step()
            
    #         if temp_score[2] > max_score:
    #             max_score = temp_score[2]
    #             save_path = os.path.join(result_dir, model_name)
    #             self.save_model(save_path)
    #             print(f'Checkpoint saved successfully, current best socre is {max_score}')
    #     log(f'the best score of the model is {max_score}')
    #     return loss_record, score_record


if __name__ == '__main__':
    model_config = {
        'embedding_dim' : 128,
        'hidden_dim' : 64,
        'n_tags' : 45,
        'n_words' : 22118,
        'start_idx': 43,  ## <start> tag index for entity tag seq
        'end_idx': 44,  ## <end> tag index for entity tag seq
        'use_cuda':True,
        'dropout_prob': 0,
        'lstm_layer_num': 1,
        'num_labels': 45
    }
    
    mymodel = BLSTM_CRF(config=model_config, show_param=True) 

    ###===========================================================
    ###模型参数测试
    ###===========================================================

    ##case1:
    all_param = list(mymodel.named_parameters()) 
    bert_param = [(n, p) for n, p in all_param if 'bert' in n]
    other_param = [(n, p) for n, p in all_param if 'bert' not in n]
    print(f'all_param: {len(all_param)}')
    print(f'bert_param: {len(bert_param)}')
    print(f'other_param: {len(other_param)}')
    for n, p in other_param:
        print(n, p.shape)

    ##case2:
    print('='*80)
    all_param = list(mymodel.named_parameters())
    crf_param = list(mymodel.crf.named_parameters())
    lstm_param = list(mymodel.lstm.named_parameters())
    fc_param = list(mymodel.hidden2tag.named_parameters())
    # bert_param = list(mymodel.bert.named_parameters())
    emb_param = list(mymodel.word_embeds.named_parameters())

    # print(f'emb_param: {len(emb_param)}')
    print(f'crf_param: {len(crf_param)}')
    print(f'lstm_param: {len(lstm_param)}')
    print(f'fc_param: {len(fc_param)}')
    print(f'emb_param: {len(emb_param)}')
    print('='*50)
    print(f'all_param: {len(all_param)}')
    # print(len(all_param))

    other_param = crf_param + fc_param + lstm_param + emb_param
    for n, p in other_param:
        print(n, p.shape)

    # no_decay = ['bias', 'reverse']
    # # choose_param = [p for n, p in lstm_param if not any(nd in n for nd in no_decay)]
    # # choose_param = [np for np in lstm_param if not any(nd in np[0] for nd in no_decay)]
    # # choose_param = [np for np in lstm_param if 'bias' in np[0]]
    # choose_param = [np for np in lstm_param if any(nd in np[0] for nd in no_decay)]
    # for name, param in lstm_param:
    #     print(name, param.shape, id(param))

    # print('='*50)
    # for name, param in choose_param:
    #     print(name, param.shape, id(param))


    ###===========================================================
    ###试训练
    ###===========================================================
    # data_set = AutoKGDataset('./d1/')
    # # train_dataset = data_set.train_dataset[:20]
    # # eval_dataset = data_set.dev_dataset[:10]
    # train_dataset = data_set.train_dataset
    # eval_dataset = data_set.dev_dataset
    # os.makedirs('result', exist_ok=True)
    # data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')
    # print(data_loader.embedding_info_dicts['entity_type_dict'])

    # train_param = {
    #     'EPOCH': 1,         #45
    #     'batch_size': 4,    #512
    #     'learning_rate_bert': 5e-5,
    #     'learning_rate_upper': 5e-3,
    #     'bert_finetune': False,
    #     'visualize_length': 2, #10
    #     'isshuffle': True,
    #     'result_dir': './result/',
    #     'model_name':'model_test.p'
    # }
    # mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)

    # eval_param = {
    #     'batch_size':100, 
    #     'issave':False, 
    #     'result_dir': './result/'
    # }
    # mymodel.eval_model(data_loader, data_set=eval_dataset, hyper_param=eval_param)
