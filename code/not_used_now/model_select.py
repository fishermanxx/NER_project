import os
import sys
import argparse
import numpy as np

# from ent_model import BLSTM_CRF
from ent_model_dropout import BLSTM_CRF
from utils import KGDataLoader
from common import get_logger, Timer
from dataset import AutoKGDataset

import matplotlib.pyplot as plt

CPU_TRAIN = 1500
CPU_EVAL = 400
CPU_EPOCH = 20
CPU_BATCHSIZE = 32
CPU_VISUAL = 10

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class Model_selector:
    def __init__(self, params: dict, data_loader: KGDataLoader):
        '''
        :params:
            @params['batch_size_list']
            @params['lr_list']
            @params['EPOCH']
            @params['result_dir']
            @params['use_cuda']

            @params['isshuffle']
            @params['visualize_length']
        '''
        self.batch_size_list = params.get('batch_size_list', [4])
        self.lr_list = params.get('lr_list', [1e-3, 5e-3, 1e-2, 5e-2])

        self.param = {}
        self.param['EPOCH'] = params.get('EPOCH', 3)
        self.param['result_dir'] = params.get('result_dir', None)
        self.param['isshuffle'] = params.get('isshuffle', True)
        self.param['visualize_length'] = params.get('visualize_length', 10)

        self.use_cuda = params.get('use_cuda', 0)
        self.data_loader = data_loader

    def build_model(self, show_param=False):
        use_cuda = self.use_cuda
        model_params = {
            'embedding_dim' : 64,
            'hidden_dim' : 128,
            'n_tags' : len(self.data_loader.ent_seq_map_dict),
            'n_words' : len(self.data_loader.character_location_dict),
            'start_idx': self.data_loader.ent_seq_map_dict[self.data_loader.START_TAG],  ## <start> tag index for entity tag seq
            'end_idx': self.data_loader.ent_seq_map_dict[self.data_loader.END_TAG],  ## <end> tag index for entity tag seq
            'use_cuda':use_cuda,
            'dropout_prob': 0,
            'lstm_layer_num': 1
        }
        mymodel = BLSTM_CRF(model_params, show_param=show_param)
        return mymodel

    def grid_search(self):
        self.param['batch_size'] = self.batch_size_list[0]
        if self.use_cuda:
            train_data_set = self.data_loader.dataset.train_dataset
            eval_data_set = self.data_loader.dataset.dev_dataset
        else:
            train_data_set = self.data_loader.dataset.train_dataset[:CPU_TRAIN]
            eval_data_set = self.data_loader.dataset.dev_dataset[:CPU_EVAL]          
        loss_all = []
        score_all = []
        for lr in self.lr_list:
            temp_model = self.build_model(show_param=True)
            self.param['learning_rate'] = lr
            self.param['model_name'] = f'model_lr_{lr}.p'
            timer = Timer()
            timer.set(3600)
            with timer.time_limit(f'training_lr_{lr}'):
                t_loss_record, t_score_record = temp_model.train_model(self.data_loader, train_data_set, eval_data_set, hyper_param=self.param, use_cuda=self.use_cuda)
                temp_model.eval_model(self.data_loader, eval_data_set, self.param, use_cuda=self.use_cuda)
                loss_all.append(t_loss_record)
                score_all.append(t_score_record)
        return loss_all, score_all

def _here(*args):
    # 返回包含此.py文件的目录的父目录
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(here, *args)

def _parse_args():
    '''
    :return
        @args.dataset_dir
        @args.code_dir
        @args.result_dir
        @args.time_budget
        @args.task
    '''
    root_dir = _here(os.pardir)  ##总目录
    default_dataset_dir = os.path.join(root_dir, 'd1')
    default_code_dir = os.path.join(root_dir, 'code')
    default_result_dir = os.path.join(root_dir, 'result')

    default_time_budget = 1200
    default_task = 'baidu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset, contain .data and .solution file")

    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the code, contain .py file")

    parser.add_argument('--result_dir', type=str,
                        default=default_result_dir,
                        help="Directory storing the processing result, including model and dict infomation")

    parser.add_argument('--time_budget', type=float,
                        default=default_time_budget,
                        help="Time budget for training model")

    parser.add_argument('--task', type=str,
                        default=default_task,
                        help="Default task name - baidu")

    parser.add_argument('--use_cuda', type=int,
                        default=0,
                        help="whether to use cuda - defualt False")
    
    args = parser.parse_args()
    LOGGER.debug(f'sys.argv = {sys.argv}')
    LOGGER.debug(f'Using dataset_dir: {args.dataset_dir}')
    LOGGER.debug(f'Using code_dir: {args.code_dir}')
    LOGGER.debug(f'Using result_dir: {args.result_dir}')
    LOGGER.debug(f'use_cuda: {args.use_cuda}')
    return args

def _init_python_path(args):
    # 检查保存文件目录是否存在，不存在就创造出来, 将所有文件模块添加到搜索路径
    sys.path.append(args.code_dir)
    os.makedirs(args.result_dir, exist_ok=True)

def plot_img(arr, var_list, filename='img.png'):
    '''
    :param
        @arr: (N, T), N-模型数目
        @var_list: 模型变化量
    '''
    plt.figure()
    n = arr.shape[0]
    for i in range(n):
        plt.plot(arr[i], label=var_list[i])
    plt.legend()
    # loss_img_path = os.path.join(args.result_dir, 'loss_all.png')
    plt.savefig(filename)

def main():
    LOGGER.info("===== Start Select param")
    LOGGER.info('===== Initialize args')
    args = _parse_args()
    _init_python_path(args)

    ##获取数据集
    dataset = AutoKGDataset(args.dataset_dir)  

    LOGGER.info('===== Load metadata')
    metadata = dataset.get_metadata()
    args.time_budget = metadata.get('time_budget', args.time_budget)
    LOGGER.info(f'Time budget: {args.time_budget}')

    data_loader = KGDataLoader(dataset, rebuild=False, temp_dir=args.result_dir)

    if args.use_cuda:
        params = {
            'batch_size_list': [256],
            'lr_list' : [3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2], 
            'EPOCH' : 50,
            'result_dir' : args.result_dir,
            'use_cuda' : args.use_cuda,
            'visualize_length' : 10
        }
    else:
        params = {
            'batch_size_list': [CPU_BATCHSIZE],
            'lr_list' : [1e-3, 5e-3, 1e-2], 
            'EPOCH' : CPU_EPOCH,
            'result_dir' : args.result_dir,
            'use_cuda' : args.use_cuda,
            'visualize_length' : CPU_VISUAL
        }

    selector = Model_selector(params, data_loader)
    loss_all, score_all = selector.grid_search()

    loss_all = np.array(loss_all)
    score_all = np.array(score_all)
    print(score_all.shape)

    ## Save loss and scores 
    loss_save_path = os.path.join(args.result_dir, 'loss_all.txt')
    np.savetxt(loss_save_path, loss_all)

    precision_save_path = os.path.join(args.result_dir, 'precision_all.txt')
    np.savetxt(precision_save_path, score_all[:, :, 0])
    recall_save_path = os.path.join(args.result_dir, 'precision_all.txt')
    np.savetxt(recall_save_path, score_all[:, :, 1])
    f1_save_path = os.path.join(args.result_dir, 'f1_all.txt')
    np.savetxt(f1_save_path, score_all[:, :, 2])

    ## Plot loss and scores
    loss_img_path = os.path.join(args.result_dir, 'loss_all.png')
    plot_img(loss_all, params['lr_list'], loss_img_path)

    precision_img_path = os.path.join(args.result_dir, 'precision_all.png')
    plot_img(score_all[:, :, 0], params['lr_list'], precision_img_path)    
    recall_img_path = os.path.join(args.result_dir, 'recall_all.png')
    plot_img(score_all[:, :, 1], params['lr_list'], recall_img_path)
    f1_img_path = os.path.join(args.result_dir, 'f1_all.png')
    plot_img(score_all[:, :, 2], params['lr_list'], f1_img_path)


if __name__ == '__main__':
    main()