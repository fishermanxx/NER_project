from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from utils import show_metadata, show_dict_info
# from ent_model import BLSTM_CRF
from ent_model_dropout import BLSTM_CRF
from model_bert_lstm_crf import BERT_LSTM_CRF
from model_bert_mlp import BERT_MLP
from model_bert_mlp2 import BERT_NER

from common import get_logger, Timer

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

CPU_TRAIN = 1000
CPU_EVAL = 400
CPU_EPOCH = 3
CPU_BATCHSIZE = 4
CPU_VISUAL = 20

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


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
    
    parser.add_argument('--mode', type=str,
                        default='train',
                        help="Mode of the task: train, eval, predict")


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

def plot_img(arr, filename='img.png'):
    plt.figure()
    n = arr.shape[0]
    for i in range(n):
        plt.plot(arr[i])
    # loss_img_path = os.path.join(args.result_dir, 'loss_all.png')
    plt.savefig(filename)

def _train(mymodel, args, data_loader, train_dataset=None, eval_dataset=None, RELOAD_MODEL=None, use_cuda=False):
    old_model_path = os.path.join(args.result_dir, RELOAD_MODEL)
    if RELOAD_MODEL is not None and os.path.exists(old_model_path):
        mymodel.load_model(old_model_path)
        LOGGER.info("Reload model successfully~")
    else:
        LOGGER.info(f'There is no such file in {old_model_path}, Rebuild model')

    if use_cuda:
        train_param = {
            'EPOCH': 30,         #45
            'batch_size': 64,    #512
            'learning_rate_bert': 5e-5,
            'learning_rate_upper': 5e-5,
            'bert_finetune': False,
            'visualize_length': 20, #10
            'isshuffle': True,
            'result_dir': args.result_dir,
            'model_name':'model_test.p'
        }

    else:
        train_param = {
            'EPOCH': CPU_EPOCH,         #45
            'batch_size': CPU_BATCHSIZE,    #512
            'learning_rate_bert': 5e-5,
            'learning_rate_upper': 1e-3,
            'visualize_length': CPU_VISUAL, #10
            'isshuffle': True,
            'result_dir': args.result_dir,
            'model_name':'model_test.p'
        }        

    timer = Timer()
    timer.set(args.time_budget)
    loss_record = None
    with timer.time_limit('training'):
        # loss_record, score_record = mymodel.train_model(data_loader, train_dataset, eval_dataset, hyper_param, use_cuda=use_cuda)
        # loss_record, score_record = mymodel.train_model(data_loader, train_dataset, eval_dataset, hyper_param)
        loss_record, score_record = mymodel.train_model(data_loader, hyper_param=train_param, train_dataset=train_dataset, eval_dataset=eval_dataset)

    loss_record = np.array(loss_record)
    loss_save_path = os.path.join(args.result_dir, 'loss_train.txt')
    loss_img_path = os.path.join(args.result_dir, 'loss.png')
    np.savetxt(loss_save_path, loss_record)

    score_record = np.array(score_record)
    score_save_path = os.path.join(args.result_dir, 'score_train.txt')
    score_img_path = os.path.join(args.result_dir, 'score.png')
    np.savetxt(score_save_path, score_record)  ### (epochs, 3)

    loss = np.loadtxt(loss_save_path).reshape(1, -1)
    score = np.loadtxt(score_save_path).T
    plot_img(loss, loss_img_path)
    plot_img(score, score_img_path)


def _predict(mymodel, args, data_loader, data_set=None, RELOAD_MODEL=None, use_cuda=False):
    old_model_path = os.path.join(args.result_dir, RELOAD_MODEL)
    if RELOAD_MODEL is not None and os.path.exists(old_model_path):
        mymodel.load_model(old_model_path)
        LOGGER.info("Reload model successfully~")
    else:
        LOGGER.info(f'There is no such file in {old_model_path}, End predict')
        return   

    hyper_param = {
        'batch_size': 100,
        'issave': True,
        'result_dir': args.result_dir
    }

    timer = Timer()
    timer.set(args.time_budget)
    with timer.time_limit('predict'):
        mymodel.predict(data_loader, data_set=data_set, hyper_param=hyper_param)   

def _eval(mymodel: BLSTM_CRF, args, data_loader, data_set=None, RELOAD_MODEL=None, use_cuda=False):
    old_model_path = os.path.join(args.result_dir, RELOAD_MODEL)
    if RELOAD_MODEL is not None and os.path.exists(old_model_path):
        mymodel.load_model(old_model_path)
        LOGGER.info("Reload model successfully~")
    else:
        LOGGER.info(f'There is no such file in {old_model_path}, End evaluation')
        return

    LOGGER.info(f'reload model {RELOAD_MODEL}')
    hyper_param = {
        'batch_size': 100,
        'issave': True,
        'result_dir': args.result_dir
    }

    timer = Timer()
    timer.set(args.time_budget)
    with timer.time_limit('eval'):
        mymodel.eval_model(data_loader, data_set, hyper_param)

def main():
    LOGGER.info("===== Start program")
    LOGGER.info('===== Initialize args')
    args = _parse_args()
    _init_python_path(args)

    LOGGER.info(f'===== task mode: {args.mode}')

    ##获取数据集
    dataset = AutoKGDataset(args.dataset_dir)  
    # show_metadata(dataset.metadata_)

    LOGGER.info('===== Load metadata')
    LOGGER.info(f'===== use_cuda: {args.use_cuda}')
    metadata = dataset.get_metadata()
    args.time_budget = metadata.get('time_budget', args.time_budget)
    LOGGER.info(f'Time budget: {args.time_budget}')

    data_loader = KGDataLoader(dataset, rebuild=False, temp_dir=args.result_dir)
    # show_dict_info(data_loader)
    # print(data_loader.entity_type_dict)

    ## Reload model
    model_params = {
        'embedding_dim' : 768,
        'hidden_dim' : 64,
        'n_tags' : len(data_loader.ent_seq_map_dict),
        'n_words' : len(data_loader.character_location_dict),
        'start_idx': data_loader.ent_seq_map_dict[data_loader.START_TAG],  ## <start> tag index for entity tag seq
        'end_idx': data_loader.ent_seq_map_dict[data_loader.END_TAG],  ## <end> tag index for entity tag seq
        'use_cuda':args.use_cuda,
        'dropout_prob': 0,
        'lstm_layer_num': 1
        # 'num_labels': len(data_loader.ent_seq_map_dict)
    }
    # mymodel = BLSTM_CRF(model_params, show_param=True)   
    # mymodel = BERT_LSTM_CRF(model_params, show_param=True) 
    # mymodel = BERT_MLP(model_params, show_param=True)
    mymodel = BERT_NER(model_params, show_param=True)

    if args.use_cuda:
        train_dataset = dataset.train_dataset
        test_dataset = dataset.test_dataset
        eval_dataset = dataset.dev_dataset
    else:
        train_dataset = dataset.train_dataset[:CPU_TRAIN]
        test_dataset = dataset.test_dataset
        eval_dataset = dataset.dev_dataset[:CPU_EVAL]      

    if args.mode == 'train':
        LOGGER.info('===== Start Train')
        _train(mymodel, args, data_loader, train_dataset=train_dataset, eval_dataset=eval_dataset, RELOAD_MODEL='', use_cuda=args.use_cuda)
        # _train(mymodel, args, data_loader, train_dataset=train_dataset, eval_dataset=eval_dataset, RELOAD_MODEL='model_test.p', use_cuda=args.use_cuda)

        LOGGER.info('===== Start Eval')
        _eval(mymodel, args, data_loader, data_set=eval_dataset, RELOAD_MODEL='', use_cuda=args.use_cuda)
        # _eval(mymodel, args, data_loader, data_set=eval_dataset, RELOAD_MODEL='model_test.p', use_cuda=args.use_cuda)

    if args.mode == 'eval':
        LOGGER.info('===== Start Eval')
        _eval(mymodel, args, data_loader, data_set=eval_dataset, RELOAD_MODEL='model_lr_0.01.p', use_cuda=args.use_cuda)

    if args.mode == 'predict':
        LOGGER.info('===== Start Predict')
        _predict(mymodel, args, data_loader, data_set=test_dataset, RELOAD_MODEL='model_test.p', use_cuda=args.use_cuda)
    
    # root_dir = _here(os.pardir)
    # solution_path = os.path.join(root_dir, 's1/test.solution')
    # test_dataset = dataset._read_dataset(solution_path)
    # _eval(mymodel, args, data_loader, data_set=test_dataset)


if __name__ == '__main__':
    main()