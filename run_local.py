
import os
import sys
import argparse
import logging

from multiprocessing import Process

VERBOSITY_LEVEL = 'INFO'
# VERBOSITY_LEVEL = 'WARNING'
logging.basicConfig(
    level = getattr(logging, VERBOSITY_LEVEL),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)


def _here(*args):
    # 返回包含此.py文件的目录
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(here, *args)

def _parse_args():
    default_starting_kit_dir = _here()
    default_dataset_dir = os.path.join(default_starting_kit_dir, 'd1')
    default_code_dir = os.path.join(default_starting_kit_dir, 'code')
    default_time_budget = 7200
    default_task = "baidu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset, contain .data and .solution file")

    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the code, contain .py file")

    parser.add_argument('--time_budget', type=float,
                        default=default_time_budget,
                        help="Time budget for training model")

    parser.add_argument('--task', type=str,
                        default=default_task,
                        help="Default task name - baidu")

    parser.add_argument('--mode', type=str,
                        default='train',
                        help="Mode of the task: train, eval, predict, model_select")

    parser.add_argument('--use_cuda', type=int,
                        default=0,
                        help="whether to use cuda - defualt False")
    args = parser.parse_args()
    return args

def run(dataset_dir, code_dir, task, time_budget=7200, mode='train', use_cuda=False):
    path_train = os.path.join(code_dir, 'train.py')
    path_select_param = os.path.join(code_dir, 'model_select.py')

    command_train = (
        'python3 -u '
        f'{path_train} --dataset_dir={dataset_dir} '
        f'--code_dir={code_dir} --time_budget={time_budget} --task={task} '
        f'--use_cuda={use_cuda} --mode={mode}'
    )

    command_select_param = (
        'python3 -u '
        f'{path_select_param} --dataset_dir={dataset_dir} '
        f'--code_dir={code_dir} --time_budget={time_budget} --task={task} '
        f'--use_cuda={use_cuda}'
    )


    def run_train():
        os.system(command_train)
    def run_select_param():
        os.system(command_select_param)

    train_process = Process(name='train', target=run_train)
    select_param_process = Process(name='select_param', target=run_select_param)

    if mode == 'model_select':
        select_param_process.start()
    else:
        train_process.start()

def main():
    args = _parse_args()

    logging.info("#"*80)
    logging.info("Begin running local test")
    logging.info(f"code_dir = {args.code_dir}")
    logging.info(f"dataset_dir = {args.dataset_dir}")
    logging.info(f"task = {args.task}")
    logging.info("#"*80)

    run(args.dataset_dir, args.code_dir, args.task, args.time_budget, args.mode, args.use_cuda)

if __name__ == '__main__':
    main()