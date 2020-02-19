CUDA_VISIBLE_DEVICES=7 python3 run_local.py \
    --use_cuda='1' \
    --dataset_dir='./data/d5' \
    --answer_dir='./data/s5' \
    --code_dir='./code' \
    --task='baidu_person' \
    --mode='train' \
    | tee log_info.txt