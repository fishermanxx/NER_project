CUDA_VISIBLE_DEVICES=7 python3 run_local.py \
    --use_cuda='1' \
    --dataset_dir='./data/d6' \
    --answer_dir='./data/s6' \
    --code_dir='./code' \
    --task='renminribao' \
    --mode='train' \
    | tee log_info.txt
