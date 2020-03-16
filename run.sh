CUDA_VISIBLE_DEVICES=4 python3 run_local.py \
    --use_cuda='1' \
    --dataset_dir='./data/d4' \
    --answer_dir='./data/s4' \
    --code_dir='./code' \
    --task='renminribao' \
    --mode='train' \
    | tee log_info.txt
