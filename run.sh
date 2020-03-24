python3 run_local.py \
    --use_cuda='1' \
    --dataset_dir='./data/d15' \
    --answer_dir='./data/s15' \
    --code_dir='./rel_code' \
    --task='relation' \
    --mode='train' \
    | tee log_info.txt
