python3 run_local.py \
    --use_cuda='1' \
    --dataset_dir='./data/d4' \
    --answer_dir='./data/s4' \
    --code_dir='./rel_code' \
    --task='baidu_company' \
    --mode='train' \
    | tee log_info.txt
