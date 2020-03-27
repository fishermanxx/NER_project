python3 run_local.py \
    --use_cuda='1' \
    --dataset_dir='./data/d3' \
    --answer_dir='./data/s3' \
    --code_dir='./rel_code' \
    --task='baidu_person_rel' \
    --mode='train' \
    | tee log_info.txt
