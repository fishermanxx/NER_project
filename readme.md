NER and Relation extraction Project
=====
# Requirements:
    0) python 3.6
    1) numpy 1.18.1
    2) sklearn 0.22.1
    3) matplotlib 3.1.3
    4) torch 1.4.0
    5) torchvision 0.5.0
    6) jieba 0.42.1
    7) transformers 2.4.1
    8) pytorch-crf  0.7.2

# File path:
    ./data/d1/: the dir of dataset
    ./data/s1/: the dir of solution
    ./data/vocab.txt: the pretrained tokenize dict used by bert
    ./code: the dir of code
    ./readme.md
    ./run.sh
    ./run_local.py
    ./requirements.txt

# Usage
    bash run.sh 

To change the task mode, you can choose from `train, eval, predict` in the file `run.sh`:

    --mode='train'    

To change the dataset for training, you need to change the file `run.sh`:

    --dataset_dir='./data/d5'
    --answer_dir='./data/s5'
    --task='xxx'

To change to different model for training, you need to change the file `./code/train.py(main)`, such as 

    # mymodel = BLSTM_CRF(model_params, show_param=True)   
    # mymodel = BERT_LSTM_CRF(model_params, show_param=True) 
    # mymodel = BERT_MLP(model_params, show_param=True)
    mymodel = BERT_NER(model_params, show_param=True)
    # mymodel = BERT_CRF(model_params, show_param=True)

To change the training parameters such as learning_rate, batch_size, training_epoch, you need to change the file `./code/train.py(_train)`

    train_param = {
        'EPOCH': 15,         
        'batch_size': 64,    
        'learning_rate_bert': 5e-5,
        'learning_rate_upper': 5e-3,
        'bert_finetune': True,
        'visualize_length': 20,
        'isshuffle': True,
        'result_dir': args.result_dir,
        'model_name':'model_test.p'
    }

# Training output result
it will generate a fold at the root dir such as `./result/` and a log file `./log_info.txt`, the `./result/` file containing:

    joint_embedding_info_dict.pkl  #preprocess dict info
    model_test.p  # trained model result
    test_data_mat_dict.pkl  #preprocessed test dataset
    train_data_mat_dict.pkl  #preprocessed train dataset
    loss_train.txt 
    score_train.txt
    loss.png
    score.png

# Dataset details
| 表格      | 第一列     | 第二列     |
| ---------- | :-----------:  | :-----------: |
| 第一行     | 第一列     | 第二列     |