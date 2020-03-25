from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
# from model import BLSTM_CRF, CRF
from model_bert_mlp import BERT_MLP
import os


EPOCH = 3


if __name__ == '__main__':
    data_set = AutoKGDataset('./d1/')
    train_dataset = data_set.train_dataset[:50]
    eval_dataset = data_set.dev_dataset[:50]
    os.makedirs('result', exist_ok=True)
    data_loader = KGDataLoader(data_set, rebuild=False, temp_dir='result/')
    train_data_mat_dict = data_loader.transform(train_dataset)

    data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)

    model_params = {
        'embedding_dim' : 768,
        # 'hidden_dim' : 64,
        'n_tags' : len(data_loader.ent_seq_map_dict),
        'n_words' : len(data_loader.character_location_dict),
        # 'start_idx': data_loader.ent_seq_map_dict[data_loader.START_TAG],  ## <start> tag index for entity tag seq
        # 'end_idx': data_loader.ent_seq_map_dict[data_loader.END_TAG],  ## <end> tag index for entity tag seq
        'use_cuda':False,
        # 'dropout_prob': 0,
        # 'lstm_layer_num': 1
    }
    mymodel = BERT_MLP(params=model_params)

    # hyper_param = {
    #     'EPOCH': 1,         #45
    #     'batch_size': 4,    #512
    #     'learning_rate': 1e-2,
    #     'visualize_length': 2, #10
    #     'isshuffle': True,
    #     'result_dir': 'result',
    #     'model_name':'model_test.p'
    # }
    # mymodel.train_model(data_loader, train_dataset, eval_dataset, hyper_param, use_cuda=False)

    # hyper_param = {
    #     'batch_size': 10,
    #     'issave': True,
    #     'result_dir': 'result'
    # }
    # mymodel.eval_model(data_loader, data_set=eval_dataset, hyper_param=hyper_param, use_cuda=False)

    for epoch in range(EPOCH):
        print('EPOCH: %d' % epoch)
        for data_batch in data_generator:
            x, pos, _, _, y_ent, lens, data_list = data_batch
            print(x.shape, pos.shape, y_ent.shape)    ##(batch_size, max_length)
            sentence = data_list[0]['input']
            # print([(i, sentence[i]) for i in range(len(sentence))])

            ###======================for BERT-MLP-MODEL only==================================
            mymodel._loss(x, y_ent, lens, use_cuda=False)
            mymodel._output(x, lens, use_cuda=False)

            print(x[0])
            word_dict = data_loader.character_location_dict
            rev_word_dict = data_loader.inverse_character_location_dict
            print(list(word_dict.items())[1300:1350])
            print(list(rev_word_dict.items())[1300:1350])
            print(sentence)
            print(list(rev_word_dict[i] for i in x[0]))
            ###======================for LSTM-CRF-MODEL only===================================
            # features = mymodel._get_lstm_features(x)

            # ###检验crf model 各层编码
            # tscore = mycrf._transition_score(y_ent, sentence_length)
            # escore = mycrf._ejection_score(features, y_ent, sentence_length)
            # pathscore = mycrf.path_score(features, y_ent, sentence_length)
            # vscore, paths = mycrf.viterbi_decode(features, sentence_length)
            # lognormscore = mycrf.log_norm_score(features, sentence_length)
            # print('t_score', tscore)
            # print('e_score', escore)
            # print('path_score', pathscore)
            # print('decode score', vscore)
            # print('log_norm_score', lognormscore)
            
            # ###检验BiLSTM-CRF 总model各层编码
            # score = mymodel.path_score(x, y_ent, sentence_length)
            # loss = mymodel.neg_log_likelihood(x, y_ent, sentence_length)
            # paths, scores = mymodel.predict(x, sentence_length)
            # print('bilstm-crf \t path score:\t', score.shape, score)
            # print('bilstm-crf \t loss:\t', loss.shape, loss)
            # print('bilstm-crf \t predict scores:\t', scores.shape, scores)
            # print('bilstm-crf \t predict paths:\t', paths.shape)
            # # print(paths)
            break
        break

