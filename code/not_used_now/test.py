from dataset import AutoKGDataset
from utils import KGDataLoader, Batch_Generator
from model import BLSTM_CRF, CRF

from model import START_TAG, END_TAG

EPOCH = 3

def dict_add_key(d, k):
    d[k] = len(d)
    return d

if __name__ == '__main__':
    data_set = AutoKGDataset('./d1/')
    train_dataset = data_set.train_dataset[:10]

    data_loader = KGDataLoader(data_set, rebuild=False, istest=False, temp_dir='preprocessed_data/')
    train_data_mat_dict = data_loader.transform(train_dataset)

    data_generator = Batch_Generator(train_data_mat_dict, batch_size=4, data_type='ent', isshuffle=True)

    tag2idx = data_loader.ent_seq_map_dict
    tag2idx = dict_add_key(tag2idx, START_TAG)
    tag2idx = dict_add_key(tag2idx, END_TAG)
    word2idx = data_loader.character_location_dict


    mymodel = BLSTM_CRF(tag2idx, word2idx, embedding_dim=128, hidden_dim=64)
    mycrf = CRF(tag2idx)

    for epoch in range(EPOCH):
        print('EPOCH: %d' % epoch)
        for data_batch in data_generator:
            x, pos, _, _, y_ent, sentence_length, data_list = data_batch
            # print(x.shape, pos.shape, y_ent.shape)    ##(batch_size, max_length)
            # sentence = data_list[0]['input']
            # print([(i, sentence[i]) for i in range(len(sentence))])

            features = mymodel._get_lstm_features(x)

            ###检验crf model 各层编码
            tscore = mycrf._transition_score(y_ent, sentence_length)
            escore = mycrf._ejection_score(features, y_ent, sentence_length)
            pathscore = mycrf.path_score(features, y_ent, sentence_length)
            vscore, paths = mycrf.viterbi_decode(features, sentence_length)
            lognormscore = mycrf.log_norm_score(features, sentence_length)
            print('t_score', tscore)
            print('e_score', escore)
            print('path_score', pathscore)
            print('decode score', vscore)
            print('log_norm_score', lognormscore)
            
            ###检验BiLSTM-CRF 总model各层编码
            score = mymodel.path_score(x, y_ent, sentence_length)
            loss = mymodel.neg_log_likelihood(x, y_ent, sentence_length)
            paths, scores = mymodel.predict(x, sentence_length)
            print('bilstm-crf \t path score:\t', score.shape, score)
            print('bilstm-crf \t loss:\t', loss.shape, loss)
            print('bilstm-crf \t predict scores:\t', scores.shape, scores)
            print('bilstm-crf \t predict paths:\t', paths.shape)
            # print(paths)
            break
        break