from dataset import AutoKGDataset
from utils import BaseLoader
from utils import  KGDataLoader
import re
import numpy as np
import sys
import matplotlib.pyplot as plt

def draw_loss(loss_path, names, img_path='test.jpg'):
    loss_all = np.loadtxt(loss_path)
    for i in range(len(names)):
        plt.plot(loss_all[i], label=names[i])
    plt.legend()
    plt.savefig(img_path)

def check_data_case(data1, data_loader):
    print('==========================CHECK==============================')
    print(data1['input'])
    entitys = data1['output']['entity_list']
    brief_entitys = [(entitys[i]['entity'], entitys[i]['entity_type']) for i in range(len(entitys))]
    print(brief_entitys)
    print('=============================================================')
    print()
    transf_data = data_loader.transform([data1])
    emb_input = transf_data['cha_matrix'][0]
    print(emb_input)
    print(emb_input.max())
    # transf_data['y_ent_matrix']
    # transf_data['pos_matrix']
    # transf_data['sentence_length']
    # transf_data['data_list']

def show_dict(dictname, d, k=10):
    print('='*100)
    print(dictname, len(d))
    print(list(d.items())[:k])

def is_seq_equal(arr1, arr2):
    check = (arr1 == arr2)
    return check.sum() == len(check)

def print_idseq(seq):
    print([(i, seq[i]) for i in range(len(seq))])

def check_trn_scr(trn_scr, trn_all, trans, label_ext):
    for batch_id in range(trn_scr.shape[0]):
        label_seq = label_ext[batch_id].detach().numpy()
        # print('label_seq')
        # print(len(label_seq), [(i, label_seq[i]) for i in range(len(label_seq))])

        ##check for trn_all
        flag = True
        for i in range(trn_all.shape[1]):
            pre = trn_all[batch_id, i, :].detach().numpy()
            tar = trans[:, label_ext[batch_id, 1+i]].detach().numpy()

            if not is_seq_equal(pre, tar):
                flag = False
        if flag:
            print('trn_all is correct')
        else:
            print('trn_all is wrong')

        ##check for trn_scr
        pre_scr = trn_scr[batch_id].detach().numpy()
        tar_scr = []
        for i in range(len(label_seq)-1):
            from_tag = label_seq[i]
            to_tag = label_seq[i+1]
            temp_scr = trans[from_tag, to_tag].item()
            tar_scr.append(temp_scr)
        tar_scr = np.array(tar_scr)

        # print_idseq(tar_scr)

        if is_seq_equal(pre_scr, tar_scr):
            print('trn_scr is correct')
        else:
            print('trn_scr is wrong')
    
def check_mask_seq(mask, seq):
    mask = mask.detach().numpy()
    seq = seq.detach().numpy()
    print(len(seq[0]))
    for batch_id in range(len(mask)):
        mask_i = mask[batch_id]
        seq_i = seq[batch_id]
        # print(len(mask_i), len(seq_i))
        # print_idseq(mask_i)
        print_idseq(seq_i)

def check_char_matrix_dict(encode_sentence, char_d):
    decode_sentence = [char_d[i] for i in encode_sentence if i!=0]
    print('='*80)
    print(encode_sentence)
    print(decode_sentence)

def check_ent_matrix_dict(encode_sentence, eseq_d, etype_d):
    def ent_type(s):
        if s == 'ELSE':
            return s
        idx = int(re.findall(r'\d+', s)[0])
        etype = etype_d[idx]
        return etype
    decode_s1 = [eseq_d[i] for i in encode_sentence]
    decode_s2 = [ent_type(eseq_d[i]) for i in encode_sentence]
    print('='*80)
    print('encode_setence: ', encode_sentence)
    print('decode_s1 ', decode_s1)
    print('decode_s2 ', decode_s2)

if __name__ == '__main__':
    # data_set = AutoKGDataset('./d1/')
    # data_loader = KGDataLoader(data_set, rebuild=False, istest=False, temp_dir='preprocessed_data/')

    # show_dict('ent_seq_map_dict', data_loader.ent_seq_map_dict)
    # show_dict('inverse_ent_seq_map_dict', data_loader.inverse_ent_seq_map_dict)

    # show_dict('rel_seq_map_dict', data_loader.rel_seq_map_dict)
    # show_dict('inverse_rel_seq_map_dict', data_loader.inverse_rel_seq_map_dict)

    # show_dict('entity_type_dict', data_loader.entity_type_dict)
    # show_dict('inverse_entity_type_dict', data_loader.inverse_entity_type_dict)

    # test_data_set = data_set.train_dataset[:100]
    # res_dict = data_loader.transform_ent(test_data_set, istest=False)


    # char_matrix = res_dict['cha_matrix']
    # y_ent_matrix = res_dict['y_ent_matrix']
    # for i in range(len(test_data_set)):
    #     data1 = test_data_set[i]
    #     print(data1['input'])
    #     e_list = data1['output']['entity_list']
    #     for e in e_list:
    #         print(e['entity_type'], '--->', e['entity'])   
    #     # check_char_matrix_dict(char_matrix[i], data_loader.inverse_character_location_dict)
    #     # check_ent_matrix_dict(y_ent_matrix[i], data_loader.inverse_ent_seq_map_dict, data_loader.inverse_entity_type_dict)
    #     # print(char_matrix[0])
    #     test = data_loader._obtain_entity(y_ent_matrix[i], data1['input'])
    #     print(test)
    #     break

    # trans_back = data_loader.transform_back(res_dict, data_type='ent')
    
    # print(test_data_set[0])
    # print('='*80)
    # print(trans_back[0])


    draw_loss('/Users/work-xx-pc/Desktop/xx-work/bilstm_crf_test/result/loss_all.txt', [1e-3, 5e-3, 1e-2, 5e-2], 'loss_img.jpg')