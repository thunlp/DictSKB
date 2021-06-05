import torch
import torch.nn as nn
from consistency_utils import *
import argparse
from tqdm import tqdm
from EvaluationDataset import *
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--descending_c_dict', type=float, default=0.6)
parser.add_argument('--descending_c_hownet', type=float, default=0.6)
parser.add_argument('--threshold_dict', type=float, default=0.2)
parser.add_argument('--threshold_hownet', type=float, default=0.2)
parser.add_argument('--eval_epochs', type=int, default=20)
parser.add_argument('--eval_type', type=str, default='dict')
args = parser.parse_args()
print(args)



Dic_dict, hownet_dict = get_sememe_dict()
dict_sememes_list, hownet_sememes_list = get_sememe_list()
dict_sememes_inv, hownet_sememes_inv = get_sememes_inv()
dict_valid_word, hownet_valid_word = get_valid_word()
dict_valid_word_inv, hownet_valid_word_inv = get_valid_word_inv()

device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')


descending_dict_vec = torch.zeros(90000)
descending_hownet_vec = torch.zeros(90000)

for i in range(90000):
    descending_dict_vec[i] = pow(args.descending_c_dict, i)
    descending_hownet_vec[i] = pow(args.descending_c_hownet, i)




def Get_AP(sememeStd, sememePre, typ):
    '''
    Calculate the Average Precision of sememe prediction
    '''
    AP = 0
    hit = 0
    for i in range(len(sememePre)):
        if sememePre[i] in sememeStd:
            hit += 1
            AP += float(hit) / (i + 1)
    if AP == 0:
        # print('Calculate AP Error')
        # if typ == 'dict':
        #     print('Sememe Standard:' + ' '.join([dict_sememes_list[id] for id in sememeStd]))
        #     print('Sememe Predicted:' + ' '.join([dict_sememes_list[id] for id in sememePre]))
        # else:
        #     print('Sememe Standard:' + ' '.join([hownet_sememes_list[id] for id in sememeStd]))
        #     print('Sememe Predicted:' + ' '.join([hownet_sememes_list[id] for id in sememePre]))
        return 0
    else:
        AP /= float(len(sememeStd))
    # if typ == 'dict':
    #     print('AP: ', AP)
    #     print('Sememe Standard:' + ' '.join([dict_sememes_list[id] for id in sememeStd]))
    #     print('Sememe Predicted:' + ' '.join([dict_sememes_list[id] for id in sememePre]))
    #     print('*' * 10)
    return AP


def Get_F1(sememeStdList, sememeSelectList):
    '''
    Calculate the F1 score of sememe prediction
    '''
    TP = len(set(sememeStdList) & set(sememeSelectList))
    FP = len(sememeSelectList) - TP
    FN = len(sememeStdList) - TP
    if TP+FP == 0:
        precision= 0
    else:
        precision = float(TP) / (TP + FP)
    if TP+FN == 0:
        recall = 0
    else:
        recall = float(TP) / (TP + FN)
    if (precision + recall) == 0:
        return 0, 0, 0
    F1 = 2 * precision * recall / (precision + recall)
    return F1, precision, recall


ap_result = []
f1_result = []
def computeScore(refer_data_vec, refer_data_sememes_tag_tensors, test_data_vec, test_data_sememes_tag_tensors, typ='dict'):
    refer_data_sememes_tag_tensors = refer_data_sememes_tag_tensors.to(device)

    total_ap = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0

    refer_data_vec = refer_data_vec.cpu()
    refer_data_vec_norm = torch.norm(refer_data_vec, dim=1, keepdim=True)
    refer_data_vec = refer_data_vec / refer_data_vec_norm

    test_data_vec = test_data_vec.transpose(0, 1).cpu()
    test_data_vec_norm = torch.norm(test_data_vec, dim=0, keepdim=True)
    test_data_vec = test_data_vec / test_data_vec_norm

    relevant_matrix = torch.matmul(refer_data_vec, test_data_vec).transpose(0, 1)

    sorted_relevant_matrix, sorted_record_idx = torch.sort(relevant_matrix, dim=1, descending=True)
    # print(sorted_relevant_matrix.size(), descending_dict_vec.size())
    descending_relevant_matrix = torch.mul(sorted_relevant_matrix, descending_dict_vec)

    for i in tqdm(range(10000)):
        sememe_golden_tag_list = torch.where(condition=(test_data_sememes_tag_tensors[i] == 1))[
            0].cpu().numpy().tolist()

        matched_vector = descending_relevant_matrix[i].to(device)  # 90000
        sorted_idx = sorted_record_idx[i].to(device)
        re_selected_sememes_tag_tensors = torch.index_select(refer_data_sememes_tag_tensors, dim=0,
                                                             index=sorted_idx)  # 90000, 2500
        sememe_score = torch.matmul(matched_vector, re_selected_sememes_tag_tensors)  # 2535
        if typ == 'dict':
            sememe_Pre_tensor = torch.where(condition=(sememe_score > args.threshold_dict))[0]
        else:
            sememe_Pre_tensor = torch.where(condition=(sememe_score > args.threshold_hownet))[0]
        sememe_selected_output = torch.index_select(sememe_score, dim=0, index=sememe_Pre_tensor)
        _, idx = sememe_selected_output.sort(dim=0, descending=True)
        sememe_Pre_list = torch.index_select(sememe_Pre_tensor, dim=0, index=idx).cpu().numpy().tolist()

        ap = Get_AP(sememe_golden_tag_list, sememe_Pre_list, typ)
        f1, precision, recall = Get_F1(sememe_golden_tag_list, sememe_Pre_list)
        total_ap += ap
        ap_result.append(ap)
        total_f1 += f1
        f1_result.append(f1)
        total_precision += precision
        total_recall += recall
    mAP = total_ap / 10000
    mF1 = total_f1 / 10000
    mprecision = total_precision / 10000
    mrecall = total_recall / 10000
    return mAP, mF1, mprecision, mrecall


def dict_sememe_predict_eval(epoch):
    sememe_embedding = nn.Embedding(len(dict_sememes_list), 300).to(device)
    dict_sememes_embedding = sememe_embedding
    all_refer_data_vec,  all_senses_sememes_tag_tensors = getEvaluationDictdata(dict_sememes_embedding, device, )
    split_list = [0]*100000
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    all_mAP = 0
    all_mF1 = 0
    all_mprecision = 0
    all_mrecall = 0
    for i, (train_ids, test_ids) in enumerate(sfolder.split(split_list, split_list)):
        # print('here')
        print('epoch:{}/{}, k-fold: {}/{}. Begin'.format(epoch+1, 10, i+1, 10))
        train_ids_li = train_ids.tolist()
        test_ids_li = test_ids.tolist()
        refer_data_vec = []
        refer_data_sememes_tag = []
        test_data_vec = []
        test_data_sememes_tag = []
        for i in train_ids_li:
            refer_data_vec.append(all_refer_data_vec[i])
            refer_data_sememes_tag.append(all_senses_sememes_tag_tensors[i])
        for i in test_ids_li:
            test_data_vec.append(all_refer_data_vec[i])
            test_data_sememes_tag.append(all_senses_sememes_tag_tensors[i])
        refer_data_vec = torch.stack(refer_data_vec, dim=0)
        refer_data_sememes_tag = torch.stack(refer_data_sememes_tag, dim=0)
        test_data_vec = torch.stack(test_data_vec, dim=0)
        test_data_sememes_tag = torch.stack(test_data_sememes_tag, dim=0)
        mAP, mF1, mprecision, mrecall = computeScore(refer_data_vec, refer_data_sememes_tag, test_data_vec, test_data_sememes_tag,
                                'dict')
        all_mAP += mAP
        all_mF1 += mF1
        all_mprecision += mprecision
        all_mrecall += mrecall


    return all_mAP / 10, all_mF1 / 10, all_mprecision / 10 , all_mrecall / 10


def hownet_sememe_predict_eval(epoch):
    sememe_embedding = nn.Embedding(len(hownet_sememes_list), 300).to(device)
    hownet_sememes_embedding = sememe_embedding
    all_refer_data_vec,  all_senses_sememes_tag_tensors = getEvaluationHownetdata(hownet_sememes_embedding, device)
    split_list = [0]*100000
    sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    all_mAP = 0
    all_mF1 = 0
    all_mprecision = 0
    all_mrecall = 0
    for i, (train_ids, test_ids) in enumerate(sfolder.split(split_list, split_list)):
        print('epoch:{}/{}, k-fold: {}/{}. Begin'.format(epoch + 1, 10, i + 1, 10))
        train_ids_li = train_ids.tolist()
        test_ids_li = test_ids.tolist()
        refer_data_vec = []
        refer_data_sememes_tag = []
        test_data_vec = []
        test_data_sememes_tag = []
        for i in train_ids_li:
            refer_data_vec.append(all_refer_data_vec[i])
            refer_data_sememes_tag.append(all_senses_sememes_tag_tensors[i])
        for i in test_ids_li:
            test_data_vec.append(all_refer_data_vec[i])
            test_data_sememes_tag.append(all_senses_sememes_tag_tensors[i])
        refer_data_vec = torch.stack(refer_data_vec, dim=0)
        refer_data_sememes_tag = torch.stack(refer_data_sememes_tag, dim=0)
        test_data_vec = torch.stack(test_data_vec, dim=0)
        test_data_sememes_tag = torch.stack(test_data_sememes_tag, dim=0)
        mAP, mF1, mprecision, mrecall = computeScore(refer_data_vec, refer_data_sememes_tag, test_data_vec, test_data_sememes_tag,
                                'hownet')
        all_mAP += mAP
        all_mF1 += mF1
        all_mprecision += mprecision
        all_mrecall += mrecall

    return all_mAP / 10, all_mF1 / 10, all_mprecision / 10 , all_mrecall / 10


def evaluation(type):
    epochs = args.eval_epochs
    total_mAP = 0
    total_mF1 = 0
    total_mprecision = 0
    total_mrecall = 0
    for j in range(epochs):
        if type == 'dict':
            mAP, mF1, mprecision, mrecall = dict_sememe_predict_eval(j)
        else:
            mAP, mF1, mprecision, mrecall = hownet_sememe_predict_eval(j)
        total_mAP += mAP
        total_mF1 += mF1
        total_mprecision += mprecision
        total_mrecall += mrecall
    mean_map = total_mAP / epochs
    mean_mf1 = total_mF1 / epochs
    mean_mprecision = total_mprecision / epochs
    mean_mrecall = total_mrecall / epochs
    print('Eval_type: {}, mAP: {}, mF1: {}, mprecision: {}, mrecall: {}'
          .format(type, mean_map, mean_mf1, mean_mprecision, mean_mrecall))
    import numpy as np
    np.save(type + 't-test-ap', ap_result)
    np.save(type + 't-test-f1', f1_result)


if __name__ == '__main__':
    evaluation(args.eval_type)
