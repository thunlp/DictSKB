import torch
import numpy as np
from tqdm import tqdm
import pickle

sense_tensor_dict: dict = np.load('../../PrepareSememeDict/sense_tensor_dict.npy', allow_pickle=True).item()
orig_candidates: dict = np.load('word_candidates_sense_1.pkl', allow_pickle=True)


def get_embedding_vec(word):
    if word in sense_tensor_dict.keys():
        return [tensor for _, tensor in sense_tensor_dict[word]]
    else:
        return []


def get_word_pos(word):
    if word in sense_tensor_dict.keys():
        target_vec_li = sense_tensor_dict[word]
        pos_list = []
        for pos, _ in target_vec_li:
            pos_list.append(pos)
        for i, pos in enumerate(pos_list):
            if pos == 'adjective':
                pos_list[i] = 'adj'
                continue
            if pos == 'adverb':
                pos_list[i] = 'adv'
                continue
        return pos_list
    else:
        return []


def get_candidates_scores(pos_list):
    candidates_scores = {}
    if 'adj' in pos_list:
        candidates_scores['adj'] = []
    if 'adv' in pos_list:
        candidates_scores['adv'] = []
    if 'noun' in pos_list:
        candidates_scores['noun'] = []
    if 'verb' in pos_list:
        candidates_scores['verb'] = []
    return candidates_scores


def get_pos_tensor_dict(pos_list):
    pos_dict = {}
    if 'adj' in pos_list:
        pos_dict['adj'] = []
    if 'adv' in pos_list:
        pos_dict['adv'] = []
    if 'noun' in pos_list:
        pos_dict['noun'] = []
    if 'verb' in pos_list:
        pos_dict['verb'] = []
    return pos_dict


def generate_candidates(processed_candidate_dict):
    with open('aux_files/dataset_13837.pkl', 'rb') as fh:
        dataset = pickle.load(fh)
    f = open('sss_dict.pkl', 'rb')
    NNS, NNPS, JJR, JJS, RBR, RBS, VBD, VBG, VBN, VBP, VBZ, inv_NNS, inv_NNPS, inv_JJR, inv_JJS, inv_RBR, inv_RBS, inv_VBD, inv_VBG, inv_VBN, inv_VBP, inv_VBZ = pickle.load(
        f)
    pos_list = ['noun', 'verb', 'adj', 'adv']
    pos_set = set(pos_list)

    s_ls = ['NNS', 'NNPS', 'JJR', 'JJS', 'RBR', 'RBS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    s_noun = ['NNS', 'NNPS']
    s_verb = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    s_adj = ['JJR', 'JJS']
    s_adv = ['RBR', 'RBS']
    # word_pos = {}
    # word_vec = {}
    word_candidate = {}
    #
    # for w1, i1 in dataset.dict.items():
    #     w1_s_flag = 0
    #     w1_orig = None
    #     for s in s_ls:
    #         if w1 in eval(s):
    #             w1_s_flag = 1
    #             w1_orig = eval(s)[w1]
    #             break
    #     if w1_s_flag == 0:
    #         w1_orig = w1
    #     embedding_vecs = get_embedding_vec(w1_orig)
    #     word_pos_list = get_word_pos(w1_orig)
    #     word_pos[i1] = word_pos_list
    #     word_vec[i1] = embedding_vecs
    #     assert len(embedding_vecs) == len(word_pos_list)
    # print(word_vec)
    # print(word_vec)
    # print(word_pos)
    # assert len(word_vec) == len(word_pos)
    for w1, i1 in tqdm(dataset.dict.items()):
        word_candidate[i1] = {}
        word_candidate[i1]['adj'] = []
        word_candidate[i1]['adv'] = []
        word_candidate[i1]['verb'] = []
        word_candidate[i1]['noun'] = []

        w1_s_flag = 0
        w1_orig = None
        for s in s_ls:
            if w1 in eval(s):
                w1_s_flag = 1
                w1_orig = eval(s)[w1]
                break
        if w1_s_flag == 0:
            w1_orig = w1

        count_dict = processed_candidate_dict[i1]
        if w1_orig not in sense_tensor_dict.keys():
            continue

        sense_tensor_list = sense_tensor_dict[w1_orig]
        w1_pos = get_word_pos(w1_orig)



        pos_tensor_dict = get_pos_tensor_dict(w1_pos)
        candidate_score_dict = get_candidates_scores(w1_pos)


        for pos, tensor in sense_tensor_list:
            if pos == 'adjective':
                temp_pos = 'adj'
            elif pos == 'adverb':
                temp_pos = 'adv'
            else:
                temp_pos = pos
            if temp_pos in pos_tensor_dict.keys():
                pos_tensor_dict[temp_pos].append(tensor)

        for w2, i2 in dataset.dict.items():
            if i1 == i2:
                continue

            w2_s_flag = 0
            w2_orig = None
            for s in s_ls:
                if w2 in eval(s):
                    w2_s_flag = 1
                    w2_orig = eval(s)[w2]
                    break
            if w2_s_flag == 0:
                w2_orig = w2

            if w2_orig not in sense_tensor_dict.keys():
                continue
            sense_tensor_list_w2 = sense_tensor_dict[w2_orig]
            w2_pos = get_word_pos(w2_orig)
            all_pos = set(w1_pos) & set(w2_pos) & pos_set
            if len(all_pos) == 0:
                continue
            for pos, tensor in sense_tensor_list_w2:
                if pos == 'adjective':
                    temp_pos = 'adj'
                elif pos == 'adverb':
                    temp_pos = 'adv'
                else:
                    temp_pos = pos
                if temp_pos in all_pos:
                    for matched_tensor in pos_tensor_dict[temp_pos]:
                        score = torch.matmul(matched_tensor, tensor).item()
                        candidate_score_dict[temp_pos].append((i2, score))
        for pos, score_list in candidate_score_dict.items():
            sorted_li = sorted(score_list, key=lambda items: items[1], reverse=True)
            chose_num = count_dict[pos]
            i = 0
            while len(word_candidate[i1][pos]) != chose_num:
                target_i = sorted_li[i][0]
                if target_i in word_candidate[i1][pos]:
                    i += 1
                else:
                    word_candidate[i1][pos].append(target_i)
                    i += 1
    return word_candidate


def preprocess_orig_candidates():
    processed_candidates = {}
    for id, candidates_dict in tqdm(orig_candidates.items()):
        processed_candidates[id] = {}
        for pos, candidates_list in candidates_dict.items():
            processed_candidates[id][pos] = len(candidates_list)
    return processed_candidates


if __name__ == '__main__':
    processed_candidates = preprocess_orig_candidates()
    candidate_dict = generate_candidates(processed_candidates)
    f = open('word_candidates_sense_baseline.pkl', 'wb')
    pickle.dump(candidate_dict, f)
    f.close()
