import pickle
file_path = 'word_candidates_sense.pkl'
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
np.random.seed(1111)
VOCAB_SIZE = 13837
with open('aux_files/dataset_%d.pkl' % VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
with open(file_path, 'rb') as fp:
    word_candidate = pickle.load(fp)
with open('pos_tags_test.pkl', 'rb') as fp:
    test_pos_tags = pickle.load(fp)

max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

SAMPLE_SIZE = len(dataset.test_y)
test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)

pos_list = ['JJ', 'NN', 'RB', 'VB']

all_neighbor_list = []
for j in tqdm(test_idx):
    pos_tags = test_pos_tags[test_idx[j]]
    x_orig = test_x[test_idx[j]]
    neigbhours_list = []
    x_adv = x_orig.copy()
    x_len = np.sum(np.sign(x_orig))
    x_len = int(x_len)
    for i in range(x_len):
        if x_adv[i] not in range(1, 50000):
            neigbhours_list.append([])
            continue
        pair = pos_tags[i]
        if pair[1][:2] not in pos_list:
            # print('here')
            neigbhours_list.append([-1])
            continue
        if pair[1][:2] == 'JJ':
            pos = 'adj'
        elif pair[1][:2] == 'NN':
            pos = 'noun'
        elif pair[1][:2] == 'RB':
            pos = 'adv'
        else:
            pos = 'verb'
        if pos in word_candidate[x_adv[i]]:
            neigbhours_list.append([neighbor for neighbor in word_candidate[x_adv[i]][pos]])
        else:
            # print('here')
            # raise RuntimeError
            neigbhours_list.append([-1])
    all_neighbor_list.append(neigbhours_list)

all_candidates_word = 0
candidate_word = 0
for candidate_list in all_neighbor_list:
    for li in candidate_list:
        if -1 not in li:
            all_candidates_word += len(li)
            candidate_word += 1
print(all_candidates_word / candidate_word)

