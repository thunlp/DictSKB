import pickle
from keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append('..')
from gptlm import GPT2LM

VOCAB_SIZE = 13837
max_len = 250
# tool = language_tool_python.LanguageTool('en-US')

with open('aux_files/dataset_%d.pkl' % VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
#
# pkl_file = open('AD_dpso_sem_bert.pkl', 'rb')
# success_list, fail_list, adv_orig_label, adv_orig, adv_list, dist_list = pickle.load(pkl_file)
# pkl_file.close()

# pkl_file = open('AD_dpso_sem.pkl', 'rb')
# success_list_lstm, fail_list_lstm, adv_orig_label_lstm, adv_orig_lstm, adv_list_lstm, dist_list_lstm = pickle.load(pkl_file)
# pkl_file.close()



pkl_file = open('./imdb_dictionary.pickle', 'rb')
word_dict, inv_word_dict = pickle.load(pkl_file)
pkl_file.close()



test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')


def load_sent_li(adv_list, adv_orig):
    target_li = []
    for sent in adv_list:
        sent_li = sent.tolist()
        temp_li = [inv_word_dict[idx] for idx in sent_li if idx != 0]
        target_li.append(' '.join(temp_li))

    origin_li = []
    for sent_id in adv_orig:
        sent = test_x[sent_id]
        sent_li = sent.tolist()
        temp_li = [inv_word_dict[idx] for idx in sent_li if idx != 0]
        origin_li.append(' '.join(temp_li))
    return origin_li, target_li





def getPPL(adv_li):
    lm = GPT2LM()
    total_ppl = 0
    from tqdm import tqdm
    for sent in tqdm(adv_li):
        loss = lm(sent)
        # print(loss)
        total_ppl += loss
    print(total_ppl / len(adv_li))


# def getGrammarError():
#     origin_total_error = 0
#     adv_total_error = 0
#     for sent in origin_li:
#         error = len(tool.check(sent))
#         origin_total_error += error
#     for sent in adv_li:
#         error = len(tool.check(sent))
#         adv_total_error += error
#     print((adv_total_error - origin_total_error) / len(origin_li))

def getModifyRate():
    import numpy as np
    print(np.mean(dist_list))
def getPPLLIST(adv_li):
    pplli = []
    lm = GPT2LM()
    total_ppl = 0
    from tqdm import tqdm
    for sent in tqdm(adv_li):
        loss = lm(sent)
        pplli.append(loss)
        total_ppl += loss
    return pplli

if __name__ == '__main__':
    pkl_file = open('AD_dpso_sem_dict.pkl', 'rb')
    success_list_lstm, fail_list_lstm, adv_orig_label_lstm, adv_orig_lstm, adv_list_lstm, dist_list_lstm = pickle.load(
        pkl_file)
    pkl_file.close()
    origin_li_lstm1, adv_li_lstm1 = load_sent_li(adv_list_lstm, adv_orig_lstm)
    getPPL(adv_li_lstm1)
    # pkl_file = open('AD_dpso_sem_baseline.pkl', 'rb')
    # success_list_lstm2, fail_list_lstm2, adv_orig_label_lstm2, adv_orig_lstm2, adv_list_lstm2, dist_list_lstm2 = pickle.load(
    #     pkl_file)
    # pkl_file.close()

    # origin_li_lstm1, adv_li_lstm1 = load_sent_li(adv_list_lstm, adv_orig_lstm)
    # origin_li_lstm2, adv_li_lstm2 = load_sent_li(adv_list_lstm2, adv_orig_lstm2)
    # pplli_lstm1 = getPPLLIST(adv_li_lstm1)
    # pplli_lstm2 = getPPLLIST(adv_li_lstm2)

    # from scipy import stats


    # print(stats.ttest_ind(success_list_lstm, success_list_lstm2))
    # getModifyRate()
    # lm = GPT2LM()
    # print(lm('ritchie \'s treatment of the class reversal is ham fisted from the repetitive that keep getting thrown in people \'s faces to the fact is such a joke'))
    # print(lm('Ritchie\'s treatment of the class reversal is ham fisted from the repetitive that keep getting thrown in people\'s faces to the fact is such a joke'))
    # getGrammarError()

