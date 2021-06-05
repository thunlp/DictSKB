import pickle

f = open('word_candidates_sense.pkl', 'rb')
word_candidates: dict = pickle.load(f, )
f.close()
f = open('word_candidates_sense_hownet.pkl', 'rb')
word_candidates_hownet: dict = pickle.load(f, )
f.close()
f = open('word_candidates_sense_baseline.pkl', 'rb')
word_candidates_baseline: dict = pickle.load(f, )
f.close()


def CalculateTotalCandidates():
    hownet_candidates_num = 0
    dict_candidates_num = 0
    for _, candidate_dict in word_candidates.items():
        for _, candidate_list in candidate_dict.items():
            dict_candidates_num += len(candidate_list)
    for _, candidate_dict in word_candidates_hownet.items():
        for _, candidate_list in candidate_dict.items():
            hownet_candidates_num += len(candidate_list)
    print('dict total candidate number: ', dict_candidates_num)
    print('hownet total candidate number: ', hownet_candidates_num)


def CalculateWordsHaveCandidates():
    hownet_candidate_num = 0
    dict_candidate_num = 0
    temp = 0
    for _, candidate_dict in word_candidates.items():
        for _, candidate_list in candidate_dict.items():
            if len(candidate_list) != 0:
                dict_candidate_num += 1
                break
    for _, candidate_dict in word_candidates_hownet.items():
        for _, candidate_list in candidate_dict.items():
            if len(candidate_list) != 0:
                hownet_candidate_num += 1
                break
    for _, candidate_dict in word_candidates_baseline.items():
        for _, candidate_list in candidate_dict.items():
            if len(candidate_list) != 0:
                temp += 1
                break
    print('dict total words that have candidates: ', dict_candidate_num)
    print('hownet total words that have candidates: ', hownet_candidate_num)
    print(temp)

def CalculatePerPOSHaveCandidates():
    hownet_candidate_num = 0
    dict_candidate_num = 0
    temp = 0
    for _, candidate_dict in word_candidates.items():
        for _, candidate_list in candidate_dict.items():
            if len(candidate_list) != 0:
                dict_candidate_num += 1
    for _, candidate_dict in word_candidates_hownet.items():
        for _, candidate_list in candidate_dict.items():
            if len(candidate_list) != 0:
                hownet_candidate_num += 1
    for _, candidate_dict in word_candidates_baseline.items():
        for _, candidate_list in candidate_dict.items():
            if len(candidate_list) != 0:
                temp += 1
    print('dict total POS that have candidates: ', dict_candidate_num)
    print('hownet total POS that have candidates: ', hownet_candidate_num)
    print(temp)

if __name__ == '__main__':
    CalculateTotalCandidates()
    CalculateWordsHaveCandidates()
    CalculatePerPOSHaveCandidates()
