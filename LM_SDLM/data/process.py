def process_core_sememe_dict(sememe_source:dict):
    # sememe_source: dict = np.load('../PrepareSememeDict/core_sememe_dict.npy', allow_pickle=True).item()
    sememe_dict = {}
    for word, sememe_list in sememe_source.items():
        temp_li = []
        for pos, sememe_bag in sememe_list:
            if len(sememe_bag) != 0 and sememe_bag not in temp_li:
                temp_li.append((pos, sememe_bag))
        if len(temp_li) != 0:
            sememe_dict[word] = temp_li
    return sememe_dict
import numpy as np
sememe_dict:dict = np.load('./sememe_dict_uncased.npy',allow_pickle=True).item()
sememe_dict = process_core_sememe_dict(sememe_dict)
np.save('./sememe_dict_uncased.npy', sememe_dict)