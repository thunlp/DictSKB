import numpy as np
import OpenHowNet
import OpenDictNet



hownet_dict = OpenHowNet.HowNetDict()
Dic_dict = OpenDictNet.DictNetdict()

dict_sememesdict = Dic_dict.sememe_dict


dict_sememes_list: list = np.load('./dict_sememes.npy', allow_pickle=True).tolist()
hownet_sememes_list: list = np.load('./hownet_sememes.npy', allow_pickle=True).tolist()
dict_sememes_inv = {}
hownet_sememes_inv = {}
for i, sememe in enumerate(dict_sememes_list):
    dict_sememes_inv[sememe] = i
for i, sememe in enumerate(hownet_sememes_list):
    hownet_sememes_inv[sememe] = i

hownet_valid_word: list = np.load('./hownet_valid_word.npy', allow_pickle=True).tolist()
dict_valid_word: list = np.load('./dict_valid_word.npy', allow_pickle=True).tolist()
hownet_valid_word_inv = {}
dict_valid_word_inv = {}
for i, word in enumerate(hownet_valid_word):
    hownet_valid_word_inv[word] = i
for i, word in enumerate(dict_valid_word):
    dict_valid_word_inv[word] = i

dict_words = Dic_dict.get_all_words()
hownet_words = hownet_dict.get_en_words()




def get_sememe_list():
    return dict_sememes_list, hownet_sememes_list


def get_sememes_inv():
    return dict_sememes_inv, hownet_sememes_inv


def get_sememe_dict():
    return Dic_dict, hownet_dict



def get_valid_word():
    return dict_valid_word, hownet_valid_word


def get_valid_word_inv():
    return dict_valid_word_inv, hownet_valid_word_inv


def get_all_words():
    return dict_words, hownet_words

def get_all_sensesAndsememes():
    dict_all_senses_sememes = []
    hownet_all_senses_sememes = []
    for word, sememe_list in dict_sememesdict.items():
        for pos, sememe_set in sememe_list:
            li = []
            for sememe in sememe_set:
                li.append(dict_sememes_inv[sememe])
            if len(li) != 0:
                dict_all_senses_sememes.append(set(li))
    for word in hownet_words:
        li = []
        sememes_set = hownet_dict.get_sememes_by_word(word, structured=False, lang="en", merge=True)
        if isinstance(sememes_set, dict):
            continue
        for sememe in sememes_set:
            li.append(hownet_sememes_inv[sememe])
        if len(li) != 0:
            hownet_all_senses_sememes.append(set(li))
    return dict_all_senses_sememes, hownet_all_senses_sememes


