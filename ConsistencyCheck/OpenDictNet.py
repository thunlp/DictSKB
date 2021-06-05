import numpy as np

class DictNetdict():
    def __init__(self):
        self.sememe_dict: dict = np.load('./core_sememe_dict.npy', allow_pickle=True).item()
        self.dict_word = []
        for word, _ in self.sememe_dict.items():
            self.dict_word.append(word)

    def get_sememes_by_word(self, target_word, merge=False, structured=True, lang="en"):
        sememes_sets = self.sememe_dict[target_word]
        sememes_li = []
        for _, sememesset in sememes_sets:
            sememes_dic = {}
            sememes_dic['word'] = target_word
            sememes_dic['sememes'] = sememesset
            sememes_li.append(sememes_dic)
        if not merge:
            return sememes_li
        else:
            sememes = []
            for dic in sememes_li:
                sememes += list(dic['sememes'])
            return set(sememes)

    def get_word_pos(self, target_word):
        sememes_sets = self.sememe_dict[target_word]
        pos_list = []
        for pos, _ in sememes_sets:
            pos_list.append(pos)
        for i, pos in enumerate(pos_list):
            if pos == 'adjective':
                pos_list[i] = 'adj'
                continue
            if pos == 'adverb':
                pos_list[i] = 'adv'
                continue
        return pos_list

    def get_all_words(self):
        return self.dict_word



# dictnet = DictNetdict()
# print(dictnet.get_sememes_by_word('tackle'))
# # print(dictnet.get_sememes_by_word('kind', merge=True))
# print(dictnet.get_word_pos('tackle'))