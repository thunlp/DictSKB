import os
import torch
import numpy as np
import locale

dic_lemma = {}
for line in open('./data/lemmatization.txt', encoding='utf-8'):
    line = line.strip().split()
    dic_lemma[line[1]] = line[0]


# st_list = [ ',', '.','=','-', '...', ':', '@', '``', '``', '—', '(', ')', '[', ']', '$', 'N']
# templi = []
# keys = processed_sememe_dict.keys()
# from tqdm import tqdm
# temp_count = 0
# for word in tqdm(li):
#     word = word.lower()
#     if word in dic_lemma.keys():
#         word = dic_lemma[word]
#     if word in st_list:
#         continue
#     if word not in keys:
#         flag = 0
#         if word.isdigit():
#             if word != '²' and word != '³':
#                 digital_list = word_tokenize(translateNumberToEnglish(int(word)))
#                 for digital in digital_list:
#                     if digital not in keys:
#                         flag = 1
#             else:
#                 temp_count += 1
#         if flag:
#             count += 1
#             templi.append(word)
# # pprint.pprint(templi)
# print(temp_count)
# print(count / len(li))




class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # self.dict_sememe_source: dict = np.load('../PrepareSememeDict/core_sememe_dict_4.npy', allow_pickle=True).item()
        # self.addAllwordindict()

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    # def addAllwordindict(self):
    #     for word, _ in self.dict_sememe_source.items():
    #         self.add_word(word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        self.dict_sememe_source: dict = np.load('./data/sememe_dict_uncased.npy', allow_pickle=True).item()
        self.dic_lemma = self.read_lemmatization('./data/lemmatization.txt')
        self.dictionary.add_word('<unk>')
        self.dictionary.add_word('<number>')

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir, encoding='utf-8'):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma

    def tokenize(self, path):
        """Tokenizes a text file."""
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    word = word.lower()
                    if word in self.dict_sememe_source.keys():
                        self.dictionary.add_word(word)
                    else:
                        if word in self.dic_lemma.keys():
                            converted_word = self.dic_lemma[word]
                            if converted_word in self.dict_sememe_source.keys():
                                self.dictionary.add_word(converted_word)
                        else:
                            if word.isdigit():
                                continue
                            # if word in self.st_list:
                            #     self.dictionary.add_word(word)

                                # if word != '²' and word != '³':
                                #     digital_list = word_tokenize(translateNumberToEnglish(int(word)))
                                #     for digit in digital_list:
                                #         if digit in self.dict_sememe_source.keys():
                                #             self.dictionary.add_word(digit)

                            # else:
                            #     self.dictionary.add_word(converted_word)


        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            # ids = torch.LongTensor(tokens)
            ids = []
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    word = word.lower()
                    if word in self.dictionary.word2idx.keys():
                        ids.append(self.dictionary.word2idx[word])
                        token += 1
                    else:
                        if word in self.dic_lemma.keys():
                            converted_word = self.dic_lemma[word]
                            if converted_word in self.dictionary.word2idx.keys():
                                ids.append(self.dictionary.word2idx[converted_word])
                                token += 1
                            else:
                                ids.append(self.dictionary.word2idx['<unk>'])
                                token += 1
                        else:
                            if word.isdigit():
                                ids.append(self.dictionary.word2idx['<number>'])
                                token += 1
                                # if word != '²' and word != '³':
                                #     digital_list = word_tokenize(translateNumberToEnglish(int(word)))
                                #     for digit in digital_list:
                                #         if digit in self.dict_sememe_source.keys():
                                #             ids.append(self.dictionary.word2idx[digit])
                                #             token += 1
                                #         else:
                                #             ids.append(self.dictionary.word2idx['<unk>'])
                                #             token += 1
                                # else:
                                #     ids.append(self.dictionary.word2idx['<unk>'])
                                #     token += 1
                            else:
                                ids.append(self.dictionary.word2idx['<unk>'])
                                token += 1
            return torch.LongTensor(ids)


                    # if word in self.dict_sememe_source.keys():
                    #     ids[token] = self.dictionary.word2idx[word]
                    #     token += 1
                    # else:
                    #     if word in self.dic_lemma.keys():
                    #         converted_word = self.dic_lemma[word]
                    #         if converted_word in self.dict_sememe_source.keys():
                    #             ids[token] = self.dictionary.word2idx[converted_word]
                    #             token += 1
                    #         else:
                    #             ids[token] = self.dictionary.word2idx['<unk>']
                    #             token += 1
                    #     else:
                    #         ids[token] = self.dictionary.word2idx['<unk>']
                    #         token += 1
