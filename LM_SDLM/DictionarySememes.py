# coding: utf-8
import torch
import re
import os
import data
from data import Dictionary
import numpy as np
import locale
from nltk.tokenize import word_tokenize
NUMBER_CONSTANT = {0:"zero ", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven",
                8:"eight", 9:"nine", 10:"ten", 11:"eleven", 12:"twelve", 13:"thirteen",
                14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen", 18:"eighteen", 19:"nineteen" };
IN_HUNDRED_CONSTANT = {2:"twenty", 3:"thirty", 4:"forty", 5:"fifty", 6:"sixty", 7:"seventy", 8:"eighty", 9:"ninety"}
BASE_CONSTANT = {0:" ", 1:"hundred", 2:"thousand", 3:"million", 4:"billion"}
def translateNumberToEnglish(number):
    if str(number).isnumeric():
        if str(number)[0] == '0' and len(str(number)) > 1:
            return translateNumberToEnglish(int(number[1:]));
        if int(number) < 20:
            return NUMBER_CONSTANT[int(number)];
        elif int(number) < 100:
            if str(number)[1] == '0':
                return IN_HUNDRED_CONSTANT[int(str(number)[0])];
            else:
                return IN_HUNDRED_CONSTANT[int(str(number)[0])] + "-" + NUMBER_CONSTANT[int(str(number)[1])];
        else:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8' );
            strNumber = locale.format("%d", number, grouping=True);
            numberArray = str(strNumber).split(",")
            stringResult = ""
            groupCount = len(numberArray) + 1;
            for groupNumber in numberArray:
                if groupCount > 1 and groupNumber[0:] != "000":
                    stringResult += str(getUnderThreeNumberString(str(groupNumber))) + " ";
                else:
                    break;
                groupCount -= 1;
                if groupCount > 1:
                    stringResult += BASE_CONSTANT[groupCount] + ","
            endPoint = len(stringResult) - len(" hundred,")
            # return stringResult[0:endPoint];
            return stringResult

    else:
        print(number)
        raise RuntimeError
def getUnderThreeNumberString(number):
    if str(number).isnumeric() and len(number) < 4:
        if len(number) < 3:
            return translateNumberToEnglish(int(number))
        elif len(number) == 3 and number[0:] == "000":
            return " "
        elif len(number) == 3 and number[1:] == "00":
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1]
        else:
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1] + " and " + translateNumberToEnglish((number[1:]));
    else:
        print("number must below 1000")


class SememeDictionary(object):
    def __init__(self, corpus):
        self.word2idx = {}
        self.idx2word = []
        self.idx2freq = []
        self.idx2senses = []
        self.threshold = -1
        self.sememe_dict = Dictionary()
        self.threshold = 0
        self.dic_lemma = self.read_lemmatization('./data/lemmatization.txt')
        temp_sememe_source: dict = np.load('./data/sememe_dict_uncased.npy', allow_pickle=True).item()
        valid_word = corpus.dictionary.idx2word
        self.dict_sememe_source = {}
        for word, li in temp_sememe_source.items():
            if word in valid_word:
                self.dict_sememe_source[word] = li
        self.add_word('<unk>', ['<unk>'])
        self.add_word('N', ['cardinal'])
        self.add_word('<number>', ['number'])



        for word, sememe_list in self.dict_sememe_source.items():
            all_bag = []
            for pos, sememe_bag in sememe_list:
                self.add_word(word, sememe_bag)
                all_bag.append(sememe_bag)

        self.sememe_dict.idx2freq = [0] * len(self.sememe_dict)

    def senses_belong(self, sememes_bag, senses_bag):
        for i in range(len(senses_bag)):
            if len(set(sememes_bag + senses_bag[i])) == len(sememes_bag) \
                    and len(sememes_bag) == len(senses_bag[i]):
                return True
        return False

    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir, encoding='utf-8'):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma

    def add_word(self, word, sememes_bag):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.idx2senses.append([])
            self.idx2freq.append(0)
            self.word2idx[word] = len(self.idx2word) - 1

        idx = self.word2idx[word]
        sememe_bag_idx = []
        for sememe in sememes_bag:
            sememe_bag_idx.append(self.sememe_dict.add_word(sememe))
        sememe_bag_idx = list(set(sememe_bag_idx))
        if not self.senses_belong(sememe_bag_idx, self.idx2senses[idx]):
            self.idx2senses[idx].append(sememe_bag_idx)

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def summary(self, print_sememes=False):
        print('=' * 69)
        print('-' * 31 + 'SUMMARY' + '-' * 31)
        print('Number of Sememes: {}'.format(len(self.sememe_dict)))
        print('Number of Words: {}'.format(len(self.idx2word)))
        tot_senses = 0
        tot_sememes = 0
        for i in range(len(self.idx2word)):
            tot_senses += len(self.idx2senses[i])
            for j in range(len(self.idx2senses[i])):
                tot_sememes += len(self.idx2senses[i][j])
        ws_ratio = (tot_senses + 0.0) / len(self.idx2word)
        ss_ratio = (tot_sememes + 0.0) / tot_senses
        print('Mean Senses per Word: {}'.format(ws_ratio))
        print('Mean Sememes per Sense: {}'.format(ss_ratio))
        print('=' * 69)
        if print_sememes:
            print(','.join(self.sememe_dict.idx2word))

    def exist(self, word):
        return word in self.word2idx

    def add_word_f(self, word):
        word = word.lower()
        # if word == '-':
        #     print(word in self.word2idx)
        if word not in self.word2idx:
            if word in self.dic_lemma.keys():
                word = self.dic_lemma[word]
                if word not in self.word2idx:
                    print(word)
                    raise ValueError("Word don't exist a")
            else:
                if word.isdigit():
                    return
                else:
                    print(word)
                    print(word in self.word2idx)
                    raise ValueError("Word don't exist")

            # print(word)
            # raise ValueError("Word don't exist")
        idx = self.word2idx[word]
        for sense in self.idx2senses[idx]:
            for sememe in sense:
                self.sememe_dict.idx2freq[sememe] += 1
        self.idx2freq[self.word2idx[word]] += 1

    def query_count(self, word):
        if word not in self.word2idx:
            raise ValueError("Word don't exist")
        return self.idx2freq[self.word2idx[word]]

    def freq_le(self, k):
        tot = 0
        for idx in range(len(self.idx2word)):
            if self.idx2freq[idx] < k:
                tot += 1
        return tot

    def freq_ge(self, k):
        tot = 0
        for idx in range(len(self.idx2word)):
            if self.idx2freq[idx] >= k:
                tot += 1
        return tot

    def set_threshold(self, threshold):
        self.threshold = threshold

    def sememe_word_visit(self, word_dict):
        sememe_word = []
        sememe_sense = []
        for i in range(len(self.sememe_dict)):
            sememe_word.append([])
            sememe_sense.append([])
        maximum_senses = 0
        tot_senses = 0
        for word_id in range(len(self.word2idx)):
            if self.idx2freq[word_id] >= self.threshold:
                maximum_senses = max(maximum_senses, len(self.idx2senses[word_id]))
                for sense in self.idx2senses[word_id]:
                    # if tot_senses == 86:
                    #     print('here')
                    #     print(sense)
                    for sememe in sense:
                        # if tot_senses == 85:
                        #     raise RuntimeError
                        sememe_word[sememe].append(word_id)
                        sememe_sense[sememe].append(tot_senses)
                    tot_senses += 1


        tot = 0
        tot_sememes = 0
        max_words = 0
        a = []
        sememe_word_pair = [[], []]
        sememe_sense_pair = [[], []]
        sememe_idx = []
        word_sense = []
        for i in range(len(word_dict)):
            word_sense.append([])
        for i in range(len(self.sememe_dict)):
            cur_str = self.sememe_dict.idx2word[i]
            cur_str += ': '
            words = []
            for j in range(len(sememe_word[i])):
                word_id = sememe_word[i][j]
                sense_id = sememe_sense[i][j]
                words.append(self.idx2word[word_id])
                sememe_word_pair[0].append(tot_sememes)
                sememe_word_pair[1].append(word_dict[self.idx2word[word_id]])
                sememe_sense_pair[0].append(tot_sememes)
                sememe_sense_pair[1].append(sense_id)
                word_sense[word_dict[self.idx2word[word_id]]].append(sense_id)
            tot += len(sememe_word[i])
            max_words = max(max_words, len(sememe_word[i]))
            a += sememe_word[i]
            cur_str += ','.join(words)
            if len(set(sememe_word[i])) > 0:
                sememe_idx.append(tot_sememes)
            else:
                sememe_idx.append(-1)
            tot_sememes += len(sememe_word[i]) > 0
        for i in range(len(word_dict)):
            word_sense[i] = list(set(word_sense[i]))
        print('Total words: {}'.format(len(set(a))))
        print('Maximum words per sememe: {}'.format(max_words))
        print('Maximum sense per word: {}'.format(maximum_senses))
        print('Total respective semems: {}'.format(tot_sememes))
        print('Total sememe-word pairs: {}'.format(tot))
        return sememe_word_pair, sememe_idx, sememe_sense_pair, word_sense

    def visit(self, word, mode='full'):
        if word not in self.word2idx:
            raise ValueError('No word!')
        idx = self.word2idx[word]
        if mode == 'sbag':
            sememes = []
            for sense in self.idx2senses[idx]:
                for sememe in sense:
                    sememes.append(sememe)
            sememes = set(sememes)
            sememes_str = []
            for sememe in sememes:
                sememes_str.append(self.sememe_dict.idx2word[sememe])
            print(word + ':' + ','.join(sememes_str))
        if mode == 'full':
            print('Word: ' + word + ', total {} means'.
                  format(len(self.idx2senses[idx])))
            for i in range(len(self.idx2senses[idx])):
                sememes_list = []
                for j in range(len(self.idx2senses[idx][i])):
                    sememes_list.append(
                        self.sememe_dict.idx2word[self.idx2senses[idx][i][j]])
                sememes = ','.join(sememes_list)
                print('Sense #{}: '.format(i + 1) + sememes)


if __name__ == '__main__':

    corpus = data.Corpus('../LM_sub/data/wikitext-2')
    overall_dict = SememeDictionary(corpus)

    overall_dict.summary()



    def add_word_overall(source):
        for id in source:
            word = corpus.dictionary.idx2word[id]
            overall_dict.add_word_f(word)


    add_word_overall(corpus.train)
    add_word_overall(corpus.test)
    add_word_overall(corpus.valid)
    overall_dict.set_threshold(1)
    sememe_word_pair, sememe_idxs, sememe_sense_pair, _____ = \
        overall_dict.sememe_word_visit(corpus.dictionary.word2idx)
    # while True:
    #     str = input('Enter Word: ')
    #     str.strip('\n')
    #     overall_dict.visit(str)
