from nltk.corpus import wordnet as wn
import pickle
# for synset in wn.synsets('car', pos=wn.NOUN):
#     print(synset.lemma_names())

with open('aux_files/dataset_13837.pkl', 'rb') as fh:
    dataset = pickle.load(fh)

def getposstr(wn_pos):
    if wn_pos == wn.NOUN:
        return 'noun'
    if wn_pos == wn.ADV:
        return 'adv'
    if wn_pos == wn.ADJ or wn_pos== wn.ADJ_SAT:
        return 'adj'
    if wn_pos == wn.VERB:
        return 'verb'

pos_list = [wn.NOUN, wn.ADV, wn.ADJ, wn.VERB]
vocab = {}
for w, i in dataset.dict.items():
    vocab[w] = i


tyc = {}
for w, i in vocab.items():
    tyc[i] = {}
    for pos in pos_list:
        syns = []
        for synset in wn.synsets(w, pos=pos):
            syns += synset.lemma_names()
        t = [vocab[s] for s in syns if s in vocab.keys() and s != w]
        tyc[i][getposstr(pos)] = list(set(t))
fw = open('word_cands_tyc.pkl', 'wb')
pickle.dump(tyc, fw)
