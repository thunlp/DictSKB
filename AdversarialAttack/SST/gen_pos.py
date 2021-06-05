import pickle


with open('./aux_files/dataset_13837.pkl','rb') as fp:
    dataset=pickle.load(fp)
from nltk.tag import StanfordPOSTagger
jar = 'stanford-postagger-full-2018-10-16/stanford-postagger.jar'
model = 'stanford-postagger-full-2018-10-16/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

from tqdm import tqdm

train_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.train_seqs]
test_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.test_seqs]
all_pos_tags=[]
test_pos_tags=[]
for text in tqdm(train_text):
    pos_tags = pos_tagger.tag(text)
    all_pos_tags.append(pos_tags)
for text in test_text:
    pos_tags = pos_tagger.tag(text)
    test_pos_tags.append(pos_tags)
f = open('pos_tags.pkl','wb')
pickle.dump(all_pos_tags,f)
f=open('pos_tags_test.pkl','wb')
pickle.dump(test_pos_tags,f)
