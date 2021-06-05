# Running instructions on Textual Adversarial Attack
## Requirements
- tensorflow-gpu == 1.14.0
- keras == 2.2.4
- sklearn == 0.0
- anytree == 2.6.0
- nltk == 3.4.5
- pytorch_transformers == 1.0.0
- loguru == 0.3.2

## Directly Run the Experiment
We have preprocessed the data you need to run this experiment. So you can directly use our preprocessed data to attack models. However, we only provide pretrained BILSTM model because of requirement of file size. So, if you want to attack bert model, you may follow instructions in the next section to train the bert victim model.

To attack Bi-LSTM:
```
python AD_dpso_sem.py
```


## Process Data and Train Model
- Process SST-2 Data
```
python data_utils.py
```
- Generate Candidate Substitution Words
```
python gen_pos.py
python lemma.py
python gen_candidates.py
```
- Train BiLSTM Model
```
python train_model.py
```
- Train BERT Model
```
python SST_BERT.py
```
To attack Bi-LSTM:
```
python AD_dpso_sem.py
```
To attack BERT:
```
python AD_dpso_sem_bert.py
```