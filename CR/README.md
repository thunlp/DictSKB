# Running instructions on Sentiment Analysis
## Requirements
- Pytorch == 1.5.0

## Experiments



For CR model, you could use the model trained on SNLI Dataset and run the pretrained models following instructions below.

- Enter target directory
```
cd CR_eval
```
- Run experiments(you may want to change the outputmodelname to match the pretrained model's name in the NLI experiment)
```
python infersent.py --model LSTM_cell --outputmodelname LSTM_cell.encoder.pkl --sememe_size 2047
```
You could also change the model type to test different models.

