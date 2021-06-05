# Running instructions on Natural Language Inference
## Requirements
- Pytorch == 1.5.0

## Experiments

For NLI model, you could run by:
```
python3 main.py --encoder_type LSTM_cell --word_emb_path dataset/GloVe/glove.840B.300d.txt  --dpout_fc 0.3  --gpu_id 0 
```
Or you could change the encoder_type to test different models.

For example, you could test the GRU_cell model by:
```
python3 main.py --encoder_type GRU_cell --word_emb_path dataset/GloVe/glove.840B.300d.txt  --dpout_fc 0.25  --gpu_id 0 
```