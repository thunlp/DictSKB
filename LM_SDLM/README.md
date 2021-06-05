# Running instructions on Language Model
## Requirements
- Pytorch == 0.3.1 

## Experiments
For Tied lstm:
```
python3 run_tied_lstm.py --emsize 1500 --nhid 1500 --dropout 0.7 --data ./data/wikitext-2  --save TL.pt  --lr 20 --epoch 80 --cuda
```

For Awd lstm:

```
python3 run_awd_lstm.py --batch_size 15 --data ./data/wikitext-2  --dropouti 0.5 --dropouth 0.2    --epoch 100 --save AWD.pt  --cuda
```

You could also evaluate the models using PTB datasets by changing the Command-line arguments.




