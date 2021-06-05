# Running instructions on Intrinsic Evaluation
## Requirements
- Pytorch == 0.3.1 


## Intrinsic Evaluation
You could run intrinsic evaluation by:
```
python ConsistentEvaluation.py --gpu_id 0 --eval_epochs 10 --eval_type dict  --threshold_dict 0.8   --descending_c_dict 0.93
```