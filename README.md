# DictSKB

This is the official repository of the code and data of the paper **Automatic Construction of Sememe Knowledge Bases via Dictionaries**, which is published on Findings of ACL: ACL-IJCNLP 2021 [[pdf](https://arxiv.org/pdf/2105.12400)].

## Evaluation Datasets and Tools

- Download [Glove vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip), which is used for NLI, CR and Textual Adversarial Attack experiments.

- Download [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml#Download), which is used for generating candidate words for Textual Adversarial Attack. However, you can directly use our preprocessed data.

## Running Instructions
Please see the `README.md` files in `ConsistencyCheck/`, `CR/`, `LM_SDLM/` and `NLI/` for specific running instructions for each model on corresponding intrinsic evaluation and downstream tasks.

## Citation

Please kindly cite our paper:

```
@article{qi2021automatic,
  title={Automatic Construction of Sememe Knowledge Bases via Dictionaries},
  author={Qi, Fanchao and Chen, Yangyi and Wang, Fengyu and Liu, Zhiyuan and Chen, Xiao and Sun, Maosong},
  journal={arXiv preprint arXiv:2105.12585},
  year={2021}
}
```
