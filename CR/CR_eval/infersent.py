# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
from sememe import Sememe
import argparse
parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--model", type=str, default='LSTM_cell', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputmodelname", type=str)
parser.add_argument("--sememe_size", type=int)
params, _ = parser.parse_known_args()


MODEL_PATH = []
MODEL_PATH.append('../../NLI/savedir/'+ params.outputmodelname)
# for dir in os.listdir('../../NLI/savedir'):
#     if params.model in dir and 'encoder' in dir:
#         if 'BI' not in params.model:
#             if 'BI' not in dir:
#                 MODEL_PATH.append('../../NLI/savedir/' + dir)
#         else:
#             if 'BI' in dir:
#                 MODEL_PATH.append('../../NLI/savedir/' + dir)
#MODEL_PATH = '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir/' + 'model.pickle_BIGRU_baseline_9791.encoder.pkl'

# get models.py from InferSent repo
from models import InferSent
from models_s import *

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = './data'
PATH_TO_W2V = '../../NLI/dataset/GloVe/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
#MODEL_PATH = 'encoder/infersent1.pkl'
#MODEL_PATH = '/data1/private/qinyujia/Sememe-enhanced-RNN-qin/savedir/model.pickle_BILSTM_baseline_2465.encoder.pkl'
V = 1 # version of InferSent

#assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), 'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

sememe_dir = '../../NLI/dataset/sememe.txt'
hownet_dir = '../../NLI/dataset/sememe_dict.txt'
lemma_dir =  '../../NLI/dataset/lemmatization.txt'
sememe = Sememe(hownet_dir = hownet_dir, sememe_dir = sememe_dir, lemma_dir = lemma_dir, filename = hownet_dir, lower = True)

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch, size):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False, size=size)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (CR)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
size = params.sememe_size
if __name__ == "__main__":
    # Load InferSent model

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    config_nli_model = {
    'word_emb_dim'   :  300   ,
    'enc_lstm_dim'   :  2048   ,
    'sememe_dim'     :  300     ,
    'sememe_size'    :  size
    }


    #model = InferSent(params_model)
    model = eval(params.model)(config_nli_model, sememe)
    #model = BILSTM_baseline(config_nli_model, sememe)
    for model_path in MODEL_PATH:
        model.load_state_dict(torch.load(model_path))
        model.set_w2v_path(PATH_TO_W2V)

        params_senteval['infersent'] = model.cuda()
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = ['CR']
        results = se.eval(transfer_tasks, size)
        print(results)
