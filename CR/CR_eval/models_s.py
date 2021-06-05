import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import init

class SememeSumLstm(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.reset_parameters()
    def node_forward(self, inputs):
        iou = self.ioux(inputs)# three Wx+b
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = torch.mul(i, u)
        h = torch.mul(o, torch.tanh(c))
        return c, h
    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        c = []
        h = []
        for time in range(max_time):
            new_c, new_h = self.node_forward(inputs[time])
            c.append(new_c)
            h.append(new_h)
        return torch.stack(c, 0), torch.stack(h, 0)

    def reset_parameters(self):
        layers = [self.ioux]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

class SememeSumGRU(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumGRU, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.reset_parameters()
    def node_forward(self, inputs):
        iou = self.ioux(inputs)# three Wx+b
        i, o = torch.split(iou, iou.size(1) // 2, dim=1)
        i, o = torch.sigmoid(i), torch.tanh(o)

        h = torch.mul(i,o)
        return h
    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        h = []
        for time in range(max_time):
            new_h = self.node_forward(inputs[time])
            h.append(new_h)
        return torch.stack(h, 0)

    def reset_parameters(self):
        layers = [self.ioux]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

class LSTM_baseline(nn.Module):
    def __init__(self, config, sememe):
        super(LSTM_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.emb_sememe = nn.Embedding(2186, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fh, self.fs]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)
    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class LSTM_concat(nn.Module):
    def __init__(self, config, sememe):
        super(LSTM_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.fh, self.fs]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, word_emb, length, sememe_data):
        emb_s_1 = self.sememe_sum(sememe_data)
        inputs = torch.cat([word_emb, emb_s_1], dim = 2)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class LSTM_gate(nn.Module):
    def __init__(self, config, sememe):
        super(LSTM_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        #self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fh, self.W_c]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx(inputs) + self.fh(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return (c, h)

    def forward(self, inputs, length, sememe_data):
        sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe
    
    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
class LSTM_cell_bert_baseline(nn.Module):
    def __init__(self, config, ):
        super(LSTM_cell_bert_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']

        self.sememesumlstm = SememeSumLstm(512, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(512, self.enc_lstm_dim)

        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        # 乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        # ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        # fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False
        self.dic_lemma = self.read_lemmatization('../../NLI/dataset/lemmatization.txt')
        self.sense_tensor_dict = np.load('../../PrepareSememeDict/sense_tensor_dict.npy', allow_pickle=True).item()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fh, self.fs, self.fx_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_c, sememe_h, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
            self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c)  # part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c)  # part of memory cell induced by sememe-child
        c = torch.mul(i, u) + fc + fc_s  # sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, length, def_vec):
        sememe_c, sememe_h = self.sememe_sum(def_vec)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, sememe_c.size()[2]).fill_(0.).requires_grad_(),
              inputs[0][0].detach().new(batch_size, sememe_h.size()[2]).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_c[time], sememe_h[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack([output[length[i] - 1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):

        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(input_s[i].float())
        input_sememe = torch.stack(input_sememe, dim=0)
        sememe_c, sememe_h = self.sememesumlstm(input_sememe)
        return sememe_c, sememe_h

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)' % (len(self.word_vec), len(new_word_vec)))

    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        # return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort


    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma

    def get_def_vec_by_word(self, word):
        word_lower = word.lower()
        if word_lower in self.dic_lemma.keys():
            word_lower = self.dic_lemma[word_lower]
        if word_lower in self.sense_tensor_dict.keys():
            tensor_list = self.sense_tensor_dict[word_lower]
            base_tensor = np.zeros(512)
            for pos, tensor in tensor_list:
                base_tensor = np.add(base_tensor, tensor)
            base_tensor = base_tensor / float(len(tensor_list))
            return base_tensor
        else:
            return np.zeros(512)

    def get_batch(self, batch, emb_dim=300, ):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        # sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype=np.uint8)
        def_data = np.zeros((len(batch[0]), len(batch), 512), dtype=np.float)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                def_data[j, i] = self.get_def_vec_by_word(batch[i][j])

        return torch.from_numpy(embed).float(), torch.from_numpy(def_data).cuda()



    def encode(self, sentences, bsize=64, tokenize=True, verbose=False,):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], 300, )
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                len(embeddings) / (time.time() - tic),
                'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings


class LSTM_cell(nn.Module):
    def __init__(self, config, sememe):
        super(LSTM_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fh, self.fs, self.fx_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_c, sememe_h, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
            self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c) # part of memory cell induced by sememe-child
        c = torch.mul(i, u) + fc + fc_s #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, length, sememe_data):
        sememe_c, sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, sememe_c.size()[2]).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, sememe_h.size()[2]).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_c[time], sememe_h[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_c, sememe_h = self.sememesumlstm(input_sememe)
        return sememe_c, sememe_h
    
    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300, size=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False, size=300):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe,300, size)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class LSTM_extra_void(nn.Module):
    def __init__(self, config):
        super(LSTM_extra_void, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.W_s = nn.Linear(config['sememe_size'], self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.fh, self.W]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, length, sememe_data):
        emb_s = sememe_data.float().cuda()
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)

            emb_s_sum = emb_s[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

class LSTM_extra_concat(nn.Module):
    def __init__(self, config):
        super(LSTM_extra_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.fh, self.W]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, emb_s_concat):
        child_c = hx[0]
        child_h = hx[1]
        inputs = torch.cat([inputs, emb_s_concat], dim = 1)
        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, length, sememe_data):
        emb_s = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx, emb_s[time])
            output.append(next_hx[1])
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)
            emb_s_sum = emb_s[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class LSTM_extra_gate(nn.Module):
    def __init__(self, config):
        super(LSTM_extra_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        self.ious = nn.Linear(self.in_dim, 4 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.fc_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fh, self.W, self.fx_s, self.fh_s, self.fc_s, self.fs]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, emb_s):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(emb_s)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx(inputs) + self.fh(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c(emb_s)))

        return (c, h)

    def forward(self, inputs, length, sememe_data):
        emb_s = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx, emb_s[time])
            output.append(next_hx[1])
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)
            emb_s_sum = emb_s[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class LSTM_extra_cell(nn.Module):
    def __init__(self, config):
        super(LSTM_extra_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h

        self.W_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fs, self.fh, self.W]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_c, sememe_h, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
            self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c) # part of memory cell induced by sememe-child
        c = torch.mul(i, u) + fc + fc_s #sum means sigma
        h = torch.mul(o, torch.tanh(c))

        return (c, h)

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_c, sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_c[time], sememe_h[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)

            emb_s_sum = sememe_h[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_c, sememe_h = self.sememesumlstm(input_sememe)
        return sememe_c, sememe_h

class BILSTM_baseline(nn.Module):
    def __init__(self, config, sememe):
        super(BILSTM_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s, self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, sent, sent_len, sememe_data):
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward)
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward)
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class BILSTM_concat(nn.Module):
    def __init__(self, config, sememe):
        super(BILSTM_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.fx, self.fx_b, self.fh, self.fh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
 

class BILSTM_gate(nn.Module):
    def __init__(self, config, sememe):
        super(BILSTM_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.ious_b = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c_b = nn.Linear(self.in_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s, self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b, self.W_c, self.W_c_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx(inputs) + self.fh(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc#sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx_b(inputs) + self.fh_b(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c_b(sememe_h)))
        return (c, h)

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float().cuda(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
 
class BILSTM_cell(nn.Module):
    def __init__(self, config, sememe):
        super(BILSTM_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s, self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_c, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
                self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c)
        c = torch.mul(i, u) + fc + fc_s#sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_c, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        f_s_b = torch.sigmoid(
            self.fs_b(sememe_h) + self.fx_s_b(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s_b, sememe_c)
        c = torch.mul(i, u) + fc + fc_s #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)
    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sememe_c, sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        sememe_c = sememe_c.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_c[time, 0:pack_length[time]], sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_c[max_time-time-1, 0:pack_length[max_time-time-1]], sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_c, sememe_h = self.sememesumlstm(input_sememe)
        return sememe_c, sememe_h
    
    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300, size=300):
        print(size)
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False, size=3000):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe, 300, size)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings


class BILSTM_cell_bert_baseline(nn.Module):
    def __init__(self, config, ):
        super(BILSTM_cell_bert_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        # self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(512, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(512, self.enc_lstm_dim)
        # self.sememe_dim = config['sememe_dim']
        # self.sememe_size = config['sememe_size']
        # self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        # self.pool_type = config['pool_type']
        # 乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        # ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        # fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False
        self.dic_lemma = self.read_lemmatization('../../NLI/dataset/lemmatization.txt')
        self.sense_tensor_dict = np.load('../../PrepareSememeDict/sense_tensor_dict.npy', allow_pickle=True).item()

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s,
                  self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_c, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
            self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c)  # part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c)
        c = torch.mul(i, u) + fc + fc_s  # sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_c, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        f_s_b = torch.sigmoid(
            self.fs_b(sememe_h) + self.fx_s_b(inputs)
        )
        fc = torch.mul(f, child_c)  # part of memory cell induced by word-child
        fc_s = torch.mul(f_s_b, sememe_c)
        c = torch.mul(i, u) + fc + fc_s  # sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, sent, sent_len, def_vecs):
        # hx: (child_c, child_h)
        sememe_c, sememe_h = self.sememe_sum(def_vecs)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        sememe_c = sememe_c.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype=np.int)
        time_point = batch_size - 1
        last_point = 0
        while (True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point + 1
            last_point = sent_len_sorted[time_point]
            if (sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point - 1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
                      sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward,
                                        sememe_c[time, 0:pack_length[time]], sememe_h[time, 0:pack_length[time]])
            output_forward.append(
                torch.cat([next_hx[1], torch.zeros([batch_size - next_hx[1].size()[0], self.mem_dim], device='cuda')],
                          dim=0))
            if (time < max_time - 1):
                hx_forward = (next_hx[0][0:pack_length[time + 1]], next_hx[1][0:pack_length[time + 1]])

        output_backward = [[] for i in range(max_time)]
        hx_backward = (sent[0][0].detach().new(pack_length[max_time - 1], self.mem_dim).fill_(0.).requires_grad_(),
                       sent[0][0].detach().new(pack_length[max_time - 1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time - time - 1, 0:pack_length[max_time - time - 1]], hx_backward,
                                         sememe_c[max_time - time - 1, 0:pack_length[max_time - time - 1]],
                                         sememe_h[max_time - time - 1, 0:pack_length[max_time - time - 1]])
            output_backward[max_time - time - 1] = torch.cat(
                [next_hx[1], torch.zeros([batch_size - next_hx[1].size()[0], self.mem_dim], device='cuda')], dim=0)
            if (time < max_time - 1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros(
                    [pack_length[max_time - time - 2] - next_hx[0].size()[0], self.mem_dim]).cuda()], dim=0), \
                               torch.cat([next_hx[1], torch.zeros(
                                   [pack_length[max_time - time - 2] - next_hx[1].size()[0], self.mem_dim]).cuda()],
                                         dim=0))
        a = torch.stack(output_forward, dim=0)
        b = torch.stack(output_backward, dim=0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i] - 1][i] for i in range(batch_size)], dim=0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim=1)
        return final_output

    def sememe_sum(self, input_s):

        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(input_s[i].float())
        input_sememe = torch.stack(input_sememe, dim=0)
        sememe_c, sememe_h = self.sememesumlstm(input_sememe)
        return sememe_c, sememe_h

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)' % (len(self.word_vec), len(new_word_vec)))

    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        # return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma

    def get_def_vec_by_word(self, word):
        word_lower = word.lower()
        if word_lower in self.dic_lemma.keys():
            word_lower = self.dic_lemma[word_lower]
        if word_lower in self.sense_tensor_dict.keys():
            tensor_list = self.sense_tensor_dict[word_lower]
            base_tensor = np.zeros(512)
            for pos, tensor in tensor_list:
                base_tensor = np.add(base_tensor, tensor)
            base_tensor = base_tensor / float(len(tensor_list))
            return base_tensor
        else:
            return np.zeros(512)

    def get_batch(self, batch, emb_dim=300, ):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        # sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype=np.uint8)
        def_data = np.zeros((len(batch[0]), len(batch), 512), dtype=np.float)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                def_data[j, i] = self.get_def_vec_by_word(batch[i][j])

        return torch.from_numpy(embed).float(), torch.from_numpy(def_data).cuda()



    def encode(self, sentences, bsize=64, tokenize=True, verbose=False, size=3000):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], 300)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                len(embeddings) / (time.time() - tic),
                'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class BILSTM_extra_void(nn.Module):
    def __init__(self, config):
        super(BILSTM_extra_void, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.W_s = nn.Linear(config['sememe_size'], self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(config['sememe_size'], self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s, self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        emb_s = sememe_data.float().cuda()
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward)
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward)
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)

        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = emb_s[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = emb_s[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)
        return final_output

class BILSTM_extra_concat(nn.Module):
    def __init__(self, config):
        super(BILSTM_extra_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.fx, self.fx_b, self.fh, self.fh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)

        sememe_h = sememe_h.index_select(1, idx_unsort)
        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)

        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class BILSTM_extra_gate(nn.Module):
    def __init__(self, config):
        super(BILSTM_extra_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.ious_b = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s, self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b, self.W_c, self.W_c_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx(inputs) + self.fh(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc#sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx_b(inputs) + self.fh_b(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c_b(sememe_h)))
        return (c, h)

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        sememe_h = sememe_h.index_select(1, idx_unsort)
        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)

        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float().cuda(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class BILSTM_extra_cell(nn.Module):
    def __init__(self, config):
        super(BILSTM_extra_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh_b = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.W_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.ioux_b, self.iouh, self.iouh_b, self.ious, self.ious_b, self.fx, self.fx_b, self.fx_s, self.fx_s_b, self.fh, self.fh_b, self.fs, self.fs_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_c, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
                self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c)
        c = torch.mul(i, u) + fc + fc_s#sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def node_backward(self, inputs, hx, sememe_c, sememe_h):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh_b(child_h) + self.fx_b(inputs)
        )
        f_s_b = torch.sigmoid(
            self.fs_b(sememe_h) + self.fx_s_b(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s_b, sememe_c)
        c = torch.mul(i, u) + fc + fc_s #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)
    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sememe_c, sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        sememe_c = sememe_c.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = (inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(),
              inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_c[time, 0:pack_length[time]], sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = (next_hx[0][0:pack_length[time+1]], next_hx[1][0:pack_length[time+1]])


        output_backward = [[] for i in range(max_time)]
        hx_backward = (inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_(),
                  inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_())
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_c[max_time-time-1, 0:pack_length[max_time-time-1]], sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx[1], torch.zeros([batch_size-next_hx[1].size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = (torch.cat([next_hx[0], torch.zeros([pack_length[max_time-time-2]-next_hx[0].size()[0], self.mem_dim]).cuda()], dim = 0), \
                                torch.cat([next_hx[1], torch.zeros([pack_length[max_time-time-2]-next_hx[1].size()[0], self.mem_dim]).cuda()], dim = 0))
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        sememe_h = sememe_h.index_select(1, idx_unsort)
        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_c, sememe_h = self.sememesumlstm(input_sememe)
        return sememe_c, sememe_h

class GRU_baseline(nn.Module):
    def __init__(self, config, sememe):
        super(GRU_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)

        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class GRU_concat(nn.Module):
    def __init__(self, config, sememe):
        super(GRU_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)

        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, word_emb, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        inputs = torch.cat([word_emb, sememe_h], dim = 2)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class GRU_gate(nn.Module):
    def __init__(self, config, sememe):
        super(GRU_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fh_s, self.fs, self.Uh, self.W_c]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        o_c = self.fx_s(inputs) + self.fh_s(child_h) + self.fs(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
class GRU_cell_bert_baseline(nn.Module):
    def __init__(self, config):
        super(GRU_cell_bert_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememesumlstm = SememeSumLstm(512, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(512, self.enc_lstm_dim)

        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        # 乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.reset_parameters()
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False
        self.dic_lemma = self.read_lemmatization('../../NLI/dataset/lemmatization.txt')
        self.sense_tensor_dict = np.load('../../PrepareSememeDict/sense_tensor_dict.npy', allow_pickle=True).item()


    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1 - z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, length, def_vec):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(def_vec)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, sememe_h.size()[2]).fill_(0.).requires_grad_()

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack([output[length[i] - 1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):

        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(input_s[i].float())
        input_sememe = torch.stack(input_sememe, dim=0)
        sememe_h = self.sememesumGRU(input_sememe)
        return sememe_h

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)' % (len(self.word_vec), len(new_word_vec)))

    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        # return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort
    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma

    def get_def_vec_by_word(self, word):
        word_lower = word.lower()
        if word_lower in self.dic_lemma.keys():
            word_lower = self.dic_lemma[word_lower]
        if word_lower in self.sense_tensor_dict.keys():
            tensor_list = self.sense_tensor_dict[word_lower]
            base_tensor = np.zeros(512)
            for pos, tensor in tensor_list:
                base_tensor = np.add(base_tensor, tensor)
            base_tensor = base_tensor / float(len(tensor_list))
            return base_tensor
        else:
            return np.zeros(512)

    def get_batch(self, batch, emb_dim=300, ):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        # sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype=np.uint8)
        def_data = np.zeros((len(batch[0]), len(batch), 512), dtype=np.float)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                def_data[j, i] = self.get_def_vec_by_word(batch[i][j])

        return torch.from_numpy(embed).float(), torch.from_numpy(def_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False,):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], 300,)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                len(embeddings) / (time.time() - tic),
                'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class GRU_cell(nn.Module):
    def __init__(self, config, sememe):
        super(GRU_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, sememe_h.size()[2]).fill_(0.).requires_grad_()

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack([output[length[i]-1][i] for i in range(len(length))], 0)

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_h = self.sememesumGRU(input_sememe)
        return sememe_h
    
    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300, size=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False, size=300):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe, 300,size)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class GRU_extra_void(nn.Module):
    def __init__(self, config):
        super(GRU_extra_void, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s = nn.Linear(config['sememe_size'], self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fs, self.fh, self.W, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = sememe_data.float().cuda()
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx)
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)

            emb_s_sum = sememe_h[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

class GRU_extra_concat(nn.Module):
    def __init__(self, config):
        super(GRU_extra_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fs, self.fh, self.W, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx
        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx, sememe_h[time])
            output.append(next_hx)
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)

            emb_s_sum = sememe_h[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class GRU_extra_gate(nn.Module):
    def __init__(self, config):
        super(GRU_extra_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.in_dim, 2 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.fh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fs, self.fh, self.W, self.Uh, self.Uh_s, self.fh_s, self.W_c]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        o_c = self.fx_s(inputs) + self.fh_s(child_h) + self.fs(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)

            emb_s_sum = sememe_h[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class GRU_extra_cell(nn.Module):
    def __init__(self, config):
        super(GRU_extra_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']

        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        #fs是专门处理sememe传过来的c和h
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()
    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fs, self.fh, self.W, self.Uh]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, length, sememe_data):
        # hx: (child_c, child_h)
        sememe_h = self.sememe_sum(sememe_data)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx
        new_output = []
        new_output_2 = []
        for i in range(len(length)):
            hidden_old = torch.stack(output[0:length[i]], dim = 0)[:, i, :]
            new_output_2.append(torch.index_select(output[length[i]-1], 0, torch.tensor(i, device = 'cuda')))
            hidden = self.W(hidden_old)

            emb_s_sum = sememe_h[0:length[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output.append(torch.mm(att.transpose(1,0), hidden_old))
        new_output = self.W_p(torch.squeeze(torch.stack(new_output, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2, dim = 0)))
        new_output = torch.tanh(new_output)
        return new_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_h = self.sememesumGRU(input_sememe)
        return sememe_h

class BIGRU_baseline(nn.Module):
    def __init__(self, config, sememe):
        super(BIGRU_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)


        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s,self.ioux_b, self.iouh_b, self.ious_b, self.fx_b, self.Uh_b, self.Uh_s_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward)
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward)
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)

        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class BIGRU_concat(nn.Module):
    def __init__(self, config, sememe):
        super(BIGRU_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(2 * self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.Uh, self.ioux_b, self.iouh_b, self.fx_b, self.Uh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx

        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx, sememe_h):
        child_h = hx

        inputs = torch.cat([inputs, sememe_h], dim = 1)
        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, sememe_data):
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class BIGRU_gate(nn.Module):
    def __init__(self, config, sememe):
        super(BIGRU_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s_b = nn.Linear(self.mem_dim, self.mem_dim)


        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c_b = nn.Linear(self.in_dim, self.mem_dim)
        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.ioux_b, self.iouh_b, self.ious_b, self.fx_b, self.Uh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        o_c = self.fx_s(inputs) + self.fh_s(child_h) + self.fs(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return h

    def node_backward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        o_c = self.fx_s_b(inputs) + self.fh_s_b(child_h) + self.fs_b(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c_b(sememe_h)))
        return h

    def forward(self, sent, sent_len, sememe_data):
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), 2186), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
class BIGRU_cell_bert_baseline(nn.Module):
    def __init__(self, config,):
        super(BIGRU_cell_bert_baseline, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']

        self.sememesumlstm = SememeSumLstm(512, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(512, self.enc_lstm_dim)

        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        # self.pool_type = config['pool_type']
        # 乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.max_pad = True
        self.reset_parameters()

        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False
        self.dic_lemma = self.read_lemmatization('../../NLI/dataset/lemmatization.txt')
        self.sense_tensor_dict = np.load('../../PrepareSememeDict/sense_tensor_dict.npy', allow_pickle=True).item()


    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.ioux_b, self.iouh_b, self.ious_b, self.fx_b,
                  self.Uh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1 - z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1 - z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, def_vec):
        sememe_h = self.sememe_sum(def_vec)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype=np.int)
        time_point = batch_size - 1
        last_point = 0
        while (True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point + 1
            last_point = sent_len_sorted[time_point]
            if (sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point - 1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward,
                                        sememe_h[time, 0:pack_length[time]])
            output_forward.append(
                torch.cat([next_hx, torch.zeros([batch_size - next_hx.size()[0], self.mem_dim], device='cuda')], dim=0))
            if (time < max_time - 1):
                hx_forward = next_hx[0:pack_length[time + 1]]

        output_backward = [[] for i in range(max_time)]
        hx_backward = sent[0][0].detach().new(pack_length[max_time - 1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time - time - 1, 0:pack_length[max_time - time - 1]], hx_backward,
                                         sememe_h[max_time - time - 1, 0:pack_length[max_time - time - 1]])
            output_backward[max_time - time - 1] = torch.cat(
                [next_hx, torch.zeros([batch_size - next_hx.size()[0], self.mem_dim], device='cuda')], dim=0)
            if (time < max_time - 1):
                hx_backward = torch.cat(
                    [next_hx, torch.zeros([pack_length[max_time - time - 2] - next_hx.size()[0], self.mem_dim]).cuda()],
                    dim=0)
        a = torch.stack(output_forward, dim=0)
        b = torch.stack(output_backward, dim=0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i] - 1][i] for i in range(batch_size)], dim=0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim=1)
        return final_output

    def sememe_sum(self, input_s):
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(input_s[i].float())
        input_sememe = torch.stack(input_sememe, dim=0)
        sememe_h = self.sememesumGRU(input_sememe)
        return sememe_h

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)' % (len(self.word_vec), len(new_word_vec)))

    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        # return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def read_lemmatization(self, lemma_dir):
        dic_lemma = {}
        for line in open(lemma_dir):
            line = line.strip().split()
            dic_lemma[line[1]] = line[0]
        return dic_lemma

    def get_def_vec_by_word(self, word):
        word_lower = word.lower()
        if word_lower in self.dic_lemma.keys():
            word_lower = self.dic_lemma[word_lower]
        if word_lower in self.sense_tensor_dict.keys():
            tensor_list = self.sense_tensor_dict[word_lower]
            base_tensor = np.zeros(512)
            for pos, tensor in tensor_list:
                base_tensor = np.add(base_tensor, tensor)
            base_tensor = base_tensor / float(len(tensor_list))
            return base_tensor
        else:
            return np.zeros(512)

    def get_batch(self, batch, emb_dim=300, ):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        # sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype=np.uint8)
        def_data = np.zeros((len(batch[0]), len(batch), 512), dtype=np.float)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                def_data[j, i] = self.get_def_vec_by_word(batch[i][j])

        return torch.from_numpy(embed).float(), torch.from_numpy(def_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False, ):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], 300, )
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                len(embeddings) / (time.time() - tic),
                'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class BIGRU_cell(nn.Module):
    def __init__(self, config, sememe):
        super(BIGRU_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)


        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.max_pad = True
        self.reset_parameters()
        self.sememe = sememe
        self.bos = '<s>'
        self.eos = '</s>'
        self.max_pad = True
        self.moses_tok = False

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.ioux_b, self.iouh_b, self.ious_b, self.fx_b, self.Uh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h)+ torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, sememe_data):
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = sent[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = sent[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        final_output_forward = torch.stack([sent_output_forward[sent_len[i]-1][i] for i in range(batch_size)], dim = 0)
        final_output = torch.cat([final_output_forward, sent_output_backward[0]], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_h = self.sememesumGRU(input_sememe)
        return sememe_h

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))
    
    '''
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)
    '''

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return True
        #return self.enc_lstm.bias_hh_l0.data.is_cuda

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def get_batch(self, batch, sememe, emb_dim=300, size=300):
        embed = np.zeros((len(batch[0]), len(batch), 300))
        sememe_data = np.zeros((len(batch[0]), len(batch), size), dtype = np.uint8)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
                for k in sememe.read_word_sememe(batch[i][j]):
                    sememe_data[j, i, k] = 1
        return torch.from_numpy(embed).float(), torch.from_numpy(sememe_data).cuda()

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False, size=300):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_s = self.get_batch(sentences[stidx:stidx + bsize], self.sememe, 300, size=size)
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward(batch, lengths[stidx:stidx + bsize], batch_s).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

class BIGRU_extra_void(nn.Module):
    def __init__(self, config):
        super(BIGRU_extra_void, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)


        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.W_s = nn.Linear(config['sememe_size'], self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(config['sememe_size'], self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s,self.ioux_b, self.iouh_b, self.ious_b, self.fx_b, self.Uh_b, self.Uh_s_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, sememe_data):
        # hx: (child_c, child_h)
        emb_s = sememe_data.float().cuda()
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward)
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward)
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)

        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)

        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = emb_s[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = emb_s[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)
        return final_output

class BIGRU_extra_concat(nn.Module):
    def __init__(self, config):
        super(BIGRU_extra_concat, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)


        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(2 * self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.max_pad = True
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.Uh, self.Uh_s,self.ioux_b, self.iouh_b, self.fx_b, self.Uh_b, self.Uh_s_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, emb_s):
        child_h = hx
        inputs = torch.cat([inputs, emb_s], dim = 1)
        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx, emb_s):
        child_h = hx
        inputs = torch.cat([inputs, emb_s], dim = 1)
        iou = self.ioux_b(inputs) + self.iouh_b(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, sememe_data):
        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        emb_s = emb_s.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, emb_s[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, emb_s[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)

        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        emb_s = emb_s.index_select(1, idx_unsort)

        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = emb_s[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = emb_s[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class BIGRU_extra_gate(nn.Module):
    def __init__(self, config):
        super(BIGRU_extra_gate, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fs_b = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s_b = nn.Linear(self.mem_dim, self.mem_dim)


        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.W_c_b = nn.Linear(self.in_dim, self.mem_dim)
        self.max_pad = True
        self.W_s = nn.Linear(self.in_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(self.in_dim, self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.ioux_b, self.iouh_b, self.ious_b, self.fx_b, self.Uh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        o_c = self.fx_s(inputs) + self.fh_s(child_h) + self.fs(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return h

    def node_backward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        o_c = self.fx_s_b(inputs) + self.fh_s_b(child_h) + self.fs_b(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c_b(sememe_h)))
        return h

    def forward(self, sent, sent_len, sememe_data):

        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        sememe_h = sememe_h.index_select(1, idx_unsort)

        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        return input_sememe

class BIGRU_extra_cell(nn.Module):
    def __init__(self, config):
        super(BIGRU_extra_cell, self).__init__()
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.sememe_dim = config['sememe_dim']
        self.sememesumlstm = SememeSumLstm(self.sememe_dim, self.enc_lstm_dim)
        self.sememesumGRU = SememeSumGRU(self.sememe_dim, self.enc_lstm_dim)
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']
        self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)

        self.in_dim = config['word_emb_dim']
        self.mem_dim = config['enc_lstm_dim']
        #self.pool_type = config['pool_type']
        #乘3代表3种矩阵，它后来用split分开了
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.ioux_b = nn.Linear(self.in_dim, 2 * self.mem_dim)

        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.iouh_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious_b = nn.Linear(self.mem_dim, 2 * self.mem_dim)


        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_b = nn.Linear(self.in_dim, self.mem_dim)

        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_b = nn.Linear(self.mem_dim, self.mem_dim)

        self.max_pad = True
        self.W_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim, self.mem_dim)
        self.query = nn.Embedding(2*self.mem_dim, 1)
        self.W_p = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_s_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.query_b = nn.Embedding(2*self.mem_dim, 1)
        self.W_p_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_x_b = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.ioux_b, self.iouh_b, self.ious_b, self.fx_b, self.Uh_b]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def node_backward(self, inputs, hx, sememe_h):
        child_h = hx

        iou = self.ioux_b(inputs) + self.iouh_b(child_h) + self.ious_b(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx_b(inputs) + self.Uh_b(torch.mul(r, child_h) + torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, sent, sent_len, sememe_data):

        sememe_h = self.sememe_sum(sememe_data)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda()
        sent = sent.index_select(1, idx_sort)
        sememe_h = sememe_h.index_select(1, idx_sort)
        max_time, batch_size, _ = sent.size()
        pack_length = np.zeros([max_time, 1], dtype = np.int)
        time_point = batch_size-1
        last_point = 0
        while(True):
            pack_length[last_point: sent_len_sorted[time_point]] = time_point+1
            last_point = sent_len_sorted[time_point]
            if(sent_len_sorted[time_point] == max_time):
                break
            time_point = time_point-1
        pack_length = torch.from_numpy(pack_length).cuda()
        output_forward = []
        hx_forward = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_forward(sent[time, 0:pack_length[time]], hx_forward, sememe_h[time, 0:pack_length[time]])
            output_forward.append(torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0))
            if(time < max_time-1):
                hx_forward = next_hx[0:pack_length[time+1]]


        output_backward = [[] for i in range(max_time)]
        hx_backward = inputs[0][0].detach().new(pack_length[max_time-1], self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            next_hx = self.node_backward(sent[max_time-time-1, 0:pack_length[max_time-time-1]], hx_backward, sememe_h[max_time-time-1, 0:pack_length[max_time-time-1]])
            output_backward[max_time-time-1] = torch.cat([next_hx, torch.zeros([batch_size-next_hx.size()[0], self.mem_dim], device = 'cuda')], dim = 0)
            if(time < max_time-1):
                hx_backward = torch.cat([next_hx, torch.zeros([pack_length[max_time-time-2]-next_hx.size()[0], self.mem_dim]).cuda()], dim = 0)
        a = torch.stack(output_forward, dim = 0)
        b = torch.stack(output_backward, dim = 0)
        idx_unsort = torch.from_numpy(idx_unsort).cuda()
        sent_output_forward = a.index_select(1, idx_unsort)
        sent_output_backward = b.index_select(1, idx_unsort)
        sememe_h = sememe_h.index_select(1, idx_unsort)

        new_output_forward = []
        new_output_2_forward = []
        new_output_backward = []
        for i in range(len(sent_len)):
            hidden_old_forward = sent_output_forward[0:sent_len[i], i, :]
            new_output_2_forward.append(sent_output_forward[sent_len[i]-1, i])
            hidden = self.W(hidden_old_forward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query.weight))
            new_output_forward.append(torch.mm(att.transpose(1,0), hidden_old_forward))
        new_output_forward = self.W_p(torch.squeeze(torch.stack(new_output_forward, dim = 0))) + self.W_x(torch.squeeze(torch.stack(new_output_2_forward, dim = 0)))
        new_output_forward = torch.tanh(new_output_forward)

        for i in range(len(sent_len)):
            hidden_old_backward = sent_output_backward[0:sent_len[i], i, :]
            hidden = self.W_b(hidden_old_backward)

            emb_s_sum = sememe_h[0:sent_len[i], i, :]
            emb_s_sum = self.W_s_b(emb_s_sum)
            hidden = torch.cat([hidden, emb_s_sum], dim = 1)
            att = torch.tanh(torch.mm(hidden, self.query_b.weight))
            new_output_backward.append(torch.mm(att.transpose(1,0), hidden_old_backward))
        new_output_backward = self.W_p_b(torch.squeeze(torch.stack(new_output_backward, dim = 0))) + self.W_x_b(sent_output_backward[0])
        new_output_backward = torch.tanh(new_output_backward)

        final_output = torch.cat([new_output_forward, new_output_backward], dim = 1)
        return final_output

    def sememe_sum(self, input_s):
        emb_sememe = self.emb_sememe.weight
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)
        sememe_h = self.sememesumGRU(input_sememe)
        return sememe_h

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']
        self.sememe_dim = config['sememe_dim']
        self.sememe_size = config['sememe_size']

        #self.emb_sememe = nn.Embedding(self.sememe_size, self.sememe_dim)
        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4*self.enc_lstm_dim
        self.inputdim = int(self.inputdim)
        if self.nonlinear_fc:
            if 'BI' in self.encoder_type:
                self.classifier = nn.Sequential(
                    nn.Dropout(p=self.dpout_fc),
                    nn.Linear(self.inputdim * 2, self.fc_dim),
                    nn.Tanh(),
                    nn.Dropout(p=self.dpout_fc),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.Tanh(),
                    nn.Dropout(p=self.dpout_fc),
                    nn.Linear(self.fc_dim, self.n_classes),
                    )
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(p=self.dpout_fc),
                    nn.Linear(self.inputdim, self.fc_dim),
                    nn.Tanh(),
                    nn.Dropout(p=self.dpout_fc),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.Tanh(),
                    nn.Dropout(p=self.dpout_fc),
                    nn.Linear(self.fc_dim, self.n_classes),
                    )
        else:
            if 'BI' in self.encoder_type:
                self.classifier = nn.Sequential(
                    nn.Linear(self.inputdim * 2, self.fc_dim),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.Linear(self.fc_dim, self.n_classes)
                    )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(self.inputdim, self.fc_dim),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.Linear(self.fc_dim, self.n_classes)
                    )
                    
    def forward(self, s1, s2):
        u = self.encoder(s1[0], s1[1], s1[2])
        v = self.encoder(s2[0], s2[1], s2[2])

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output
