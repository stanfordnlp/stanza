from collections import Counter
import json
import pickle
import numpy as np
import random
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks

class Tokenizer(nn.Module):
    def __init__(self, nchars, emb_dim, hidden_dim, N_CLASSES=4, dropout=0):
        super().__init__()

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(emb_dim, hidden_dim, 5, padding=2)

        self.dense_clf = nn.Conv1d(hidden_dim, N_CLASSES, 1)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, feats):
        emb = self.embeddings(x)

        emb = torch.cat([emb, feats], axis=2)

        emb = emb.transpose(1, 2).contiguous()

        hid = F.relu(self.conv1)
        hid = self.dropout(hid)

        pred = self.dense_clf(hid)
        pred = self.transpose(1, 2).contiguous()

        return pred

class Vocab:
    def __init__(self, paras, lang):
        self.lang = lang
        self.build_vocab(paras)

    def build_vocab(self, paras):
        counter = Counter()
        for para in paras:
            for unit in para:
                normalized = self.normalize_unit(unit[0])
                counter[normalized] += 1

        self.id2unit = ['<PAD>'] + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self.unit2id = {w:i for i, w in enumerate(self.id2unit)}

    def normalize_unit(self, unit):
        # Normalize minimal units used by the tokenizer
        # For Vietnamese this means a syllable, for other languages this means a character
        normalized = unit
        if self.lang == 'vi':
            normalized = normalized.lstrip()

        return normalized

    def __len__(self):
        return len(self.id2unit)

class TokenizerDataGenerator:
    def __init__(self, args, data):
        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels
        self.sentences = [self.para_to_sentences(para) for para in data]
        self.args = args

        self.sentence_ids = []
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]

    def para_to_sentences(self, para):
        res = []

        current = []
        for unit, label in para:
            current += [[unit, label]]
            if label == 2: # end of sentence
                res += [current]
                current = []

        if len(current) > 0:
            res += [current]

        return res

    def next(self, vocab, feat_funcs=['space_before', 'capitalized']):
        id_pairs = random.sample(self.sentence_ids, self.args['batch_size'])

        def strings_starting(id_pair):
            pid, sid = id_pair
            res = self.sentences[pid][sid]

            assert len(res) <= args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it!'.format(args['max_seqlen'], len(res))
            for sid1 in range(sid+1, len(self.sentences[pid])):
                res += self.sentences[pid][sid1]

                if len(res) >= args['max_seqlen']:
                    res = res[:args['max_seqlen']]
                    break

            # pad with padding units and labels if necessary
            if len(res) < args['max_seqlen']:
                res += [('<PAD>', -1)] * (args['max_seqlen'] - len(res))

            return res

        res = [strings_starting(pair) for pair in id_pairs]

        funcs = []
        for feat_func in feat_funcs:
            if feat_func == 'space_before':
                func = lambda x: x.startswith(' ')
            elif feat_func == 'capitalized':
                func = lambda x: x[0].isupper()
            elif feat_func == 'all_caps':
                func = lambda x: x.isupper()
            elif feat_func == 'numeric':
                func = lambda x: re.match('^[\d\s]+$', x)
            else:
                assert False, 'Feature function "{}" is undefined.'.format(feat_func)

            funcs += [func]

        composite_func = lambda x: list(map(lambda f: f(x), funcs))

        features = [[composite_func(y[0]) for y in x] for x in res]

        units = [[vocab.unit2id[vocab.normalize_unit(y[0])] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]

        convert = lambda x: Variable(torch.from_numpy(np.array(x, dtype=np.float32)))

        units, labels, features = list(map(convert, [units, labels, features]))

        return units, labels, features

class TokenizerTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args['json_file'] is not None:
            with open(args['json_file']) as f:
                self.data = json.load(f)
        else:
            with open(args['txt_file']) as f:
                text = ''.join(f.readlines()).rstrip()

            with open(args['label_file']) as f:
                labels = ''.join(f.readlines()).rstrip()

            self.data = [list(zip(pt, [int(x) for x in pc])) for pt, pc in zip(text.split('\n\n'), labels.split('\n\n'))]

        self.data_generator = TokenizerDataGenerator(args, self.data)
        self.args = args
        self.lang = args['lang'] # language determines how token normlization is done

    @property
    def vocab(self):
        # enable lazy construction in case we're just loading the vocab from file
        if not hasattr(self, '_vocab'):
            self._vocab = Vocab(self.data, self.lang)

        return self._vocab

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = Tokenizer(len(self.vocab), args['emb_dim'], args['hidden_dim'])

            if args['mode'] == 'train':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                self.opt = optim.Adam(self._model.parameters(), lr=2e-3, betas=(.9, .9))

        return self._model

    def update(self, inputs):
        pass

    def predict(self, inputs):
        pass

    def save(self, filename):
        savedict = {
                   'vocab': self.vocab,
                   'model': self.model.state_dict(),
                   'optim': self.opt.state_dict()
                   }
        with open(filename, 'wb') as f:
            pickle.dump(savedict, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            savedict = pickle.load(f)

        self._vocab = savedict['vocab']
        self.model.load_state_dict(savedict['model'])
        self.opt.load_state_dict(savedict['optim'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--lang', type=str, help="Language")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=200, help="Dimension of hidden units")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")

    args = parser.parse_args()

    args = vars(args)
    trainer = TokenizerTrainer(args)

    print(trainer.data_generator.next(trainer.vocab))
