from bisect import bisect_left
from collections import Counter
from copy import copy
import json
import pickle
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.postprocess_vietnamese_tokenizer_data import paras_to_chunks

class Tokenizer(nn.Module):
    def __init__(self, nchars, emb_dim, hidden_dim, N_CLASSES=4, dropout=0):
        super().__init__()

        feat_dim = args['feat_dim']

        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(emb_dim + feat_dim, hidden_dim, 5, padding=2)

        self.dense_clf = nn.Conv1d(hidden_dim, N_CLASSES, 1)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, feats):
        emb = self.embeddings(x)

        emb = torch.cat([emb, feats], 2)

        emb = emb.transpose(1, 2).contiguous()

        hid = F.relu(self.conv1(emb))
        hid = self.dropout(hid)

        pred = self.dense_clf(hid)
        pred = pred.transpose(1, 2).contiguous()

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

        self._id2unit = ['<PAD>', '<UNK>'] + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id['<UNK>']

    def id2unit(self, id):
        return self._id2unit[id]

    def normalize_unit(self, unit):
        # Normalize minimal units used by the tokenizer
        # For Vietnamese this means a syllable, for other languages this means a character
        normalized = unit
        if self.lang == 'vi':
            normalized = normalized.lstrip()

        return normalized

    def normalize_token(self, token):
        token = token.lstrip().replace('\n', ' ')

        if self.lang == 'zh':
            token = token.replace(' ', '')

        return token

    def __len__(self):
        return len(self._id2unit)

class TokenizerDataGenerator:
    def __init__(self, args, data):
        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels
        self.sentences = [self.para_to_sentences(para) for para in data]
        self.args = args

        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j])]

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

    def __len__(self):
        return len(self.sentence_ids)

    def next(self, vocab, feat_funcs=['space_before', 'capitalized'], eval_offset=-1):
        def strings_starting(id_pair, offset=0):
            pid, sid = id_pair
            res = copy(self.sentences[pid][sid][offset:])

            assert args['mode'] == 'predict' or len(res) <= args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(args['max_seqlen'], len(res), ' '.join(["{}/{}".format(*x) for x in self.sentences[pid][sid]]))
            for sid1 in range(sid+1, len(self.sentences[pid])):
                res += self.sentences[pid][sid1]

                if len(res) >= args['max_seqlen']:
                    res = res[:args['max_seqlen']]
                    break

            # pad with padding units and labels if necessary
            if len(res) < args['max_seqlen']:
                res += [('<PAD>', -1)] * (args['max_seqlen'] - len(res))

            return res

        if eval_offset >= 0:
            # find unit
            if eval_offset >= self.cumlen[-1]:
                return None
            pair_id = bisect_left(self.cumlen, eval_offset)
            pair = self.sentence_ids[pair_id]
            res = [strings_starting(pair, offset=eval_offset-self.cumlen[pair_id])]
        else:
            id_pairs = random.sample(self.sentence_ids, self.args['batch_size'])
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

        units = [[vocab.unit2id(y[0]) for y in x] for x in res]
        raw_units = [[y[0] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]

        convert = lambda t: Variable(torch.from_numpy(np.array(t[0], dtype=t[1])))

        units, labels, features = list(map(convert, [(units, np.int64), (labels, np.int64), (features, np.float32)]))

        return units, labels, features, raw_units

class TokenizerTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args['json_file'] is not None:
            with open(args['json_file']) as f:
                self.data = json.load(f)
        else:
            with open(args['txt_file']) as f:
                text = ''.join(f.readlines()).rstrip()

            if args['label_file'] is not None:
                with open(args['label_file']) as f:
                    labels = ''.join(f.readlines()).rstrip()
            else:
                labels = '\n\n'.join(['0' * len(pt) for pt in text.split('\n\n')])

            self.data = [list(zip(pt.rstrip(), [int(x) for x in pc])) for pt, pc in zip(text.split('\n\n'), labels.split('\n\n'))]

        self.data_generator = TokenizerDataGenerator(args, self.data)
        self.feat_funcs = args['feat_funcs']
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
            self._model = Tokenizer(len(self.vocab), args['emb_dim'], args['hidden_dim'], dropout=args['dropout'])

            if args['mode'] == 'train':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
                self.opt = optim.Adam(self._model.parameters(), lr=2e-3, betas=(.9, .9))

        return self._model

    def update(self, inputs):
        units, labels, features, _ = inputs

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred = self.model(units, features)

        self.opt.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args['max_grad_norm'])
        self.opt.step()

        return loss.data[0]

    def predict(self, inputs):
        units, labels, features, _ = inputs

        if self.model.embeddings.weight.is_cuda:
            units = units.cuda()
            labels = labels.cuda()
            features = features.cuda()

        pred = self.model(units, features)

        return pred.data.cpu().numpy()

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
        if self.args['mode'] == 'train':
            self.opt.load_state_dict(savedict['optim'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--lang', type=str, help="Language")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=200, help="Dimension of hidden units")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--no_cuda', dest="cuda", action="store_false")

    args = parser.parse_args()

    args = vars(args)
    args['feat_funcs'] = ['space_before', 'capitalized']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = args['save_name'] if args['save_name'] is not None else '{}_tokenizer.pkl'.format(args['lang'])
    trainer = TokenizerTrainer(args)

    if args['cuda']:
        trainer.cuda()

    N = len(trainer.data_generator)
    if args['mode'] == 'train':
        steps = int(N * args['epochs'] / args['batch_size'] + .5)

        for step in range(steps):
            batch = trainer.data_generator.next(trainer.vocab, feat_funcs=trainer.feat_funcs)

            loss = trainer.update(batch)
            if step % args['report_steps'] == 0:
                print("Step {:6d}/{:6d} Loss: {:.3f}".format(step, steps, loss))

        trainer.save(args['save_name'])
    else:
        trainer.load(args['save_name'])

        offset = 0
        with open(args['conll_file'], 'w') as f:
            while True:
                batch = trainer.data_generator.next(trainer.vocab, feat_funcs=trainer.feat_funcs, eval_offset=offset)
                if batch is None:
                    break
                pred = np.argmax(trainer.predict(batch)[0], axis=1)

                current_tok = ''
                current_sent = []

                for t, p in zip(batch[3][0], pred):
                    if t == '<PAD>':
                        break
                    offset += 1
                    current_tok += t
                    if p == 1 or p == 2:
                        current_sent += [trainer.vocab.normalize_token(current_tok)]
                        current_tok = ''
                        if p == 2:
                            for i, tok in enumerate(current_sent):
                                f.write("{}\t{}{}\t{}{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 3))
                            f.write('\n')

                            current_sent = []

                if len(current_tok):
                    current_sent += [trainer.vocab.normalize_token(current_tok)]

                if len(current_sent):
                    for i, tok in enumerate(current_sent):
                        f.write("{}\t{}{}\t{}{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 3))
                    f.write('\n')
