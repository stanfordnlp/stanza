from bisect import bisect_right
from copy import copy
import json
import numpy as np
import random
import re
import torch

from .vocab import Vocab


class DataLoader:
    def __init__(self, args, input_files={'json': None, 'txt': None, 'label': None}, input_text=None, input_data=None, vocab=None, evaluation=False):
        self.args = args
        self.eval = evaluation

        # get input files
        json_file = input_files['json']
        txt_file = input_files['txt']
        label_file = input_files['label']

        # Load data and process it
        if input_data is not None:
            self.data = input_data
        elif json_file is not None:
            with open(json_file) as f:
                self.data = json.load(f)
        else:
            # set up text from file or input string
            assert txt_file is not None or input_text is not None
            if input_text is None:
                with open(txt_file) as f:
                    text = ''.join(f.readlines()).rstrip()
            else:
                text = input_text

            if label_file is not None:
                with open(label_file) as f:
                    labels = ''.join(f.readlines()).rstrip()
            else:
                labels = '\n\n'.join(['0' * len(pt.rstrip()) for pt in re.split('\n\s*\n', text)])

            self.data = [list(zip(re.sub('\s', ' ', pt.rstrip()), [int(x) for x in pc])) for pt, pc in zip(re.split('\n\s*\n', text), labels.split('\n\n')) if len(pt.rstrip()) > 0]

        self.vocab = vocab if vocab is not None else self.init_vocab()

        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels
        self.sentences = [self.para_to_sentences(para) for para in self.data]

        self.init_sent_ids()

    def init_vocab(self):
        vocab = Vocab(self.data, self.args['lang'])
        return vocab

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j])]

    def para_to_sentences(self, para):
        res = []
        funcs = []
        for feat_func in self.args['feat_funcs']:
            if feat_func == 'space_before':
                func = lambda x: 1 if x.startswith(' ') else 0
            elif feat_func == 'capitalized':
                func = lambda x: 1 if x[0].isupper() else 0
            elif feat_func == 'all_caps':
                func = lambda x: 1 if x.isupper() else 0
            elif feat_func == 'numeric':
                func = lambda x: 1 if (re.match('^([\d]+[,\.]*)+$', x) is not None) else 0
            else:
                assert False, 'Feature function "{}" is undefined.'.format(feat_func)

            funcs += [func]

        composite_func = lambda x: list(map(lambda f: f(x), funcs))

        def process_and_featurize(sent):
            return [(self.vocab.unit2id(y[0]), y[1], composite_func(y[0]), y[0]) for y in sent]

        current = []
        for unit, label in para:
            label1 = label if not self.eval else 0
            current += [[unit, label]]
            if label1 == 2 or label1 == 4: # end of sentence
                if len(current) <= self.args['max_seqlen']:
                    # get rid of sentences that are too long during training of the tokenizer
                    res += [process_and_featurize(current)]
                current = []

        if len(current) > 0:
            if self.eval or len(current) <= self.args['max_seqlen']:
                res += [process_and_featurize(current)]

        return res

    def __len__(self):
        return len(self.sentence_ids)

    def shuffle(self):
        for para in self.sentences:
            random.shuffle(para)
        self.init_sent_ids()

    def next(self, eval_offsets=None, unit_dropout=0.0):
        null_feats = [0] * len(self.sentences[0][0][0][2])

        def strings_starting(id_pair, offset=0, pad_len=self.args['max_seqlen']):
            pid, sid = id_pair
            res = copy(self.sentences[pid][sid][offset:])

            assert self.eval or len(res) <= self.args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(self.args['max_seqlen'], len(res), ' '.join(["{}/{}".format(*x) for x in self.sentences[pid][sid]]))
            for sid1 in range(sid+1, len(self.sentences[pid])):
                res += self.sentences[pid][sid1]

                if not self.eval and len(res) >= self.args['max_seqlen']:
                    res = res[:self.args['max_seqlen']]
                    break

            if unit_dropout > 0 and not self.eval:
                unkid = self.vocab.unit2id('<UNK>')
                res = [(unkid, x[1], x[2], '<UNK>') if random.random() < unit_dropout else x for x in res]

            # pad with padding units and labels if necessary
            if pad_len > 0 and len(res) < pad_len:
                padid = self.vocab.unit2id('<PAD>')
                res += [(padid, -1, null_feats, '<PAD>')] * (pad_len - len(res))

            return res

        if eval_offsets is not None:
            # find max padding length
            pad_len = 0
            for eval_offset in eval_offsets:
                if eval_offset < self.cumlen[-1]:
                    pair_id = bisect_right(self.cumlen, eval_offset) - 1
                    pair = self.sentence_ids[pair_id]
                    pad_len = max(pad_len, len(strings_starting(pair, offset=eval_offset-self.cumlen[pair_id], pad_len=0)))

            res = []
            pad_len += 1
            for eval_offset in eval_offsets:
                # find unit
                if eval_offset >= self.cumlen[-1]:
                    padid = self.vocab.unit2id('<PAD>')
                    res += [[(padid, -1, null_feats, '<PAD>')] * pad_len]
                    continue

                pair_id = bisect_right(self.cumlen, eval_offset) - 1
                pair = self.sentence_ids[pair_id]
                res += [strings_starting(pair, offset=eval_offset-self.cumlen[pair_id], pad_len=pad_len)]
        else:
            id_pairs = random.sample(self.sentence_ids, min(len(self.sentence_ids), self.args['batch_size']))
            res = [strings_starting(pair) for pair in id_pairs]

        units = [[y[0] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]
        features = [[y[2] for y in x] for x in res]
        raw_units = [[y[3] for y in x] for x in res]

        convert = lambda t: (torch.from_numpy(np.array(t[0], dtype=t[1])))

        units, labels, features = list(map(convert, [(units, np.int64), (labels, np.int64), (features, np.float32)]))

        return units, labels, features, raw_units

