from bisect import bisect_right
from copy import copy
import json
import numpy as np
import random
import logging
import re
import torch

from .vocab import Vocab

logger = logging.getLogger('stanza')

def filter_consecutive_whitespaces(para):
    filtered = []
    for i, (char, label) in enumerate(para):
        if i > 0:
            if char == ' ' and para[i-1][0] == ' ':
                continue

        filtered.append((char, label))

    return filtered

NEWLINE_WHITESPACE_RE = re.compile(r'\n\s*\n')
NUMERIC_RE = re.compile(r'^([\d]+[,\.]*)+$')
WHITESPACE_RE = re.compile(r'\s')


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
                labels = '\n\n'.join(['0' * len(pt.rstrip()) for pt in NEWLINE_WHITESPACE_RE.split(text)])

            self.data = [[(WHITESPACE_RE.sub(' ', char), int(label)) # substitute special whitespaces
                    for char, label in zip(pt.rstrip(), pc) if not (args.get('skip_newline', False) and char == '\n')] # check if newline needs to be eaten
                    for pt, pc in zip(NEWLINE_WHITESPACE_RE.split(text), NEWLINE_WHITESPACE_RE.split(labels)) if len(pt.rstrip()) > 0]

        # remove consecutive whitespaces
        self.data = [filter_consecutive_whitespaces(x) for x in self.data]

        self.vocab = vocab if vocab is not None else self.init_vocab()

        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels
        self.sentences = [self.para_to_sentences(para) for para in self.data]

        self.init_sent_ids()
        logger.debug(f"{len(self.sentence_ids)} sentences loaded.")

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
            if feat_func == 'end_of_para' or feat_func == 'start_of_para':
                # skip for position-dependent features
                continue
            if feat_func == 'space_before':
                func = lambda x: 1 if x.startswith(' ') else 0
            elif feat_func == 'capitalized':
                func = lambda x: 1 if x[0].isupper() else 0
            elif feat_func == 'all_caps':
                func = lambda x: 1 if x.isupper() else 0
            elif feat_func == 'numeric':
                func = lambda x: 1 if (NUMERIC_RE.match(x) is not None) else 0
            else:
                raise Exception('Feature function "{}" is undefined.'.format(feat_func))

            funcs.append(func)

        # stacking all featurize functions
        composite_func = lambda x: [f(x) for f in funcs]

        def process_sentence(sent):
            return [(self.vocab.unit2id(y[0]), y[1], y[2], y[0]) for y in sent]

        use_end_of_para = 'end_of_para' in self.args['feat_funcs']
        use_start_of_para = 'start_of_para' in self.args['feat_funcs']
        current = []
        for i, (unit, label) in enumerate(para):
            label1 = label if not self.eval else 0
            feats = composite_func(unit)
            # position-dependent features
            if use_end_of_para:
                f = 1 if i == len(para)-1 else 0
                feats.append(f)
            if use_start_of_para:
                f = 1 if i == 0 else 0
                feats.append(f)
            current += [(unit, label, feats)]
            if label1 == 2 or label1 == 4: # end of sentence
                if len(current) <= self.args['max_seqlen']:
                    # get rid of sentences that are too long during training of the tokenizer
                    res.append(process_sentence(current))
                current = []

        if len(current) > 0:
            if self.eval or len(current) <= self.args['max_seqlen']:
                res.append(process_sentence(current))

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

