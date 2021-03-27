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
    def __init__(self, args, input_files={'txt': None, 'label': None}, input_text=None, input_data=None, vocab=None, evaluation=False):
        self.args = args
        self.eval = evaluation

        # get input files
        txt_file = input_files['txt']
        label_file = input_files['label']

        # Load data and process it
        if input_data is not None:
            self.data = input_data
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

            skip_newline = args.get('skip_newline', False)
            self.data = [[(WHITESPACE_RE.sub(' ', char), int(label)) # substitute special whitespaces
                          for char, label in zip(pt.rstrip(), pc) if not (skip_newline and char == '\n')] # check if newline needs to be eaten
                         for pt, pc in zip(NEWLINE_WHITESPACE_RE.split(text), NEWLINE_WHITESPACE_RE.split(labels)) if len(pt.rstrip()) > 0]

        # remove consecutive whitespaces
        self.data = [filter_consecutive_whitespaces(x) for x in self.data]

        self.vocab = vocab if vocab is not None else self.init_vocab()

        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels.
        # At evaluation time, each paragraph is treated as single "sentence" as we don't know a priori where
        # sentence breaks occur. We make prediction from left to right for each paragraph and move forward to
        # the last predicted sentence break to start afresh.
        self.sentences = [self.para_to_sentences(para) for para in self.data]

        self.init_sent_ids()
        logger.debug(f"{len(self.sentence_ids)} sentences loaded.")

    def has_mwt(self):
        # presumably this only needs to be called either 0 or 1 times,
        # 1 when training and 0 any other time, so no effort is put
        # into caching the result
        for sentence in self.data:
            for word in sentence:
                if word[1] > 2:
                    return True
        return False

    def init_vocab(self):
        vocab = Vocab(self.data, self.args['lang'])
        return vocab

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j][0])]

    def para_to_sentences(self, para):
        """ Convert a paragraph to a list of processed sentences. """
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
            return [self.vocab.unit2id(y[0]) for y in sent], [y[1] for y in sent], [y[2] for y in sent], [y[0] for y in sent]

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

    def next(self, eval_offsets=None, unit_dropout=0.0, old_batch=None):
        ''' Get a batch of converted and padded PyTorch data from preprocessed raw text for training/prediction. '''
        feat_size = len(self.sentences[0][0][2][0])
        unkid = self.vocab.unit2id('<UNK>')
        padid = self.vocab.unit2id('<PAD>')

        if old_batch is not None:
            # If we have previously built a batch of data and made predictions on them, then when we are trying to make
            # prediction on later characters in those paragraphs, we can avoid rebuilding the converted data from scratch
            # and just (essentially) advance the indices/offsets from where we read converted data in this old batch.
            # In this case, eval_offsets index within the old_batch to advance the strings to process.
            ounits, olabels, ofeatures, oraw = old_batch
            lens = (ounits != padid).sum(1).tolist()
            pad_len = max(l-i for i, l in zip(eval_offsets, lens))

            units = np.full((len(ounits), pad_len), padid, dtype=np.int64)
            labels = np.full((len(ounits), pad_len), -1, dtype=np.int64)
            features = np.zeros((len(ounits), pad_len, feat_size), dtype=np.float32)
            raw_units = []

            for i in range(len(ounits)):
                eval_offsets[i] = min(eval_offsets[i], lens[i])
                units[i, :(lens[i] - eval_offsets[i])] = ounits[i, eval_offsets[i]:lens[i]]
                labels[i, :(lens[i] - eval_offsets[i])] = olabels[i, eval_offsets[i]:lens[i]]
                features[i, :(lens[i] - eval_offsets[i])] = ofeatures[i, eval_offsets[i]:lens[i]]
                raw_units.append(oraw[i][eval_offsets[i]:lens[i]] + ['<PAD>'] * (pad_len - lens[i] + eval_offsets[i]))

            units = torch.from_numpy(units)
            labels = torch.from_numpy(labels)
            features = torch.from_numpy(features)

            return units, labels, features, raw_units

        def strings_starting(id_pair, offset=0, pad_len=self.args['max_seqlen']):
            # At eval time, this combines sentences in paragraph (indexed by id_pair[0]) starting sentence (indexed 
            # by id_pair[1]) into a long string for evaluation. At training time, we just select random sentences
            # from the entire dataset until we reach max_seqlen.
            pid, sid = id_pair if self.eval else random.choice(self.sentence_ids)
            sentences = [copy([x[offset:] for x in self.sentences[pid][sid]])]

            drop_sents = False if self.eval or (self.args.get('sent_drop_prob', 0) == 0) else (random.random() < self.args.get('sent_drop_prob', 0))
            total_len = len(sentences[0][0])

            assert self.eval or total_len <= self.args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(self.args['max_seqlen'], total_len, ' '.join(["{}/{}".format(*x) for x in zip(self.sentences[pid][sid])]))
            if self.eval:
                for sid1 in range(sid+1, len(self.sentences[pid])):
                    total_len += len(self.sentences[pid][sid1][0])
                    sentences.append(self.sentences[pid][sid1])

                    if total_len >= self.args['max_seqlen']:
                        break
            else:
                while True:
                    pid1, sid1 = random.choice(self.sentence_ids)
                    total_len += len(self.sentences[pid1][sid1][0])
                    sentences.append(self.sentences[pid1][sid1])

                    if total_len >= self.args['max_seqlen']:
                        break

            if drop_sents and len(sentences) > 1:
                if total_len > self.args['max_seqlen']:
                    sentences = sentences[:-1]
                if len(sentences) > 1:
                    p = [.5 ** i for i in range(1, len(sentences) + 1)] # drop a large number of sentences with smaller probability
                    cutoff = random.choices(list(range(len(sentences))), weights=list(reversed(p)))[0]
                    sentences = sentences[:cutoff+1]

            units = [val for s in sentences for val in s[0]]
            labels = [val for s in sentences for val in s[1]]
            feats = [val for s in sentences for val in s[2]]
            raw_units = [val for s in sentences for val in s[3]]

            if not self.eval:
                cutoff = self.args['max_seqlen']
                units, labels, feats, raw_units = units[:cutoff], labels[:cutoff], feats[:cutoff], raw_units[:cutoff]

            return units, labels, feats, raw_units

        if eval_offsets is not None:
            # find max padding length
            pad_len = 0
            for eval_offset in eval_offsets:
                if eval_offset < self.cumlen[-1]:
                    pair_id = bisect_right(self.cumlen, eval_offset) - 1
                    pair = self.sentence_ids[pair_id]
                    pad_len = max(pad_len, len(strings_starting(pair, offset=eval_offset-self.cumlen[pair_id])[0]))

            pad_len += 1
            id_pairs = [bisect_right(self.cumlen, eval_offset) - 1 for eval_offset in eval_offsets]
            pairs = [self.sentence_ids[pair_id] for pair_id in id_pairs]
            offsets = [eval_offset - self.cumlen[pair_id] for eval_offset, pair_id in zip(eval_offsets, id_pairs)]

            offsets_pairs = list(zip(offsets, pairs))
        else:
            id_pairs = random.sample(self.sentence_ids, min(len(self.sentence_ids), self.args['batch_size']))
            offsets_pairs = [(0, x) for x in id_pairs]
            pad_len = self.args['max_seqlen']

        # put everything into padded and nicely shaped NumPy arrays and eventually convert to PyTorch tensors
        units = np.full((len(id_pairs), pad_len), padid, dtype=np.int64)
        labels = np.full((len(id_pairs), pad_len), -1, dtype=np.int64)
        features = np.zeros((len(id_pairs), pad_len, feat_size), dtype=np.float32)
        raw_units = []
        for i, (offset, pair) in enumerate(offsets_pairs):
            u_, l_, f_, r_ = strings_starting(pair, offset=offset, pad_len=pad_len)
            units[i, :len(u_)] = u_
            labels[i, :len(l_)] = l_
            features[i, :len(f_)] = f_
            raw_units.append(r_ + ['<PAD>'] * (pad_len - len(r_)))

        if unit_dropout > 0 and not self.eval:
            # dropout characters/units at training time and replace them with UNKs
            mask = np.random.random_sample(units.shape) < unit_dropout
            mask[units == padid] = 0
            units[mask] = unkid
            for i in range(len(raw_units)):
                for j in range(len(raw_units[i])):
                    if mask[i, j]:
                        raw_units[i][j] = '<UNK>'

        units = torch.from_numpy(units)
        labels = torch.from_numpy(labels)
        features = torch.from_numpy(features)

        return units, labels, features, raw_units

