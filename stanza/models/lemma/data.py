import random
import numpy as np
import os
from collections import Counter
import logging
import torch

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.lemma.vocab import Vocab, MultiVocab
from stanza.models.lemma import edit
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')

class DataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, conll_only=False, skip=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc

        data = self.load_doc(self.doc)

        if conll_only: # only load conll file
            return

        if skip is not None:
            assert len(data) == len(skip)
            data = [x for x, y in zip(data, skip) if not y]

        # handle vocab
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            char_vocab, pos_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab})

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab['char'], self.vocab['pos'], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        logger.debug("{} batches created.".format(len(data)))

    def init_vocab(self, data):
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = "".join(d[0] + d[2] for d in data)
        char_vocab = Vocab(char_data, self.args['lang'])
        pos_data = [d[1] for d in data]
        pos_vocab = Vocab(pos_data, self.args['lang'])
        return char_vocab, pos_vocab

    def preprocess(self, data, char_vocab, pos_vocab, args):
        processed = []
        for d in data:
            edit_type = edit.EDIT_TO_ID[edit.get_edit_type(d[0], d[2])]
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            src = char_vocab.map(src)
            pos = d[1]
            pos = pos_vocab.unit2id(pos)
            tgt = list(d[2])
            tgt_in = char_vocab.map([constant.SOS] + tgt)
            tgt_out = char_vocab.map(tgt + [constant.EOS])
            processed += [[src, tgt_in, tgt_out, pos, edit_type]]
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        pos = torch.LongTensor(batch[3])
        edits = torch.LongTensor(batch[4])
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return src, src_mask, tgt_in, tgt_out, pos, edits, orig_idx

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc):
        data = doc.get([TEXT, UPOS, LEMMA])
        data = self.resolve_none(data)
        return data

    def resolve_none(self, data):
        # replace None to '_'
        for tok_idx in range(len(data)):
            for feat_idx in range(len(data[tok_idx])):
                if data[tok_idx][feat_idx] is None:
                    data[tok_idx][feat_idx] = '_'
        return data