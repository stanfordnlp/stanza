import random
import numpy as np
import os
from collections import Counter
import logging
import torch

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import DeltaVocab
from stanza.models.mwt.vocab import Vocab
from stanza.models.common.doc import Document

logger = logging.getLogger('stanza')

class DataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, expand_unk_vocab=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc

        data = self.load_doc(self.doc, evaluation=self.eval)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        elif expand_unk_vocab:
            self.vocab = DeltaVocab(data, vocab)
        else:
            self.vocab = vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, args)
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
        assert self.eval == False # for eval vocab must exist
        vocab = Vocab(data, self.args['shorthand'])
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        for d in data:
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            tgt_in, tgt_out = self.prepare_target(vocab, d)
            src = vocab.map(src)
            processed += [[src, tgt_in, tgt_out, d[0]]]
        return processed

    def prepare_target(self, vocab, datum):
        if self.eval:
            tgt = list(datum[0])  # as a placeholder
        else:
            tgt = list(datum[1])
        tgt_in = vocab.map([constant.SOS] + tgt)
        tgt_out = vocab.map(tgt + [constant.EOS])
        return tgt_in, tgt_out

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
        assert len(batch) == 4

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        orig_text = batch[3]
        assert tgt_in.size(1) == tgt_out.size(1), \
                "Target input and output sequence sizes do not match."
        return (src, src_mask, tgt_in, tgt_out, orig_text, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc, evaluation=False):
        data = doc.get_mwt_expansions(evaluation)
        if evaluation: data = [[e] for e in data]
        return data

class BinaryDataLoader(DataLoader):
    """
    This version of the DataLoader performs the same tasks as the regular DataLoader,
    except the targets are arrays of 0/1 indicating if the character is the location
    of an MWT split
    """
    def prepare_target(self, vocab, datum):
        src = datum[0] if self.eval else datum[1]
        binary = [0]
        has_space = False
        for char in src:
            if char == ' ':
                has_space = True
            elif has_space:
                has_space = False
                binary.append(1)
            else:
                binary.append(0)
        binary.append(0)
        return binary, binary

