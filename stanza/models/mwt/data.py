import random
import numpy as np
import os
from collections import Counter, namedtuple
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as DL

import stanza.models.common.seq2seq_constant as constant
from stanza.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanza.models.common.vocab import DeltaVocab
from stanza.models.mwt.vocab import Vocab
from stanza.models.common.doc import Document

logger = logging.getLogger('stanza')

DataSample = namedtuple("DataSample", "src tgt_in tgt_out orig_text")
DataBatch = namedtuple("DataBatch", "src src_mask tgt_in tgt_out orig_text orig_idx")

# enforce that the MWT splitter knows about a couple different alternate apostrophes
# including covering some potential " typos
# setting the augmentation to a very low value should be enough to teach it
# about the unknown characters without messing up the predictions for other text
#
#      0x22, 0x27, 0x02BC, 0x02CA, 0x055A, 0x07F4, 0x2019, 0xFF07
APOS = ('"',  "'",    'ʼ',    'ˊ',    '՚',    'ߴ',    '’',   '＇')

class DataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, expand_unk_vocab=False):
        self.batch_size = batch_size
        self.args = args
        self.augment_apos = args.get('augment_apos', 0.0)
        self.evaluation = evaluation
        self.doc = doc

        data = self.load_doc(self.doc, evaluation=self.evaluation)

        # handle vocab
        if vocab is None:
            assert self.evaluation == False # for eval vocab must exist
            self.vocab = self.init_vocab(data)
            if self.augment_apos > 0 and any(x in self.vocab for x in APOS):
                for apos in APOS:
                    self.vocab.add_unit(apos)
        elif expand_unk_vocab:
            self.vocab = DeltaVocab(data, vocab)
        else:
            self.vocab = vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.evaluation:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        # shuffle for training
        if not self.evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        self.data = data
        self.num_examples = len(data)

    def init_vocab(self, data):
        assert self.evaluation == False # for eval vocab must exist
        vocab = Vocab(data, self.args['shorthand'])
        return vocab

    def maybe_augment_apos(self, datum):
        for original in APOS:
            if original in datum[0]:
                if random.uniform(0,1) < self.augment_apos:
                    replacement = random.choice(APOS)
                    datum = (datum[0].replace(original, replacement), datum[1].replace(original, replacement))
                break
        return datum

    def process(self, sample):
        if not self.evaluation and self.augment_apos > 0:
            sample = self.maybe_augment_apos(sample)
        src = list(sample[0])
        src = [constant.SOS] + src + [constant.EOS]
        tgt_in, tgt_out = self.prepare_target(self.vocab, sample)
        src = self.vocab.map(src)
        processed = [src, tgt_in, tgt_out, sample[0]]
        return processed

    def prepare_target(self, vocab, datum):
        if self.evaluation:
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
        sample = self.data[key]
        sample = self.process(sample)
        assert len(sample) == 4

        src = torch.tensor(sample[0])
        tgt_in = torch.tensor(sample[1])
        tgt_out = torch.tensor(sample[2])
        orig_text = sample[3]
        result = DataSample(src, tgt_in, tgt_out, orig_text), key
        return result

    @staticmethod
    def __collate_fn(data):
        (data, idx) = zip(*data)
        (src, tgt_in, tgt_out, orig_text) = zip(*data)

        # collate_fn is given a list of length batch size
        batch_size = len(data)

        # need to sort by length of src to properly handle
        # the batching in the model itself
        lens = [len(x) for x in src]
        (src, tgt_in, tgt_out, orig_text), orig_idx = sort_all((src, tgt_in, tgt_out, orig_text), lens)
        lens = [len(x) for x in src]

        # convert to tensors
        src = pad_sequence(src, True, constant.PAD_ID)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = pad_sequence(tgt_in, True, constant.PAD_ID)
        tgt_out = pad_sequence(tgt_out, True, constant.PAD_ID)
        assert tgt_in.size(1) == tgt_out.size(1), \
                "Target input and output sequence sizes do not match."
        return DataBatch(src, src_mask, tgt_in, tgt_out, orig_text, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def to_loader(self):
        """Converts self to a DataLoader """

        batch_size = self.batch_size
        shuffle = not self.evaluation
        return DL(self,
                  collate_fn=self.__collate_fn,
                  batch_size=batch_size,
                  shuffle=shuffle)

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
        src = datum[0] if self.evaluation else datum[1]
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

