import random
import numpy as np
import os
from collections import Counter
import torch

import models.common.seq2seq_constant as constant
from models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from models.common import conll
from models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab

class DataLoader:
    def __init__(self, filename, batch_size, args, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        assert filename.endswith('conllu'), "Loaded file must be conllu file."
        self.conll, data = self.load_file(filename, evaluation=self.eval)

        # handle vocab
        vocab_pattern = "{}/{}.{{}}.vocab".format(args['data_dir'], args['shorthand'])
        self.vocab = self.init_vocab(vocab_pattern, data)

	# filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

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
        print("{} batches created for {}.".format(len(data), filename))

    def init_vocab(self, vocab_pattern, data):
        types = ['char', 'word', 'upos', 'xpos', 'feats']
        if not all([os.path.exists(vocab_pattern.format(type_)) for type_ in types]):
            assert self.eval == False # for eval vocab file must exist
        charvocab = CharVocab(vocab_pattern.format('char'), data, self.args['shorthand'])
        wordvocab = WordVocab(vocab_pattern.format('word'), data, self.args['shorthand'], cutoff=7)
        uposvocab = WordVocab(vocab_pattern.format('upos'), data, self.args['shorthand'], idx=1)
        # TODO: make XPOSVocab language-specific
        xposvocab = XPOSVocab(vocab_pattern.format('xpos'), data, self.args['shorthand'], idx=2)
        featsvocab = FeatureVocab(vocab_pattern.format('feats'), data, self.args['shorthand'], idx=3)
        vocab = {'char': charvocab,
                'word': wordvocab,
                'upos': uposvocab,
                'xpos': xposvocab,
                'feats': featsvocab}
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        for sent in data:
            processed_sent = [map_to_ids([w[0] for w in sent], vocab['word'])]
            processed_sent += [[map_to_ids([x for x in w[0]], vocab['char']) for w in sent]]
            processed_sent += [map_to_ids([w[1] for w in sent], vocab['upos'])]
            processed_sent += [map_to_ids([w[2] for w in sent], vocab['xpos'])]
            processed_sent += [map_to_ids([w[3] for w in sent], vocab['feats'])]
            processed.append(processed_sent)
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

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, constant.PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, constant.PAD_ID)

        # TODO: deal with UPOS, XPOS, and UFeats
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        return (src, src_mask, tgt_in, tgt_out, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
        return conll_file, data

