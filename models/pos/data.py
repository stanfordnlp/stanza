import random
import numpy as np
import lzma
import os
import pickle
from collections import Counter
import torch

from models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from models.common import conll
from models.common.constant import lcode2lang
from models.common.vocab import PAD_ID, VOCAB_PREFIX
from models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, PretrainedWordVocab
from models.pos.xpos_vocab_factory import xpos_vocab_factory

class Pretrain:
    """ A loader and saver for pretrained embeddings. """

    def __init__(self, filename, vec_filename=None):
        self.filename = filename
        self.vec_filename = vec_filename

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab, self._emb = self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, '_emb'):
            self._vocab, self._emb = self.load()
        return self._emb

    def load(self):
        if os.path.exists(self.filename):
            try:
                data = torch.load(self.filename, lambda storage, loc: storage)
            except BaseException:
                print("Pretrained file exists but cannot be loaded from {}".format(self.filename))
                return self.read_and_save()
            return data['vocab'], data['emb']
        else:
            return self.read_and_save()

    def read_and_save(self):
        # load from pretrained filename
        if self.vec_filename is None:
            raise Exception("Vector file is not provided.")
        print("Reading pretrained vectors from {}...".format(self.vec_filename))
        first = True
        words = []
        failed = 0
        with lzma.open(self.vec_filename, 'rb') as f:
            for i, line in enumerate(f):
                try:
                    line = line.decode()
                except UnicodeDecodeError:
                    failed += 1
                    continue
                if first:
                    # the first line contains the number of word vectors and the dimensionality
                    first = False
                    line = line.strip().split(' ')
                    rows, cols = [int(x) for x in line]
                    emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
                    continue

                line = line.rstrip().split(' ')
                emb[i+len(VOCAB_PREFIX)-1-failed, :] = [float(x) for x in line[-cols:]]
                words.append(' '.join(line[:-cols]))

        vocab = PretrainedWordVocab(None, words, "") # TODO: fix the BaseVocab interface

        if failed > 0:
            emb = emb[:-failed]

        # save to file
        data = {'vocab': vocab, 'emb': emb}
        try:
            torch.save(data, self.filename)
            print("Saved pretrained vocab and vectors to {}".format(self.filename))
        except BaseException:
            print("Saving pretrained data failed... continuing anyway")

        return vocab, emb

class DataLoader:
    def __init__(self, filename, batch_size, args, pretrain, vocab=None, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        assert filename.endswith('conllu'), "Loaded file must be conllu file."
        self.conll, data = self.load_file(filename, evaluation=self.eval)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        self.pretrain_vocab = pretrain.vocab

	# filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

	# chunk into batches
        self.data = self.chunk_batches(data)
        print("{} batches created for {}.".format(len(self.data), filename))

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        charvocab = CharVocab(None, data, self.args['shorthand'])
        wordvocab = WordVocab(None, data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(None, data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(None, data, self.args['shorthand'])
        featsvocab = FeatureVocab(None, data, self.args['shorthand'], idx=3)
        vocab = {'char': charvocab,
                'word': wordvocab,
                'upos': uposvocab,
                'xpos': xposvocab,
                'feats': featsvocab}
        return vocab
    
    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        for sent in data:
            processed_sent = [vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [pretrain_vocab.map([w[0] for w in sent])]
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
        assert len(batch) == 6

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_long_tensor(batch[2], batch_size)
        xpos = get_long_tensor(batch[3], batch_size)
        ufeats = get_long_tensor(batch[4], batch_size)
        pretrained = get_long_tensor(batch[5], batch_size)
        sentlens = [len(x) for x in batch[0]]
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, orig_idx, word_orig_idx, sentlens, word_lens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
        return conll_file, data

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key = lambda x: len(x[0]) + random.random() * 5)

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res
