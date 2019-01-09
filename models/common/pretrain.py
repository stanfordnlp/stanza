"""
Supports for pretrained data.
"""
import os
import lzma
import numpy as np
import torch

from .vocab import BaseVocab, VOCAB_PREFIX

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

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
            except BaseException as e:
                print("Pretrained file exists but cannot be loaded from {}, due to the following exception:".format(self.filename))
                print("\t{}".format(e))
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

        vocab = PretrainedWordVocab(words, lower=True)

        if failed > 0:
            emb = emb[:-failed]

        # save to file
        data = {'vocab': vocab, 'emb': emb}
        try:
            torch.save(data, self.filename)
            print("Saved pretrained vocab and vectors to {}".format(self.filename))
        except BaseException as e:
            print("Saving pretrained data failed due to the following exception... continuing anyway")
            print("\t{}".format(e))

        return vocab, emb

