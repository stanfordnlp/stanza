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

    def __init__(self, filename, vec_filename=None, max_vocab=-1):
        self.filename = filename
        self._vec_filename = vec_filename
        self._max_vocab = max_vocab

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
        if self._vec_filename is None:
            raise Exception("Vector file is not provided.")
        print("Reading pretrained vectors from {}...".format(self._vec_filename))

        # first try reading as xz file, if failed retry as text file
        try:
            words, emb, failed = self.read_from_file(self._vec_filename, open_func=lzma.open)
        except lzma.LZMAError as err:
            print("Cannot decode vector file as xz file. Retrying as text file...")
            words, emb, failed = self.read_from_file(self._vec_filename, open_func=open)

        if failed > 0: # recover failure
            emb = emb[:-failed]
        if len(emb) - len(VOCAB_PREFIX) != len(words):
            raise Exception("Loaded number of vectors does not match number of words.")
        
        # Use a fixed vocab size
        if self._max_vocab > len(VOCAB_PREFIX) and self._max_vocab < len(words):
            words = words[:self._max_vocab - len(VOCAB_PREFIX)]
            emb = emb[:self._max_vocab]

        vocab = PretrainedWordVocab(words, lower=True)

        # save to file
        data = {'vocab': vocab, 'emb': emb}
        try:
            torch.save(data, self.filename)
            print("Saved pretrained vocab and vectors to {}".format(self.filename))
        except BaseException as e:
            print("Saving pretrained data failed due to the following exception... continuing anyway")
            print("\t{}".format(e))

        return vocab, emb

    def read_from_file(self, filename, open_func=open):
        """
        Open a vector file using the provided function and read from it.
        """
        first = True
        words = []
        failed = 0
        with open_func(filename, 'rb') as f:
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
        return words, emb, failed
