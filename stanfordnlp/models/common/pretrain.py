"""
Supports for pretrained data.
"""
import os
import re

import lzma
import logging
import numpy as np
import torch

from .vocab import BaseVocab, VOCAB_PREFIX

logger = logging.getLogger(__name__)

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class Pretrain:
    """ A loader and saver for pretrained embeddings. """

    def __init__(self, filename=None, vec_filename=None, max_vocab=-1, save_to_file=True):
        self.filename = filename
        self._vec_filename = vec_filename
        self._max_vocab = max_vocab
        self._save_to_file = save_to_file

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
        if self.filename is not None and os.path.exists(self.filename):
            try:
                data = torch.load(self.filename, lambda storage, loc: storage)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                logger.warning("Pretrained file exists but cannot be loaded from {}, due to the following exception:\n\t{}".format(self.filename, e))
                return self.read_pretrain()
            return data['vocab'], data['emb']
        else:
            return self.read_pretrain()

    def read_pretrain(self):
        # load from pretrained filename
        if self._vec_filename is None:
            raise Exception("Vector file is not provided.")
        logger.info("Reading pretrained vectors from {}...".format(self._vec_filename))

        # first try reading as xz file, if failed retry as text file
        try:
            words, emb, failed = self.read_from_file(self._vec_filename, open_func=lzma.open)
        except lzma.LZMAError as err:
            logging.warning("Cannot decode vector file %s as xz file. Retrying as text file..." % self._vec_filename)
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
        
        if self._save_to_file:
            assert self.filename is not None, "Filename must be provided to save pretrained vector to file."
            # save to file
            data = {'vocab': vocab, 'emb': emb}
            try:
                torch.save(data, self.filename)
                logger.info("Saved pretrained vocab and vectors to {}".format(self.filename))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                logger.warning("Saving pretrained data failed due to the following exception... continuing anyway.\n\t{}".format(e))

        return vocab, emb

    def read_from_file(self, filename, open_func=open):
        """
        Open a vector file using the provided function and read from it.
        """
        # some vector files, such as Google News, use tabs
        tab_space_pattern = re.compile("[ \t]+")
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

                line = tab_space_pattern.split((line.rstrip()))
                emb[i+len(VOCAB_PREFIX)-1-failed, :] = [float(x) for x in line[-cols:]]
                words.append(' '.join(line[:-cols]))
        return words, emb, failed
