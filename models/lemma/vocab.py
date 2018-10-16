"""
A class for basic vocab operations.
"""

from __future__ import print_function
import os
import random
import numpy as np
import pickle

import models.common.seq2seq_constant as constant

random.seed(1234)
np.random.seed(1234)

class Vocab(object):
    def __init__(self, unit_counter=None, threshold=0):
        assert unit_counter is not None, "unit_counter is not provided for vocab creation."
        self.unit_counter = unit_counter
        if threshold > 1:
            # remove words that occur less than thres
            self.unit_counter = dict([(k,v) for k,v in self.unit_counter.items() if v >= threshold])
        self.id2unit = sorted(self.unit_counter, key=lambda k:self.unit_counter[k], reverse=True)
        # add special tokens to the beginning
        self.id2unit = constant.VOCAB_PREFIX + self.id2unit
        self.unit2id = dict([(self.id2unit[idx],idx) for idx in range(len(self.id2unit))])
        self.size = len(self.id2unit)

    def map(self, token_list):
        """
        Map a list of tokens to their ids.
        """
        return [self.unit2id[w] if w in self.unit2id else constant.UNK_ID for w in token_list]

    def unmap(self, idx_list):
        """
        Unmap ids back to tokens.
        """
        return [self.id2unit[idx] for idx in idx_list]
    
    def get_embeddings(self, unit_vectors=None, dim=100):
        #self.embeddings = 2 * constant.EMB_INIT_RANGE * np.random.rand(self.size, dim) - constant.EMB_INIT_RANGE
        self.embeddings = np.zeros((self.size, dim))
        if unit_vectors is not None:
            assert len(list(unit_vectors.values())[0]) == dim, \
                    "Unit vectors does not have required dimension {}.".format(dim)
            for w, idx in self.unit2id.items():
                if w in unit_vectors:
                    self.embeddings[idx] = np.asarray(unit_vectors[w])
        return self.embeddings

