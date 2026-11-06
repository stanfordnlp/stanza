"""
Supports for pretrained data.
"""
import os

import logging
import numpy as np
import torch

from .vocab import BaseVocab, VOCAB_PREFIX

from stanza.resources.common import DEFAULT_MODEL_DIR

from pickle import UnpicklingError

logger = logging.getLogger('stanza')

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def normalize_unit(self, unit):
        unit = super().normalize_unit(unit)
        if unit:
            unit = unit.replace(" ","\xa0")
        return unit

class Pretrain:
    """
    A loader and saver for pretrained embeddings in Stanza's .pt format

    Either give a filename, which is loaded the first time vocab or
    emb is used, or give the vocab and emb directly, as
    stanza.utils.pretrain.convert_pretrain does when converting a
    word vector file.
    """

    def __init__(self, filename=None, vocab=None, emb=None):
        self.filename = filename
        if (vocab is None) != (emb is None):
            raise ValueError("Pretrain needs both vocab and emb, or neither")
        if vocab is not None:
            if len(vocab) != len(emb):
                raise ValueError("Pretrain vocab has %d entries, but there are %d vectors" % (len(vocab), len(emb)))
            self._vocab = vocab
            self._emb = emb
        elif filename is None:
            raise ValueError("Pretrain needs either a filename or a vocab and emb")

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, '_emb'):
            self.load()
        return self._emb

    def load(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError("Pretrained file %s does not exist.  Word vector files can be converted to a pretrain with python3 -m stanza.utils.pretrain.convert_pretrain" % self.filename)
        try:
            data = torch.load(self.filename, lambda storage, loc: storage, weights_only=True)
        except UnpicklingError as e:
            raise UnpicklingError("Unpickling %s failed.  If this is because it is an older model that needs weights_only=False, please convert it with a Stanza version 1.12.1 or earlier by loading and then saving." % self.filename) from e
        logger.debug("Loaded pretrain from {}".format(self.filename))
        if not isinstance(data, dict):
            raise RuntimeError("File {} exists but is not a stanza pretrain file.  It is not a dict, whereas a Stanza pretrain should have a dict with 'emb' and 'vocab'".format(self.filename))
        if 'emb' not in data or 'vocab' not in data:
            raise RuntimeError("File {} exists but is not a stanza pretrain file.  A Stanza pretrain file should have 'emb' and 'vocab' fields in its state dict".format(self.filename))
        self._vocab = PretrainedWordVocab.load_state_dict(data['vocab'])
        self._emb = data['emb']
        if isinstance(self._emb, np.ndarray):
            self._emb = torch.from_numpy(self._emb)

    def save(self, filename):
        directory, _ = os.path.split(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {'vocab': self.vocab.state_dict(), 'emb': self.emb}
        torch.save(data, filename, _use_new_zipfile_serialization=False)
        logger.info("Saved pretrained vocab and vectors to {}".format(filename))

    def write_text(self, filename, header=False):
        """
        Write the vocab & values to a text file
        """
        with open(filename, "w") as fout:
            if header:
                word_dim = self.emb[0].shape[0]
                fout.write("%d %d\n" % (len(self.vocab), word_dim))
            for word_idx, word in enumerate(self.vocab):
                row = self.emb[word_idx].to("cpu")
                fout.write(word)
                fout.write(" ")
                fout.write(" ".join(["%.6f" % x.item() for x in row]))
                fout.write("\n")


def find_pretrain_file(wordvec_pretrain_file, save_dir, shorthand, lang):
    """
    When training a model, look in a few different places for a .pt file

    If a specific argument was passsed in, prefer that location
    Otherwise, check in a few places:
      saved_models/{model}/{shorthand}.pretrain.pt
      saved_models/{model}/{shorthand}_pretrain.pt
      ~/stanza_resources/{language}/pretrain/{shorthand}_pretrain.pt
    """
    if wordvec_pretrain_file:
        return wordvec_pretrain_file

    default_pretrain_file = os.path.join(save_dir, '{}.pretrain.pt'.format(shorthand))
    if os.path.exists(default_pretrain_file):
        logger.debug("Found existing .pt file in %s" % default_pretrain_file)
        return default_pretrain_file
    else:
        logger.debug("Cannot find pretrained vectors in %s" % default_pretrain_file)

    pretrain_file = os.path.join(save_dir, '{}_pretrain.pt'.format(shorthand))
    if os.path.exists(pretrain_file):
        logger.debug("Found existing .pt file in %s" % pretrain_file)
        return pretrain_file
    else:
        logger.debug("Cannot find pretrained vectors in %s" % pretrain_file)

    if shorthand.find("_") >= 0:
        # try to assemble /home/user/stanza_resources/vi/pretrain/vtb.pt for example
        pretrain_file = os.path.join(DEFAULT_MODEL_DIR, lang, 'pretrain', '{}.pt'.format(shorthand.split('_', 1)[1]))
        if os.path.exists(pretrain_file):
            logger.debug("Found existing .pt file in %s" % pretrain_file)
            return pretrain_file
        else:
            logger.debug("Cannot find pretrained vectors in %s" % pretrain_file)

    # if we can't find it anywhere, return the first location searched,
    # so that the error when it fails to load names that location
    return default_pretrain_file
