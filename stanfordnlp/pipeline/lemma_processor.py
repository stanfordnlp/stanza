import random
import numpy as np
import os
from collections import Counter
import torch

from stanfordnlp.models.common.conll import FIELD_TO_IDX

import stanfordnlp.models.common.seq2seq_constant as constant
from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
from stanfordnlp.models.lemma.data import DataLoader
from stanfordnlp.models.lemma.vocab import Vocab, MultiVocab
from stanfordnlp.models.lemma import edit
from stanfordnlp.models.lemma.trainer import Trainer

DEFAULT_LEMMA_CONFIG = {
    'mode': 'predict',
    'shorthand': 'en_ewt',
    'lang': 'en_ewt',
    'cuda': True,
    'max_seqlen': 1000,
    'feat_funcs': ['space_before', 'capitalized', 'all_caps', 'numeric'],
    'feat_dim': 4,
    'model_path': 'saved_models/lemma/en_ewt_lemmatizer.pt',
    'batch_size': 1,
    'cpu': False
}


class LemmaProcessor:
    def __init__(self, config={}):
        # set up configurations
        self.args = DEFAULT_LEMMA_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # set up trainer
        self.trainer = Trainer(model_file=self.args['model_path'])
        loaded_args, vocab = self.trainer.args, self.trainer.vocab
        for k in self.args:
            if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand']:
                loaded_args[k] = self.args[k]
        loaded_args['cuda'] = self.args['cuda'] and not self.args['cpu']
        self.loaded_args = loaded_args
        self.vocab = vocab

    def process(self, doc):
        batch = DataLoader(doc, self.args['batch_size'], self.loaded_args, vocab=self.vocab, evaluation=True)
        dict_preds = self.trainer.predict_dict(batch.conll.get(['word', 'upos']))
        doc.conll_file = conll.CoNLLFile(input_str=self.write_conll_with_lemmas(batch.conll, dict_preds))

    def write_conll_with_lemmas(self, input_conll, lemmas):
        """ Write a new conll file, but use the new lemmas to replace the old ones."""
        return_string = ""
        assert input_conll.num_words == len(lemmas), "Num of lemmas does not match the number in original data file."
        lemma_idx = FIELD_TO_IDX['lemma']
        idx = 0
        for sent in input_conll.sents:
            for ln in sent:
                if '-' not in ln[0]:  # do not process if it is a mwt line
                    lm = lemmas[idx]
                    if len(lm) == 0:
                        lm = '_'
                    ln[lemma_idx] = lm
                    idx += 1
                return_string += ("\t".join(ln))
                return_string += "\n"
            return_string += "\n"
        return return_string



