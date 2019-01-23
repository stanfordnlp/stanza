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
    'beam_size': 1,
    'feat_funcs': ['space_before', 'capitalized', 'all_caps', 'numeric'],
    'feat_dim': 4,
    'model_path': 'saved_models/lemma/en_ewt_lemmatizer.pt',
    'batch_size': 1,
    'edit': True,
    'ensemble_dict': True,
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
        if self.args.get('use_identity') in ['True', True]:
            self.use_identity = True
        else:
            self.use_identity = False

    def process(self, doc):
        batch = DataLoader(doc, self.args['batch_size'], self.loaded_args, vocab=self.vocab, evaluation=True)
        dict_preds = self.trainer.predict_dict(batch.conll.get(['word', 'upos']))
        if self.use_identity:
            preds = [ln[FIELD_TO_IDX['word']] for sent in batch.conll.sents for ln in sent if '-' not in ln[0]]
        elif self.loaded_args.get('dict_only', False):
            preds = dict_preds
        else:
            print("Running the seq2seq model...")
            preds = []
            edits = []
            for i, b in enumerate(batch):
                ps, es = self.trainer.predict(b, self.args['beam_size'])
                preds += ps
                if es is not None:
                    edits += es
            preds = self.trainer.postprocess(batch.conll.get(['word']), preds, edits=edits)
            
            if self.loaded_args.get('ensemble_dict', False):
                print("[Ensembling dict with seq2seq lemmatizer...]")
                preds = self.trainer.ensemble(batch.conll.get(['word', 'upos']), preds)   
        
        # map empty string lemmas to '_'
        preds = [max([(len(x),x), (0, '_')])[1] for x in preds]
        batch.conll.set(['lemma'], preds)

