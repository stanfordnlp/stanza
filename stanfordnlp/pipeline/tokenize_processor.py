import argparse
import json
import numpy as np
import random
import re
import torch

from bisect import bisect_right
from copy import copy
from stanfordnlp.models.common import conll
from stanfordnlp.models.common import utils
from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.models.tokenize.trainer import Trainer
from stanfordnlp.models.tokenize.vocab import Vocab

DEFAULT_TOKENIZE_CONFIG = {
    'mode': 'predict',
    'shorthand': 'en_ewt',
    'lang': 'en',
    'cuda': True,
    'max_seqlen': 1000,
    'feat_funcs': ['space_before', 'capitalized', 'all_caps', 'numeric'],
    'feat_dim': 4,
    'model_path': 'saved_models/tokenize/en_ewt_tokenizer.pt'
}


# class for running the tokenizer
class TokenizeProcessor:

    def __init__(self, config=DEFAULT_TOKENIZE_CONFIG):
        # set up configurations
        self.args = DEFAULT_TOKENIZE_CONFIG
        for key in config.keys():
            self.args[key] = config[key]
        # set up trainer
        self.trainer = Trainer(self.args)
        self.trainer.load(self.args['model_path'])
        # set configurations from loaded model
        loaded_args = self.trainer.args
        self.vocab = self.trainer.vocab
        self.args['vocab_size'] = len(self.vocab)
        for k in loaded_args:
            if not k.endswith('_file') and k not in ['cuda', 'mode', 'save_dir', 'save_name']:
                self.args[k] = loaded_args[k]
        if self.args['cuda']:
            self.trainer.model.cuda()
        # set up misc
        self.mwt_dict = None

    def build_input_data(self, text):
        return DataLoader(self.args, input_text=text, vocab=self.vocab)

    def process(self, doc):
        return_sentences = []
        data_loader = self.build_input_data(doc.text)
        # more data set up for tokenization
        paragraphs = []
        for i, p in enumerate(data_loader.sentences):
            start = 0 if i == 0 else paragraphs[-1][2]
            length = sum([len(x) for x in p])
            paragraphs += [(i, start, start + length, length + 1)]  # para idx, start idx, end idx, length
        paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))
        all_preds = [None] * len(paragraphs)
        all_raw = [None] * len(paragraphs)
        eval_limit = max(3000, self.args['max_seqlen'])
        batch_size = self.trainer.args['batch_size']
        batches = int((len(paragraphs) + batch_size - 1) / batch_size)
        # process batches
        t = 0
        for i in range(batches):
            batchparas = paragraphs[i * batch_size: (i + 1) * batch_size]
            offsets = [x[1] for x in batchparas]
            t += sum([x[3] for x in batchparas])
            batch = data_loader.next(eval_offsets=offsets)
            raw = batch[3]
            N = len(batch[3][0])
            # get the predictions
            pred = np.argmax(self.trainer.predict(batch), axis=2)
            # post process after predictions
            for j, p in enumerate(batchparas):
                len1 = len([1 for x in raw[j] if x != '<PAD>'])
                if pred[j][len1 - 1] < 2:
                    pred[j][len1 - 1] = 2
                elif pred[j][len1 - 1] > 2:
                    pred[j][len1 - 1] = 4
                all_preds[p[0]] = pred[j][:len1]
                all_raw[p[0]] = raw[j]
        # generate output conll-u from predictions
        offset = 0
        oov_count = 0
        for j in range(len(paragraphs)):
            raw = all_raw[j]
            pred = all_preds[j]
            current_tok = ''
            current_sent = []
            for t, p in zip(raw, pred):
                if t == '<PAD>':
                    break
                # hack la_ittb
                if self.trainer.args['shorthand'] == 'la_ittb' and t in [":", ";"]:
                    p = 2
                offset += 1
                if self.vocab.unit2id(t) == self.vocab.unit2id('<UNK>'):
                    oov_count += 1
                current_tok += t
                if p >= 1:
                    tok = self.vocab.normalize_token(current_tok)
                    assert '\t' not in tok, tok
                    if len(tok) <= 0:
                        current_tok = ''
                        continue
                    current_sent += [(tok, p)]
                    current_tok = ''
                    if p == 2 or p == 4:
                        return_sentences.append(self.build_conllu(current_sent, self.mwt_dict))
                        current_sent = []
            if len(current_tok):
                tok = self.vocab.normalize_token(current_tok)
                assert '\t' not in tok, tok
                if len(tok) > 0:
                    current_sent += [(tok, 2)]
            if len(current_sent):
                return_sentences.append(self.build_conllu(current_sent, self.mwt_dict))
        # build initial conll_file
        conll_file = conll.CoNLLFile(input_str=('\n'.join(return_sentences)))
        # set doc's conll file
        doc.conll_file = conll_file

    def build_conllu(self, sentence, mwt_dict=None):
        return_string = ""
        i = 0
        for tok, p in sentence:
            expansion = None
            if (p == 3 or p == 4) and mwt_dict is not None:
                # MWT found, (attempt to) expand it!
                if tok in mwt_dict:
                    expansion = mwt_dict[tok][0]
                elif tok.lower() in mwt_dict:
                    expansion = mwt_dict[tok.lower()][0]
            if expansion is not None:
                return_string += ("{}-{}\t{}{}".format(i + 1, i + len(expansion), tok, "\t_" * 8))
                return_string += '\n'
                for etok in expansion:
                    return_string += ("{}\t{}{}\t{}{}".format(i + 1, etok, "\t_" * 4, i, "\t_" * 3))
                    return_string += '\n'
                    i += 1
            else:
                if len(tok) <= 0:
                    continue
                return_string += ("{}\t{}{}\t{}{}\t{}".format(
                    i + 1, tok, "\t_" * 4, i, "\t_" * 2, "MWT=Yes" if p == 3 or p == 4 else "_"))
                return_string += '\n'
                i += 1
        return return_string
