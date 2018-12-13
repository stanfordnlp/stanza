import random
import numpy as np
import os
from collections import Counter
import torch

from stanfordnlp.models.common.conll import FIELD_TO_IDX

import stanfordnlp.models.common.seq2seq_constant as constant
from stanfordnlp.models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from stanfordnlp.models.common import conll
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


class LemmaDataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, conll_only=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        self.conll, data = self.load_data(doc)

        if conll_only:  # only load conll file
            return

        # handle vocab
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            char_vocab, pos_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab})

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab['char'], self.vocab['pos'], args)
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def init_vocab(self, data):
        assert self.eval == False, "Vocab file must exist for evaluation"
        char_data = "".join(d[0] + d[2] for d in data)
        char_vocab = Vocab(char_data, self.args['lang'])
        pos_data = [d[1] for d in data]
        pos_vocab = Vocab(pos_data, self.args['lang'])
        return char_vocab, pos_vocab

    def preprocess(self, data, char_vocab, pos_vocab, args):
        processed = []
        for d in data:
            edit_type = edit.EDIT_TO_ID[edit.get_edit_type(d[0], d[2])]
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            src = char_vocab.map(src)
            pos = d[1]
            pos = pos_vocab.unit2id(pos)
            tgt = list(d[2])
            tgt_in = char_vocab.map([constant.SOS] + tgt)
            tgt_out = char_vocab.map(tgt + [constant.EOS])
            processed += [[src, tgt_in, tgt_out, pos, edit_type]]
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

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        pos = torch.LongTensor(batch[3])
        edits = torch.LongTensor(batch[4])
        assert tgt_in.size(1) == tgt_out.size(1), \
                "Target input and output sequence sizes do not match."
        return (src, src_mask, tgt_in, tgt_out, pos, edits, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_data(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'lemma'])
        return doc.conll_file, data


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
        batch = LemmaDataLoader(doc, self.args['batch_size'], self.loaded_args, vocab=self.vocab,
                                evaluation=True)
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



