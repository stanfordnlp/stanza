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
from models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab
from models.pos.vocab import CharVocab, WordVocab, XPOSVocab, FeatureVocab, PretrainedWordVocab
from models.pos.xpos_vocab_factory import xpos_vocab_factory
from models.pos.data import DataLoader as TaggerDataLoader

class DataLoader(TaggerDataLoader):
    def init_vocab(self, vocab_pattern, data):
        types = ['char', 'word', 'upos', 'xpos', 'feats', 'lemma', 'deprel']
        if not all([os.path.exists(vocab_pattern.format(type_)) for type_ in types]):
            assert self.eval == False # for eval vocab file must exist
        charvocab = CharVocab(vocab_pattern.format('char'), data, self.args['shorthand'])
        wordvocab = WordVocab(vocab_pattern.format('word'), data, self.args['shorthand'], cutoff=7, lower=True)
        self.pretrained_emb, pretrainedvocab = self.read_emb_matrix(self.args['wordvec_dir'], self.args['shorthand'], vocab_pattern.format('pretrained'))
        uposvocab = WordVocab(vocab_pattern.format('upos'), data, self.args['shorthand'], idx=1)
        xposvocab = xpos_vocab_factory(vocab_pattern.format('xpos'), data, self.args['shorthand'])
        featsvocab = FeatureVocab(vocab_pattern.format('feats'), data, self.args['shorthand'], idx=3)
        lemmavocab = WordVocab(vocab_pattern.format('lemma'), data, self.args['shorthand'], cutoff=7, idx=4, lower=True)
        deprelvocab = WordVocab(vocab_pattern.format('deprel'), data, self.args['shorthand'], idx=6)
        vocab = {'char': charvocab,
                'word': wordvocab,
                'pretrained': pretrainedvocab,
                'upos': uposvocab,
                'xpos': xposvocab,
                'feats': featsvocab,
                'lemma': lemmavocab,
                'deprel': deprelvocab}
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        xpos_replacement = [[ROOT_ID] * len(vocab['xpos'])] if isinstance(vocab['xpos'], CompositeVocab) else [ROOT_ID]
        feats_replacement = [[ROOT_ID] * len(vocab['feats'])]
        for sent in data:
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [xpos_replacement + vocab['xpos'].map([w[2] for w in sent])]
            processed_sent += [feats_replacement + vocab['feats'].map([w[3] for w in sent])]
            processed_sent += [[ROOT_ID] + vocab['pretrained'].map([w[0] for w in sent])]
            processed_sent += [[ROOT_ID] + vocab['lemma'].map([w[4] for w in sent])]
            processed_sent += [[int(w[5]) for w in sent]]
            processed_sent += [vocab['deprel'].map([w[6] for w in sent])]
            processed.append(processed_sent)
        return processed

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 9

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
        lemma = get_long_tensor(batch[6], batch_size)
        head = get_long_tensor(batch[7], batch_size)
        deprel = get_long_tensor(batch[8], batch_size)
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'xpos', 'feats', 'lemma', 'head', 'deprel'], as_sentences=True)
        return conll_file, data
