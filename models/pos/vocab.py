from collections import Counter

from models.common.vocab import Vocab as BaseVocab
from models.common.vocab import ComposedVocab
import models.common.seq2seq_constant as constant

class CharVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([c for sent in self.data for w in sent for c in w[self.idx]])

        self._id2unit = constant.VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class WordVocab(BaseVocab):
    def build_vocab(self):
        if self.lower:
            counter = Counter([w[self.idx].lower() for sent in self.data for w in sent])
        else:
            counter = Counter([w[self.idx] for sent in self.data for w in sent])
        for k in list(counter.keys()):
            if counter[k] < self.cutoff:
                del counter[k]

        self._id2unit = constant.VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class XPOSVocab(ComposedVocab):
    def __init__(self, filename, data, lang, idx=0, sep="", keyed=False):
        super().__init__(filename, data, lang, idx=idx, sep=sep, keyed=keyed)

class FeatureVocab(ComposedVocab):
    def __init__(self, filename, data, lang, idx=0, sep="|", keyed=True):
        super().__init__(filename, data, lang, idx=idx, sep=sep, keyed=keyed)

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = constant.VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}
