from collections import Counter

from models.common.vocab import Vocab as BaseVocab
from models.common.vocab import CompositeVocab, VOCAB_PREFIX, EMPTY, EMPTY_ID

class CharVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([c for sent in self.data for w in sent for c in w[self.idx]])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class WordVocab(BaseVocab):
    def __init__(self, data, lang="", idx=0, cutoff=0, lower=False, ignore=[]):
        self.ignore = ignore
        super().__init__(data, lang=lang, idx=idx, cutoff=cutoff, lower=lower)

    def id2unit(self, id):
        if len(self.ignore) > 0 and id == EMPTY_ID:
            return '_'
        else:
            return super().id2unit(id)

    def unit2id(self, unit):
        if len(self.ignore) > 0 and unit in self.ignore:
            return self._unit2id[EMPTY]
        else:
            return super().unit2id(unit)

    def build_vocab(self):
        if self.lower:
            counter = Counter([w[self.idx].lower() for sent in self.data for w in sent])
        else:
            counter = Counter([w[self.idx] for sent in self.data for w in sent])
        for k in list(counter.keys()):
            if counter[k] < self.cutoff or k in self.ignore:
                del counter[k]

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class XPOSVocab(CompositeVocab):
    def __init__(self, data, lang, idx=0, sep="", keyed=False):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class FeatureVocab(CompositeVocab):
    def __init__(self, data, lang, idx=0, sep="|", keyed=True):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

