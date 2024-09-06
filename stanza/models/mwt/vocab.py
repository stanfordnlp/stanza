from collections import Counter

from stanza.models.common.vocab import BaseVocab
import stanza.models.common.seq2seq_constant as constant

class Vocab(BaseVocab):
    def build_vocab(self):
        pairs = self.data
        allchars = "".join([src + tgt for src, tgt in pairs])
        counter = Counter(allchars)

        self._id2unit = constant.VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class MWTDeltaVocab(BaseVocab):
    def __init__(self, data, orig_vocab):
        self.orig_vocab = orig_vocab
        super().__init__(data=data, lang=orig_vocab.lang, idx=orig_vocab.idx, cutoff=orig_vocab.cutoff, lower=orig_vocab.lower)

    def build_vocab(self):
        allchars = "".join([word for sentence in self.data for word in sentence])

        unk = [c for c in allchars if c not in self.orig_vocab._unit2id]
        if len(unk) > 0:
            unk = sorted(set(unk))
            self._id2unit = self.orig_vocab._id2unit + unk
            self._unit2id = dict(self.orig_vocab._unit2id)
            for c in unk:
                self._unit2id[c] = len(self._unit2id)
        else:
            self._id2unit = self.orig_vocab._id2unit
            self._unit2id = self.orig_vocab._unit2id
