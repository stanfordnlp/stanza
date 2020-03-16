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
