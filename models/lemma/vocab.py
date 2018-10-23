from collections import Counter

from models.common.vocab import Vocab as BaseVocab
from models.common.seq2seq_constant import VOCAB_PREFIX

class Vocab(BaseVocab):
    def build_vocab(self):
        counter = Counter(self.data)
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

