from collections import Counter

from stanza.models.common.vocab import BaseVocab, BaseMultiVocab
from stanza.models.common.seq2seq_constant import VOCAB_PREFIX

class Vocab(BaseVocab):
    def build_vocab(self):
        counter = Counter(self.data)
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class MultiVocab(BaseMultiVocab):
    @classmethod
    def load_state_dict(cls, state_dict):
        new = cls()
        for k,v in state_dict.items():
            new[k] = Vocab.load_state_dict(v)
        return new
