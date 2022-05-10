from collections import Counter, OrderedDict

from stanza.models.common.vocab import BaseVocab, BaseMultiVocab, CharVocab
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.common.pretrain import PretrainedWordVocab
from stanza.models.pos.vocab import WordVocab

class TagVocab(BaseVocab):
    """ A vocab for the output tag sequence. """
    def build_vocab(self):
        counter = Counter([w[self.idx] for sent in self.data for w in sent])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class MultiVocab(BaseMultiVocab):
    def state_dict(self):
        """ Also save a vocab name to class name mapping in state dict. """
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            key2class[k] = type(v).__name__
        state['_key2class'] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        class_dict = {'CharVocab': CharVocab,
                      'PretrainedWordVocab': PretrainedWordVocab,
                      'TagVocab': TagVocab,
                      'WordVocab': WordVocab}
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop('_key2class')
        for k,v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new

