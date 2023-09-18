from collections import Counter, OrderedDict

from stanza.models.common.vocab import BaseVocab, BaseMultiVocab, CharVocab, CompositeVocab
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.common.pretrain import PretrainedWordVocab
from stanza.models.pos.vocab import WordVocab

class TagVocab(BaseVocab):
    """ A vocab for the output tag sequence. """
    def build_vocab(self):
        counter = Counter([w[self.idx] for sent in self.data for w in sent])

        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

def convert_tag_vocab(state_dict):
    if state_dict['lower']:
        raise AssertionError("Did not expect an NER vocab with 'lower' set to True")
    items = state_dict['_id2unit'][len(VOCAB_PREFIX):]
    # this looks silly, but the vocab builder treats this as words with multiple fields
    # (we set it to look for field 0 with idx=0)
    # and then the label field is expected to be a list or tuple of items
    items = [[[[x]]] for x in items]
    vocab = CompositeVocab(data=items, lang=state_dict['lang'], idx=0, sep=None)
    if len(vocab._id2unit[0]) != len(state_dict['_id2unit']):
        raise AssertionError("Failed to construct a new vocab of the same length as the original")
    if vocab._id2unit[0] != state_dict['_id2unit']:
        raise AssertionError("Failed to construct a new vocab in the same order as the original")
    return vocab

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
        class_dict = {'CharVocab': CharVocab.load_state_dict,
                      'PretrainedWordVocab': PretrainedWordVocab.load_state_dict,
                      'TagVocab': convert_tag_vocab,
                      'CompositeVocab': CompositeVocab.load_state_dict,
                      'WordVocab': WordVocab.load_state_dict}
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop('_key2class')
        for k,v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname](v)
        return new

