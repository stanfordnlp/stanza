from collections import Counter, OrderedDict

from stanfordnlp.models.common.vocab import BaseVocab, BaseMultiVocab
from stanfordnlp.models.common.vocab import CompositeVocab, VOCAB_PREFIX, EMPTY, EMPTY_ID

class CharVocab(BaseVocab):
    def build_vocab(self):
        if self.predefined:
            self._id2unit = ["<UNK>", " ", "e", "i", "n", "a", "t", "o", "r", "s", "l", "d", "u", "c", "m", "h", "p", "g", "\n", ".", "z", "w", "b", "v", "k", "f", ",", "y", "j", "\u00e9", "E", "I", "S", "A", "-", "0", "1", "'", "D", "C", "P", "T", "q", "N", "M", "2", "\u0142", "L", "R", "B", ")", "(", "\u0119", "W", "O", "\u0105", "H", "G", "U", "x", "\u00f3", "?", "\u017c", ":", "3", "9", "5", "V", "F", "4", "\u00e0", "\u015b", "K", "\u00fc", "\u00e4", "/", "\u00e8", "6", "J", "8", "\u2019", "\"", "7", "Z", "\u0107", "!", "\u00f6", "%", ";", "\u00df", "\u0144", "Y", "\u00ea", "\u2013", "Q", "\u00ad", "\u00f9", "\u00eb", "&", "\u0e00", "\u00f2", "X", "+", "\u2014", "\u201d", "\u201c", "\u201e", "\u00e7", "\u017a", "\u2022", "\u00f4", "_", "\u00bb", "\u00b0", "\u00c9", "\u00ec", "#", "\u00ab", "]", "[", "*", "\u00ef", "\u00dc", "\u00ee", "\u00e2", "<", "=", ">", "\u00a0", "\u015a", "\u00fb", "\u00c8", "\u00c4", "\u00e1", "\u093e", "\u0153", "\u2265", "\u0915", "\u00d6", "\u017b", "\u0930", "\u0141", "\u00ed", "\u0947", "$", "\u2026", "\u00c0", "\u00b7", "\u00c3", "\u2018", "\u094d", "\u093f", "\u00c7", "@", "\u0938", "\ufffd", "\u0928", "\u2193", "\u0902", "\u03bb", "\u0924", "\u0940", "\u0939", "\u00b5", "\u03b1", "\u094b", "\u092e", "\u2212", "\u00d3", "\u0104", "\u092f", "{", "\u00a9", "\u0430", "\u043b", "\u0932", "}", "\u0118", "\u092a", "\u03b7", "\u03bf", "\u0935", "\u0438", "\u03bc", "\u0440", "\u00b4", "\u00f1", "\u00bc", "~", "\u20ac", "\u00ac", "\u2191", "\u0106", "`", "\u25a0", "\u0435", "\u00a8", "\\", "\u0926", "\u00b2", "\u0395", "\u00a7", "\u25ba", "\u03c2", "\u091c", "\u00b1", "\u03c1", "\u03a4", "\u00a1", "\u0948", "\u00ba", "\u03b9", "\u043e", "\u092c", "\u044f", "\u03b4", "^", "\u0917", "\u00a4", "\u2264", "|", "\u0411", "\u0433", "\u0101", "\u014d", "\u266a", "\u03c0", "\u010d", "\u00cd", "\u03ac", "\u044a", "\u0161", "\u00fa", "\u03c4", "\u0941", "\u0442", "\u220f", "\u0391", "\u039a", "\u0392", "\u0905", "\u00e5", "\u010c", "\u00e6", "\u0936", "\u043d", "\u2020", "\u03bd", "\u00b3", "\u090f", "\u00ae", "\u00ca", "\u2122", "\u0441", "\u00f0", "\u03cd", "\u2500", "\u091a", "\u017e", "\u00f8", "\u00e3", "\u0942", "\u092d", "\u0143", "\u0131", "\u2194", "\u03a0", "\u03c5", "\u0927", "\u03a1", "\u03b5", "\u0964", "\u2192", "\u0937", "\u0925", "\u03af", "\u00aa", "\u039d", "\u03ba", "\u02dd", "\u201a", "\u043a", "\u039b", "\u03b2", "\u0422", "\u25cf", "\u0914", "\u00c2", "\u00cc", "\u0909", "\u03c3", "\u03c7", "\uf0b7", "\u0432", "\u0907", "\u03b3", "\u0393", "\u093c", "\u091f", "\u0399"]
            self._unit2id = {w:i for i, w in enumerate(self._id2unit)}
        else:
            if type(self.data[0][0]) is list: # general data from DataLoader
                counter = Counter([c for sent in self.data for w in sent for c in w[self.idx]])
            else: # data from Char LM
                counter = Counter([c for sent in self.data for c in sent])
            self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
            self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class WordVocab(BaseVocab):
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False, ignore=[]):
        self.ignore = ignore
        super().__init__(data, lang=lang, idx=idx, cutoff=cutoff, lower=lower)
        self.state_attrs += ['ignore']

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
    def __init__(self, data=None, lang="", idx=0, sep="", keyed=False):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class FeatureVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="|", keyed=True):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

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
                'WordVocab': WordVocab,
                'XPOSVocab': XPOSVocab,
                'FeatureVocab': FeatureVocab}
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop('_key2class')
        for k,v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new

