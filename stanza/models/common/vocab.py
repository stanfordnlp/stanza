from copy import copy
from collections import Counter, OrderedDict
import os
import pickle

PAD = '<PAD>'
PAD_ID = 0
UNK = '<UNK>'
UNK_ID = 1
EMPTY = '<EMPTY>'
EMPTY_ID = 2
ROOT = '<ROOT>'
ROOT_ID = 3
VOCAB_PREFIX = [PAD, UNK, EMPTY, ROOT]

class BaseVocab:
    """ A base class for common vocabulary operations. Each subclass should at least 
    implement its own build_vocab() function."""
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False):
        self.data = data
        self.lang = lang
        self.idx = idx
        self.cutoff = cutoff
        self.lower = lower
        if data is not None:
            self.build_vocab()
        self.state_attrs = ['lang', 'idx', 'cutoff', 'lower', '_unit2id', '_id2unit']

    def build_vocab(self):
        raise NotImplementedError()

    def state_dict(self):
        """ Returns a dictionary containing all states that are necessary to recover
        this vocab. Useful for serialization."""
        state = OrderedDict()
        for attr in self.state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        """ Returns a new Vocab instance constructed from a state dict. """
        new = cls()
        for attr, value in state_dict.items():
            setattr(new, attr, value)
        return new

    def normalize_unit(self, unit):
        # be sure to look in subclasses for other normalization being done
        # especially PretrainWordVocab
        if unit is None:
            return unit
        if self.lower:
            return unit.lower()
        return unit

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id[UNK]

    def id2unit(self, id):
        return self._id2unit[id]

    def map(self, units):
        return [self.unit2id(x) for x in units]

    def unmap(self, ids):
        return [self.id2unit(x) for x in ids]

    def __len__(self):
        return len(self._id2unit)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.unit2id(key)
        elif isinstance(key, int) or isinstance(key, list):
            return self.id2unit(key)
        else:
            raise TypeError("Vocab key must be one of str, list, or int")

    def __contains__(self, key):
        return self.normalize_unit(key) in self._unit2id

    @property
    def size(self):
        return len(self)

class CompositeVocab(BaseVocab):
    ''' Vocabulary class that handles parsing and printing composite values such as
    compositional XPOS and universal morphological features (UFeats).

    Two key options are `keyed` and `sep`. `sep` specifies the separator used between
    different parts of the composite values, which is `|` for UFeats, for example.
    If `keyed` is `True`, then the incoming value is treated similarly to UFeats, where
    each part is a key/value pair separated by an equal sign (`=`). There are no inherit
    order to the keys, and we sort them alphabetically for serialization and deserialization.
    Whenever a part is absent, its internal value is a special `<EMPTY>` symbol that will
    be treated accordingly when generating the output. If `keyed` is `False`, then the parts
    are treated as positioned values, and `<EMPTY>` is used to pad parts at the end when the
    incoming value is not long enough.'''

    def __init__(self, data=None, lang="", idx=0, sep="", keyed=False):
        self.sep = sep
        self.keyed = keyed
        super().__init__(data, lang, idx=idx)
        self.state_attrs += ['sep', 'keyed']

    def unit2parts(self, unit):
        # unpack parts of a unit
        if self.sep == "":
            parts = [x for x in unit]
        else:
            parts = unit.split(self.sep)
        if self.keyed:
            if len(parts) == 1 and parts[0] == '_':
                return dict()
            parts = [x.split('=') for x in parts]

            # Just treat multi-valued properties values as one possible value
            parts = dict(parts)
        elif unit == '_':
            parts = []
        return parts

    def unit2id(self, unit):
        parts = self.unit2parts(unit)
        if self.keyed:
            # treat multi-valued properties as singletons
            return [self._unit2id[k].get(parts[k], UNK_ID) if k in parts else EMPTY_ID for k in self._unit2id]
        else:
            return [self._unit2id[i].get(parts[i], UNK_ID) if i < len(parts) else EMPTY_ID for i in range(len(self._unit2id))]

    def id2unit(self, id):
        items = []
        for v, k in zip(id, self._id2unit.keys()):
            if v == EMPTY_ID: continue
            if self.keyed:
                items.append("{}={}".format(k, self._id2unit[k][v]))
            else:
                items.append(self._id2unit[k][v])
        res = self.sep.join(items)
        if res == "":
            res = "_"
        return res

    def build_vocab(self):
        allunits = [w[self.idx] for sent in self.data for w in sent]
        if self.keyed:
            self._id2unit = dict()

            for u in allunits:
                parts = self.unit2parts(u)
                for key in parts:
                    if key not in self._id2unit:
                        self._id2unit[key] = copy(VOCAB_PREFIX)

                    # treat multi-valued properties as singletons
                    if parts[key] not in self._id2unit[key]:
                        self._id2unit[key].append(parts[key])

            # special handle for the case where upos/xpos/ufeats are always empty
            if len(self._id2unit) == 0:
                self._id2unit['_'] = copy(VOCAB_PREFIX) # use an arbitrary key

        else:
            self._id2unit = dict()

            allparts = [self.unit2parts(u) for u in allunits]
            maxlen = max([len(p) for p in allparts])

            for parts in allparts:
                for i, p in enumerate(parts):
                    if i not in self._id2unit:
                        self._id2unit[i] = copy(VOCAB_PREFIX)
                    if i < len(parts) and p not in self._id2unit[i]:
                        self._id2unit[i].append(p)

            # special handle for the case where upos/xpos/ufeats are always empty
            if len(self._id2unit) == 0:
                self._id2unit[0] = copy(VOCAB_PREFIX) # use an arbitrary key

        self._id2unit = OrderedDict([(k, self._id2unit[k]) for k in sorted(self._id2unit.keys())])
        self._unit2id = {k: {w:i for i, w in enumerate(self._id2unit[k])} for k in self._id2unit}

    def lens(self):
        return [len(self._unit2id[k]) for k in self._unit2id]

class BaseMultiVocab:
    """ A convenient vocab container that can store multiple BaseVocab instances, and support 
    safe serialization of all instances via state dicts. Each subclass of this base class 
    should implement the load_state_dict() function to specify how a saved state dict 
    should be loaded back."""
    def __init__(self, vocab_dict=None):
        self._vocabs = OrderedDict()
        if vocab_dict is None:
            return
        # check all values provided must be a subclass of the Vocab base class
        assert all([isinstance(v, BaseVocab) for v in vocab_dict.values()])
        for k, v in vocab_dict.items():
            self._vocabs[k] = v

    def __setitem__(self, key, item):
        self._vocabs[key] = item

    def __getitem__(self, key):
        return self._vocabs[key]

    def state_dict(self):
        """ Build a state dict by iteratively calling state_dict() of all vocabs. """
        state = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        """ Construct a MultiVocab by reading from a state dict."""
        raise NotImplementedError



class CharVocab(BaseVocab):
    def build_vocab(self):
        if type(self.data[0][0]) is list: # general data from DataLoader
            counter = Counter([c for sent in self.data for w in sent for c in w[self.idx]])
            for k in list(counter.keys()):
                if counter[k] < self.cutoff:
                    del counter[k]
        else: # special data from Char LM
            counter = Counter([c for sent in self.data for c in sent])
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: (counter[k], k), reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

