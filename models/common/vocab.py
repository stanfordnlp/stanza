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

class Vocab:
    def __init__(self, filename, data, lang, idx=0, cutoff=0, lower=False):
        self.filename = filename
        self.data = data
        self.lang = lang
        self.idx = idx
        self.cutoff = cutoff
        self.lower = lower
        if filename is not None:
            if os.path.exists(self.filename):
                self.load()
            else:
                self.build_vocab()
                self.save()
        else:
            self.build_vocab()

    def load(self):
        with open(self.filename, 'rb') as f:
            self._id2unit = pickle.load(f)
            self._unit2id = pickle.load(f)

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self._id2unit, f)
            pickle.dump(self._unit2id, f)

    def build_vocab(self):
        raise NotImplementedError()

    def normalize_unit(self, unit):
        if self.lower:
            return unit.lower()
        return unit

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id['<UNK>']

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
        return key in self._unit2id

    @property
    def size(self):
        return len(self)

class CompositeVocab(Vocab):
    def __init__(self, filename, data, lang, idx=0, sep="", keyed=False):
        self.sep = sep
        self.keyed = keyed
        super().__init__(filename, data, lang, idx=idx)

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
            #parts = dict([[x, y.split(',')] for x, y in parts])

            # Just treat multi-valued properties values as one possible value
            parts = dict(parts)
        elif unit == '_':
            parts = []
        return parts

    def unit2id(self, unit):
        parts = self.unit2parts(unit)
        if self.keyed:
            #return [[self._unit2id[k][x] for x in parts[k]] if k in parts else [PAD_ID] for k in self._unit2id]
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
                    #for v in parts[key]:
                    #    if v not in self._id2unit[key]:
                    #        self._id2unit[key].append(v)
        
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
