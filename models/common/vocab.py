from collections import Counter
import os
import pickle

class Vocab:
    def __init__(self, filename, data, lang):
        self.filename = filename
        self.data = data
        self.lang = lang
        if os.path.exists(self.filename):
            self.load()
        else:
            self.build_vocab()
            self.save()

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
        return unit

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id['<UNK>']

    def id2unit(self, id):
        return self._id2unit[id]

    def __len__(self):
        return len(self._id2unit)
