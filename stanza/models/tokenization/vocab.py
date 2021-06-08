from collections import Counter
import re

from stanza.models.common.vocab import BaseVocab
from stanza.models.common.vocab import UNK, PAD

SPACE_RE = re.compile(r'\s')

class Vocab(BaseVocab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_replaces_spaces = any([self.lang.startswith(x) for x in ['zh', 'ja', 'ko']])

    def build_vocab(self):
        paras = self.data
        counter = Counter()
        for para in paras:
            for unit in para:
                normalized = self.normalize_unit(unit[0])
                counter[normalized] += 1

        self._id2unit = [PAD, UNK] + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def normalize_unit(self, unit):
        # Normalize minimal units used by the tokenizer
        return unit

    def normalize_token(self, token):
        token = SPACE_RE.sub(' ', token.lstrip())

        if self.lang_replaces_spaces:
            token = token.replace(' ', '')

        return token
