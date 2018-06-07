from collections import Counter

from models.common.vocab import Vocab as BaseVocab

class Vocab(BaseVocab):
    def build_vocab(self):
        paras = self.data
        counter = Counter()
        for para in paras:
            for unit in para:
                normalized = self.normalize_unit(unit[0])
                counter[normalized] += 1

        self._id2unit = ['<PAD>', '<UNK>'] + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def normalize_unit(self, unit):
        # Normalize minimal units used by the tokenizer
        # For Vietnamese this means a syllable, for other languages this means a character
        normalized = unit
        if self.lang == 'vi':
            normalized = normalized.lstrip()

        return normalized

    def normalize_token(self, token):
        token = token.lstrip().replace('\n', ' ')

        if self.lang in ['zh', 'ja', 'ko']:
            token = token.replace(' ', '')

        return token
