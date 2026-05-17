from collections import Counter, OrderedDict

from stanza.models.common.vocab import BaseVocab, BaseMultiVocab, CharVocab
from stanza.models.common.vocab import CompositeVocab, VOCAB_PREFIX, EMPTY, EMPTY_ID, PAD_ID

class WordVocab(BaseVocab):
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False, ignore=None):
        self.ignore = ignore if ignore is not None else []
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

    def __iter__(self):
        # the EMPTY shenanigans above make list() look really weird
        # when using the __len__ / __getitem__ paradigm,
        # but yielding items like this works fine
        for x in self._id2unit:
            yield x

    def __str__(self):
        return "<{}: {}>".format(type(self), ",".join("|%s|" % x for x in self._id2unit))

class XPOSVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="", keyed=False):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class FeatureVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="|", keyed=True):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

# Reserved ids for the binary "starts a known fixed multi-word expression" flag.
# PAD_ID (0) is reserved for padding so that the embedding's padding_idx behaves like the rest of the model.
FIXED_NO_ID = 1
FIXED_YES_ID = 2
FIXED_NUM_IDS = 3

class FixedExpressionVocab:
    """A lexicon of multi-word fixed expressions.

    Each entry is a tuple of lowercased word forms (e.g., ``('de', 'hecho')``).
    Used by the POS tagger as a per-token binary input feature signalling
    whether a token is the start of a known fixed multi-word expression.
    Useful for predicting the ExtPos UFeat when the data is highly imbalanced.
    """

    def __init__(self, expressions=None, lowercase=True):
        self.lowercase = lowercase
        if expressions is None:
            self.expressions = set()
        else:
            self.expressions = set(tuple(e) for e in expressions)
        self.max_len = max((len(e) for e in self.expressions), default=0)

    def __len__(self):
        return len(self.expressions)

    def __contains__(self, ngram):
        return tuple(ngram) in self.expressions

    def add(self, ngram):
        ngram = tuple(w.lower() if self.lowercase else w for w in ngram)
        if len(ngram) < 2:
            return
        self.expressions.add(ngram)
        if len(ngram) > self.max_len:
            self.max_len = len(ngram)

    def update(self, ngrams):
        for ng in ngrams:
            self.add(ng)

    def _norm(self, word):
        return word.lower() if self.lowercase else word

    def map(self, words):
        """Return a list of ids (one per token): PAD_ID for empty inputs is not
        produced here; instead each position gets FIXED_NO_ID (1) or FIXED_YES_ID (2)
        depending on whether it begins any known fixed expression."""
        n = len(words)
        if n == 0 or not self.expressions or self.max_len < 2:
            return [FIXED_NO_ID] * n
        norm = [self._norm(w) for w in words]
        out = [FIXED_NO_ID] * n
        for i in range(n):
            limit = min(self.max_len, n - i)
            for j in range(2, limit + 1):
                if tuple(norm[i:i + j]) in self.expressions:
                    out[i] = FIXED_YES_ID
                    break
        return out

    def state_dict(self):
        # Serialise the expressions as a sorted list of tuples for stable IO.
        return OrderedDict([
            ('lowercase', self.lowercase),
            ('expressions', sorted(self.expressions)),
            ('max_len', self.max_len),
        ])

    @classmethod
    def load_state_dict(cls, state_dict):
        new = cls(expressions=state_dict.get('expressions', []),
                  lowercase=state_dict.get('lowercase', True))
        # max_len is derived but trust the saved one if present (e.g. for empty vocabs)
        new.max_len = state_dict.get('max_len', new.max_len)
        return new

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
                      'FeatureVocab': FeatureVocab,
                      'FixedExpressionVocab': FixedExpressionVocab}
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict['_key2class']
        for k,v in state_dict.items():
            if k == '_key2class':
                continue
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new

