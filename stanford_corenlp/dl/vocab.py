"""
Vocabulary module for conversion between word tokens and numerical indices.
"""
__author__ = 'victor, kelvinguu'

import math
import numpy as np
import zipfile

from abc import ABCMeta, abstractmethod
from copy import copy
from collections import Counter, namedtuple, OrderedDict
from stanford_corenlp.util import get_data_or_download

class BaseVocab(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def word2index(self, w):
        """Convert string to integer."""
        raise NotImplementedError

    @abstractmethod
    def index2word(self, i):
        """Convert integer to string."""
        raise NotImplementedError

    def words2indices(self, words):
        """
        Convert a list of words into a list of indices.

        :param words: an iterable of words to map to indices.
        :return: the corresponding indices for each word.
        """
        return [self.word2index(w) for w in words]

    def indices2words(self, indices):
        """
        Convert a list of indices into a list of words.

        :param words: an iterable of ints to map to words.
        :return: the corresponding words for each int.
        """
        return [self.index2word(i) for i in indices]


class Vocab(BaseVocab, OrderedDict):
    """A mapping between words and numerical indices. This class is used to facilitate the creation of word embedding matrices.

    Example:

    .. code-block:: python

        v = Vocab('***UNK***')
        indices = v.update("I'm a list of words".split())
        print('indices')

    NOTE: UNK is always represented by the 0 index.
    """

    def __init__(self, unk):
        """Construct a Vocab object.

        :param unk: string to represent the unknown word (UNK). It is always represented by the 0 index.
        """
        super(Vocab, self).__init__()
        self._counts = Counter()
        self._unk = unk

        # assign an index for UNK
        self.add(self._unk, count=0)

    def __getitem__(self, word):
        """Get the index for a word.

        If the word is unknown, the index for UNK is returned.
        """
        return self.get(word, 0)

    def __setitem__(self, key, value, **kwargs):
        raise NotImplementedError('Use add method instead.')

    def __str__(self):
        return 'Vocab(%d words)' % len(self)

    def __eq__(self, other):
        if isinstance(other, Vocab):
            return super(Vocab, self).__eq__(other)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def add(self, word, count=1):
        """Add a word to the vocabulary and return its index.

        :param word: word to add to the dictionary.

        :param count: how many times to add the word.

        :return: index of the added word.

        WARNING: this function assumes that if the Vocab currently has N words, then
        there is a perfect bijection between these N words and the integers 0 through N-1.
        """
        if word not in self:
            super(Vocab, self).__setitem__(word, len(self))
        self._counts[word] += count
        return self[word]

    def update(self, words):
        """
        Add an iterable of words to the Vocabulary.

        :param words: an iterable of words to add. Each word will be added once.

        :return: the corresponding list of indices for each word.
        """
        return [self.add(w) for w in words]

    def word2index(self, w):
        return self[w]

    def index2word(self, i):
        return self._index2word[i]

    def freeze(self):
        return FrozenVocab(self)

    def count(self, w):
        """Get the count for a word.

        :param w: a string
        """
        return self._counts[w]

    def subset(self, words):
        """Get a new Vocab containing only the specified subset of words.

        If w is in words, but not in the original vocab, it will NOT be in the subset vocab.
        Indices will be in the order of `words`. Counts from the original vocab are preserved.

        :return (Vocab): a new Vocab object
        """
        v = self.__class__(unk=self._unk)
        unique = lambda seq: len(set(seq)) == len(seq)
        assert unique(words)
        for w in words:
            if w in self:
                v.add(w, count=self.count(w))
        return v

    @property
    def _index2word(self):
        """Mapping from indices to words.

        WARNING: this may go out-of-date, because it is a copy, not a view into the Vocab.

        :return: a list of strings
        """
        # TODO(kelvinguu): it would be nice to just use `dict.viewkeys`, but unfortunately those are not indexable

        compute_index2word = lambda: self.keys()  # this works because self is an OrderedDict

        # create if it doesn't exist
        try:
            self._index2word_cache
        except AttributeError:
            self._index2word_cache = compute_index2word()

        # update if it is out of date
        if len(self._index2word_cache) != len(self):
            self._index2word_cache = compute_index2word()

        return self._index2word_cache

    def prune_rares(self, cutoff=2):
        """
        returns a **new** `Vocab` object that is similar to this one but with rare words removed.
        Note that the indices in the new `Vocab` will be remapped (because rare words will have been removed).

        :param cutoff: words occuring less than this number of times are removed from the vocabulary.

        :return: A new, pruned, vocabulary.

        NOTE: UNK is never pruned.
        """
        keep = lambda w: self.count(w) >= cutoff or w == self._unk
        return self.subset([w for w in self if keep(w)])

    def sort_by_decreasing_count(self):
        """Return a **new** `Vocab` object that is ordered by decreasing count.

        The word at index 1 will be most common, the word at index 2 will be
        next most common, and so on.

        :return: A new vocabulary sorted by decreasing count.

        NOTE: UNK will remain at index 0, regardless of its frequency.
        """
        words = [w for w, ct in self._counts.most_common()]
        v = self.subset(words)
        return v

    @classmethod
    def from_dict(cls, word2index, unk, counts=None):
        """Create Vocab from an existing string to integer dictionary.

        All counts are set to 0.

        :param word2index: a dictionary representing a bijection from N words to the integers 0 through N-1.
                UNK must be assigned the 0 index.

        :param unk: the string representing unk in word2index.

        :param counts: (optional) a Counter object mapping words to counts

        :return: a created vocab object.
        """
        try:
            if word2index[unk] != 0:
                raise ValueError('unk must be assigned index 0')
        except KeyError:
            raise ValueError('word2index must have an entry for unk.')

        # check that word2index is a bijection
        vals = set(word2index.values())  # unique indices
        n = len(vals)

        bijection = (len(word2index) == n) and (vals == set(range(n)))
        if not bijection:
            raise ValueError('word2index is not a bijection between N words and the integers 0 through N-1.')

        # reverse the dictionary
        index2word = {idx: word for word, idx in word2index.iteritems()}

        vocab = cls(unk=unk)
        for i in range(n):
            vocab.add(index2word[i])

        if counts:
            matching_entries = set(word2index.keys()) == set(counts.keys())
            if not matching_entries:
                raise ValueError('entries of word2index do not match counts (did you include UNK?)')
            vocab._counts = counts

        return vocab

    def to_file(self, f):
        """Write vocab to a file.

        :param (file) f: a file object, e.g. as returned by calling `open`

        File format:
            word0<TAB>count0
            word1<TAB>count1
            ...

        word with index 0 is on the 0th line and so on...
        """
        for word in self._index2word:
            count = self._counts[word]
            f.write(u'{}\t{}\n'.format(word, count).encode('utf-8'))

    @classmethod
    def from_file(cls, f):
        """Load vocab from a file.

        :param (file) f: a file object, e.g. as returned by calling `open`
        :return: a vocab object. The 0th line of the file is assigned to index 0, and so on...
        """
        word2index = {}
        counts = Counter()
        for i, line in enumerate(f):
            word, count_str = line.split('\t')
            word = word.decode('utf-8')
            word2index[word] = i
            counts[word] = float(count_str)
            if i == 0:
                unk = word
        return cls.from_dict(word2index, unk, counts)


class FrozenVocab(BaseVocab):
    def __init__(self, vocab):
        self._word2index = dict(vocab)  # make a copy
        self._index2word = copy(vocab._index2word)
        # since this vocab is frozen, we do not need to worry about
        # word2index and index2word becoming inconsistent

    def word2index(self, w):
        return self._word2index.get(w, 0)

    def index2word(self, i):
        return self._index2word[i]


class EmbeddedVocab(Vocab):
    def get_embeddings(self):
        """
        :return: the embedding matrix for this vocabulary object.
        """
        raise NotImplementedError()

    def backfill_unk_emb(self, E, filled_words):
        """ Backfills an embedding matrix with the embedding for the unknown token.

        :param E: original embedding matrix of dimensions `(vocab_size, emb_dim)`.
        :param filled_words: these words will not be backfilled with unk.

        NOTE: this function is for internal use.
        """
        unk_emb = E[self[self._unk]]
        for i, word in enumerate(self):
            if word not in filled_words:
                E[i] = unk_emb


class SennaVocab(EmbeddedVocab):
    """
    Vocab object with initialization from Senna by Collobert et al.

    Reference: http://ronan.collobert.com/senna
    """

    embeddings_url = 'https://github.com/baojie/senna/raw/master/embeddings/embeddings.txt'
    words_url = 'https://raw.githubusercontent.com/baojie/senna/master/hash/words.lst'
    n_dim = 50

    def __init__(self, unk='UNKNOWN'):
        super(SennaVocab, self).__init__(unk=unk)

    @classmethod
    def gen_word_list(cls, fname):
        with open(fname) as f:
            for line in f:
                yield line.rstrip("\n\r")

    @classmethod
    def gen_embeddings(cls, fname):
        with open(fname) as f:
            for line in f:
                yield np.fromstring(line, sep=' ')

    def get_embeddings(self, rand=None, dtype='float32'):
        """
        Retrieves the embeddings for the vocabulary.

        :param rand: Random initialization function for out-of-vocabulary words. Defaults to `np.random.uniform(-0.1, 0.1, size=shape)`.
        :param dtype: Type of the matrix.
        :return: embeddings corresponding to the vocab instance.

        NOTE: this function will download potentially very large binary dumps the first time it is called.
        """
        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        embeddings = get_data_or_download('senna', 'embeddings.txt', self.embeddings_url)
        words = get_data_or_download('senna', 'words.lst', self.words_url)

        E = rand((len(self), self.n_dim)).astype(dtype)

        seen = []
        for word_emb in zip(self.gen_word_list(words), self.gen_embeddings(embeddings)):
            w, e = word_emb
            if w in self:
                seen += [w]
                E[self[w]] = e
        self.backfill_unk_emb(E, set(seen))
        return E


class GloveVocab(EmbeddedVocab):
    """
    Vocab object with initialization from GloVe by Pennington et al.

    Reference: http://nlp.stanford.edu/projects/glove
    """

    GloveSetting = namedtuple('GloveSetting', ['url', 'n_dims', 'size', 'description'])
    settings = {
        'common_crawl_48': GloveSetting('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                        [300], '1.75GB', '48B token common crawl'),
        'common_crawl_840': GloveSetting('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                         [300], '2.03GB', '840B token common crawl'),
        'twitter': GloveSetting('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                [25, 50, 100, 200], '1.42GB', '27B token twitter'),
        'wikipedia_gigaword': GloveSetting('http://nlp.stanford.edu/data/glove.6B.zip',
                                           [50, 100, 200, 300], '822MB', '6B token wikipedia 2014 + gigaword 5'),
    }

    def __init__(self, unk='UNKNOWN'):
        super(GloveVocab, self).__init__(unk=unk)

    def get_embeddings(self, rand=None, dtype='float32', corpus='common_crawl_48', n_dim=300, load_all=True):
        """
        Retrieves the embeddings for the vocabulary.

        :param rand: Random initialization function for out-of-vocabulary words. Defaults to `np.random.uniform(-0.1, 0.1, size=shape)`.
        :param dtype: Type of the matrix.
        :param corpus: Corpus to use. Please see `GloveVocab.settings` for available corpus.
        :param n_dim: dimension of vectors to use. Please see `GloveVocab.settings` for available corpus.
        :return: embeddings corresponding to the vocab instance.

        NOTE: this function will download potentially very large binary dumps the first time it is called.
        """
        assert corpus in self.settings, '{} not in supported corpus {}'.format(corpus, self.settings.keys())
        self.n_dim, self.corpus, self.setting = n_dim, corpus, self.settings[corpus]
        assert n_dim in self.setting.n_dims, '{} not in supported dimensions {}'.format(n_dim, self.setting.n_dims)

        rand = rand if rand else lambda shape: np.random.uniform(-0.1, 0.1, size=shape)
        zip_file = get_data_or_download('glove', '{}.zip'.format(self.corpus), self.setting.url, size=self.setting.size)

        n_dim = str(self.n_dim)

        with zipfile.ZipFile(zip_file) as zf:
            # should be only 1 txt file
            names = [info.filename for info in zf.infolist() if
                     info.filename.endswith('.txt') and n_dim in info.filename]
            if not names:
                s = 'no .txt files found in zip file that matches {}-dim!'.format(n_dim)
                s += '\n available files: {}'.format(names)
                raise IOError(s)
            name = names[0]

            # if load_all option is True, add all words from vocab
            if load_all:
                all_words_in_vocab = []
                with zf.open(name) as f:
                    for line in f:
                        word = str(line, "utf-8").rstrip().split(' ')[0]
                        all_words_in_vocab.append(word)
                self.update(all_words_in_vocab)

            # set up embedding matrix
            E = rand((len(self), self.n_dim)).astype(dtype)

            # load embeddings for each token
            seen = []
            with zf.open(name) as f:
                for line in f:
                    toks = str(line, "utf-8").rstrip().split(' ')
                    word = toks[0]
                    if word in self:
                        seen += [word]
                        E[self[word]] = np.array([float(w) for w in toks[1:]], dtype=dtype)
            self.backfill_unk_emb(E, set(seen))
            return E


class CharIndex:
    """Class providing an index for chars"""

    def __init__(self):
        # special character to handle TensorFlow's need for vectors of same length
        self._ignore_char = "<ignore>"
        # character representing unknown characters
        self._unk_char = "<unk>"
        # special padding character for convolution
        self._pad_char = "<pad>"
        # mapping between chars and indices
        self._char_to_index = {"<ignore>": 0, "<pad>": 1, "<unk>": 2}
        # ordered list of chars
        self._char_list = ["<ignore>", "<pad>", "<unk>"]
        # initial char embeddings
        self._embeddings_array = None
        # tensor of char embeddings
        self._embeddings_tensor = None

    def add_chars_from_string(self, input_string):
        # go through all characters in string, add to index
        for char in input_string:
            if char not in self._char_to_index:
                self._char_list.append(char)
                self._char_to_index[char] = len(self.char_list) - 1

    def random_initialize_embeddings(self, char_embedding_size=30, embed_init_const=None):
        if not embed_init_const:
            embed_const = math.sqrt(float(3)/float(self.char_embedding_size))
        else:
            embed_const = embed_init_const
        self._embeddings_array = \
            np.random.uniform(-embed_const, embed_const, [len(self.char_list), char_embedding_size])

