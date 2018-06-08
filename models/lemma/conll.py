"""
A wrapper/loader for the official conll-u format files.
"""
import os

FIELD_NUM = 10

WORD_IDX = 1
LEMMA_IDX = 2

class CoNLLFile():
    def __init__(self, filename):
        if not os.path.exists(filename):
            raise Exception("File not found at: " + filename)
        self._file = filename

    def load_conll(self):
        """
        Load data into a list of sentences, where each sentence is a list of lines,
        and each line is a list of conllu fields.
        """
        sents, cache = [], []
        with open(self.file) as infile:
            while True:
                line = infile.readline()
                if len(line) == 0:
                    break
                line = line.strip()
                if len(line) == 0:
                    if len(cache) > 0:
                        sents.append(cache)
                        cache = []
                else:
                    if line.startswith('#'): # skip comment line
                        continue
                    array = line.split('\t')
                    assert len(array) == FIELD_NUM
                    cache += [array]
            if len(cache) > 0:
                sents.append(cache)
        return sents
    
    @property
    def file(self):
        return self._file

    @property
    def sents(self):
        if not hasattr(self, '_sents'):
            self._sents = self.load_conll()
        return self._sents

    def __len__(self):
        return len(self.sents)

    @property
    def num_words(self):
        """ Num of total words, after multi-word expansion."""
        if not hasattr(self, '_num_words'):
            n = 0
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]:
                        n += 1
            self._num_words = n
        return self._num_words

    def get_words(self):
        """ Get all words (after multi-word expansion) in a huge list, Note that
        the (original) tokens that are expanded will be skipped."""
        words = []
        for sent in self.sents:
            for ln in sent:
                if '-' in ln[0]: # skip
                    continue
                words += [ln[WORD_IDX]]
        return words

    def get_lemmas(self):
        """ Get all lemmas in a huge list, like the get_words() function. """
        lemmas = []
        for sent in self.sents:
            for ln in sent:
                if '-' in ln[0]:
                    continue
                lemmas += [ln[LEMMA_IDX]]
        return lemmas

    def get_words_and_lemmas(self):
        pairs = []
        for sent in self.sents:
            for ln in sent:
                if '-' in ln[0]:
                    continue
                pairs += [(ln[WORD_IDX], ln[LEMMA_IDX])]
        return pairs

    def write_conll_with_lemmas(self, lemmas, filename):
        """ Write a new conll file, but use the new lemmas to replace the old ones."""
        assert self.num_words == len(lemmas), "Num of lemmas does not match the number in original data file."
        idx = 0
        with open(filename, 'w') as outfile:
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]: # do not process if it is a mwt line
                        lm = lemmas[idx]
                        if len(lm) == 0:
                            lm = '_'
                        ln[LEMMA_IDX] = lm
                        idx += 1
                    print("\t".join(ln), file=outfile)
                print("", file=outfile)
        return
