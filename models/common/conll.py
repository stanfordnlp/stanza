"""
A wrapper/loader for the official conll-u format files.
"""
import os

FIELD_NUM = 10

FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}

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

    def get(self, fields):
        """ Get fields from a list of field names. If only one field name is provided, return a list
        of that field; if more than one, return a list of list. Note that all returned fields are after
        multi-word expansion.
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."
        field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        results = []
        for sent in self.sents:
            for ln in sent:
                if '-' in ln[0]: # skip
                    continue
                if len(field_idxs) == 1:
                    results += [ln[field_idxs[0]]]
                else:
                    results += [[ln[fid] for fid in field_idxs]]
        return results
    
    def write_conll_with_lemmas(self, lemmas, filename):
        """ Write a new conll file, but use the new lemmas to replace the old ones."""
        assert self.num_words == len(lemmas), "Num of lemmas does not match the number in original data file."
        lemma_idx = FIELD_TO_IDX['lemma']
        idx = 0
        with open(filename, 'w') as outfile:
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]: # do not process if it is a mwt line
                        lm = lemmas[idx]
                        if len(lm) == 0:
                            lm = '_'
                        ln[lemma_idx] = lm
                        idx += 1
                    print("\t".join(ln), file=outfile)
                print("", file=outfile)
        return

    def get_mwt_expansions(self):
        word_idx = FIELD_TO_IDX['word']
        expansions = []
        src = ''
        dst = []
        for sent in self.sents:
            for ln in sent:
                if '.' in ln[0]:
                    # skip ellipsis
                    continue

                if '-' in ln[0]:
                    mwt_begin, mwt_end = [int(x) for x in ln[0].split('-')]
                    src = ln[word_idx]
                    continue

                if mwt_begin <= int(ln[0]) < mwt_end:
                    dst += [ln[word_idx]]
                elif int(ln[0]) == mwt_end:
                    dst += [ln[word_idx]]
                    expansions += [src, ' '.join(dst)]

        return expansions

    def get_mwt_expansion_cands(self):
        word_idx = FIELD_TO_IDX['word']
        cands = []
        for sid, sent in enumerate(self.sents):
            for wid, ln in enumerate(sent):
                if ln[-1] == "MWT=Yes":
                    cands += [(sid, wid), ln[word_idx]]

        return cands

    def write_conll_with_mwt_expansions(self, expansions, filename):
        idx = 0
        with open(filename, 'w') as outfile:
            for sid, sent in enumerate(self.sents):
                for wid, ln in enumerate(sent):
                    idx += 1
                    if (sid, wid) not in expansions:
                        print("{}\t{}".format(idx, "\t".join(ln[1:])), file=outfile)
                    else:
                        # print MWT expansion
                        expanded = expansion[(sid, wid)].split(' ')
                        endidx = idx + len(expanded) - 1

                        print("{}-{}\t{}".format(idx, endidx, "\t".join(ln[1:])), file=outfile)
                        for e_i, e_word in enumerate(expanded):
                            print("{}\t{}{}".format(idx + e_i, e_word, "\t_" * 8), file=outfile)

                print("", file=outfile)
                idx = 0
        return
