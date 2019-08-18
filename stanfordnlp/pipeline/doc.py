"""
Basic data structures
"""

import io
import re

from stanfordnlp.models.common.conll import FIELD_TO_IDX as CONLLU_FIELD_TO_IDX

multi_word_token_line = re.compile("([0-9]+)\-([0-9]+)")

FIELD_NUM = 10
FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}

class Document:

    def __init__(self, input_src, from_file=False):
        # self._text = text
        # self._conll_file = None
        self._sentences = []
        self._num_words = 0
        self._input_src = input_src
        self._from_file = from_file
        if self._from_file:
            self.load_annotations(self._input_src)

    # @property
    # def conll_file(self):
    #     """ Access the CoNLLFile of this document. """
    #     return self._conll_file

    # @conll_file.setter
    # def conll_file(self, value):
    #     """ Set the document's CoNLLFile value. """
    #     self._conll_file = value

    # @property
    # def text(self):
    #     """ Access text of this document. Example: 'This is a sentence.'"""
    #     return self._text

    # @text.setter
    # def text(self, value):
    #     """ Set the document's text value. Example: 'This is a sentence.'"""
    #     self._text = value

    @property
    def sentences(self):
        """ Access list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value

    @property
    def num_words(self):
        """ Access number of words for this document. """
        return self._num_words

    @num_words.setter
    def num_words(self, value):
        """ Set the number of words for this document. """
        self._num_words = value

    def load_annotations(self, file_path):
        """ Integrate info from the CoNLLFile instance. """
        conll_sents = self.read_conll(file_path)
        # self._sentences = []
        # charoffset = 0
        for token_list in conll_sents:
            s = Sentence(token_list)
            # maxoffset = max(t.end_char_offset for t in s.tokens)
            # s.text = self.text[charoffset:maxoffset]
            # charoffset = maxoffset
            self.sentences.append(s)
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])
            

    def get(self, fields, as_sentences=False):
        """ Get fields from a list of field names. If only one field name is provided, return a list
        of that field; if more than one, return a list of list. Note that all returned fields are after
        multi-word expansion.
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."

        results = []
        for sentence in self.sentences:
            cursent = []
            for word in sentence.words:
                if len(fields) == 1:
                    cursent += [getattr(word, fields[0])]
                else:
                    cursent += [[getattr(word, field) for field in fields]]
            
            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

        # field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        # results = []
        # for sent in self.sents:
        #     cursent = []
        #     for ln in sent:
        #         if '-' in ln[0]: # skip
        #             continue
        #         if len(field_idxs) == 1:
        #             cursent += [ln[field_idxs[0]]]
        #         else:
        #             cursent += [[ln[fid] for fid in field_idxs]]

        #     if as_sentences:
        #         results.append(cursent)
        #     else:
        #         results += cursent
        # return results

    def set(self, fields, contents):
        """ Set fields based on contents. If only one field (singleton list) is provided, then a list of content will be expected; otherwise a list of list of contents will be expected.
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert isinstance(contents, list), "Must provide contents as a list (one item per line)."
        assert len(fields) >= 1, "Must have at least one field."
        assert self.num_words == len(contents), "Contents must have the same number as the original file."
        
        cidx = 0
        for sentence in self.sentences:
            for word in sentence.words:
                if len(fields) == 1:
                    setattr(word, fields[0], contents[cidx])
                else:
                    for field, content in zip(fields, contents[cidx]):
                        setattr(word, field, content)
                cidx += 1
        return
                       
        # field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        # cidx = 0
        # for sent in self.sents:
        #     for ln in sent:
        #         if '-' in ln[0]:
        #             continue
        #         if len(field_idxs) == 1:
        #             ln[field_idxs[0]] = contents[cidx]
        #         else:
        #             for fid, ct in zip(field_idxs, contents[cidx]):
        #                 ln[fid] = ct
        #         cidx += 1
        # return

    def conll_as_string(self):
        """ Return current conll contents as string
        """
        # <TODO>: bug existed, consider mwt
        return_string = ''
        fields = ['index', 'text', 'lemma', 'upos', 'xpos', 'feats', 'governor', 'dependency_relation', 'deps', 'misc']
        data = self.get(fields, as_sentences=True)
        for sentence in data:
            for word in sentence:
                ln = [str(field) if field is not None else '_' for field in word]
                return_string += ('\t'.join(ln) + '\n')
            return_string += '\n'
        return return_string

    def write_conll(self, file_path):
        conll_string = self.conll_as_string()
        with open(file_path, 'w') as outfile:
            outfile.write(conll_string)
        return

    def read_conll(self, file_path):
        """
        Load data into a list of sentences, where each sentence is a list of lines,
        and each line is a list of conllu fields.
        """
        sents, cache = [], []
        with open(file_path) as infile:
            for line in infile:
                line = line.strip()
                if len(line) == 0:
                    if len(cache) > 0:
                        sents.append(cache)
                        cache = []
                else:
                    if line.startswith('#'): # skip comment line
                        continue
                    array = line.split('\t')
                    if '.' in array[0]: # <TODO>: self.ignore_gapping
                        continue
                    assert len(array) == FIELD_NUM
                    cache += [array]
        if len(cache) > 0:
            sents.append(cache)
        return sents
                
    # def write_conll_to_file(self, file_path):
    #     """ Write conll contents to file. """
    #     self.conll_file.write_conll(file_path)

class Sentence:

    def __init__(self, tokens):
        self._tokens = []
        self._words = []
        self._process_tokens(tokens)
        self._dependencies = []
        # check if there is dependency info
        if self.words[0].dependency_relation is not None:
            self.build_dependencies()

    def _process_tokens(self, tokens):
        st, en = -1, -1
        for tok in tokens:
            m = multi_word_token_line.match(tok[CONLLU_FIELD_TO_IDX['id']])
            if m:
                st, en = int(m.group(1)), int(m.group(2))
                self._tokens.append(Token(tok))
            else:
                new_word = Word(tok)
                self._words.append(new_word)
                idx = int(tok[CONLLU_FIELD_TO_IDX['id']])
                if idx <= en:
                    self._tokens[-1].words.append(new_word)
                    new_word.parent_token = self._tokens[-1]
                else:
                    self.tokens.append(Token(tok, words=[new_word]))

    @property
    def dependencies(self):
        """ Access list of dependencies for this sentence. """
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        """ Set the list of dependencies for this sentence. """
        self._dependencies = value

    @property
    def tokens(self):
        """ Access list of tokens for this sentence. """
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        """ Set the list of tokens for this sentence. """
        self._tokens = value

    @property
    def words(self):
        """ Access list of words for this sentence. """
        return self._words

    @words.setter
    def words(self, value):
        """ Set the list of words for this sentence. """
        self._words = value

    @property
    def text(self):
        """ Access the original text for this sentence. """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the original text for this sentence. """
        self._text = value

    def build_dependencies(self):
        for word in self.words:
            if word.governor == 0:
                # make a word for the ROOT
                governor = Word(["0", "ROOT", "_", "_", "_", "_", "-1", "_", "_", "_", "_", "_"])
            else:
                # id is index in words list + 1
                governor = self.words[word.governor-1]
            self.dependencies.append((governor, word.dependency_relation, word))

    def print_dependencies(self, file=None):
        for dep_edge in self.dependencies:
            print((dep_edge[2].text, dep_edge[0].index, dep_edge[1]), file=file)

    def dependencies_string(self):
        dep_string = io.StringIO()
        self.print_dependencies(file=dep_string)
        return dep_string.getvalue().strip()

    def print_tokens(self, file=None):
        for tok in self.tokens:
            print(tok, file=file)

    def tokens_string(self):
        toks_string = io.StringIO()
        self.print_tokens(file=toks_string)
        return toks_string.getvalue().strip()

    def print_words(self, file=None):
        for word in self.words:
            print(word, file=file)

    def words_string(self):
        wrds_string = io.StringIO()
        self.print_words(file=wrds_string)
        return wrds_string.getvalue().strip()


class Token:

    def __init__(self, token_entry, words=None):
        self._index = token_entry[CONLLU_FIELD_TO_IDX['id']]
        self._text = token_entry[CONLLU_FIELD_TO_IDX['word']]

        if token_entry[CONLLU_FIELD_TO_IDX['misc']] != '_':
            for item in token_entry[CONLLU_FIELD_TO_IDX['misc']].split('|'):
                key_value = item.split('=')
                if len(key_value) == 1: 
                    # print(token_entry) # <COMMENT>: some key_value can not be splited, maybe data error?
                    continue
                key, value = key_value
                if key in ['beginCharOffset', 'endCharOffset']:
                    value = int(value)
                setattr(self, f'_{key}', value)
        if words is None:
            self.words = []
        else:
            self.words = words

    @property
    def words(self):
        """ Access the list of syntactic words underlying this token. """
        return self._words

    @words.setter
    def words(self, value):
        """ Set this token's list of underlying syntactic words. """
        self._words = value
        for w in self._words:
            w.parent_token = self

    @property
    def index(self):
        """ Access index of this token. """
        return self._index

    @index.setter
    def index(self, value):
        """ Set the token's index value. """
        self._index = value

    @property
    def text(self):
        """ Access text of this token. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the token's text value. Example: 'The'"""
        self._text = value

    @property
    def begin_char_offset(self):
        return self._beginCharOffset

    @property
    def end_char_offset(self):
        return self._endCharOffset

    def __repr__(self):
        return f"<{self.__class__.__name__} index={self.index};words={self.words};begin_char_offset={self.begin_char_offset};end_char_offset={self.end_char_offset}>"

class Word:

    def __init__(self, word_entry):
        self._index = word_entry[CONLLU_FIELD_TO_IDX['id']]
        self._text = word_entry[CONLLU_FIELD_TO_IDX['word']]
        self._lemma = word_entry[CONLLU_FIELD_TO_IDX['lemma']]
        if self._lemma == '_':
            self._lemma = None
        self._upos = word_entry[CONLLU_FIELD_TO_IDX['upos']]
        self._xpos = word_entry[CONLLU_FIELD_TO_IDX['xpos']]
        self._feats = word_entry[CONLLU_FIELD_TO_IDX['feats']]
        if self._upos == '_':
            self._upos = None
            self._xpos = None
            self._feats = None
        self._governor = word_entry[CONLLU_FIELD_TO_IDX['head']]
        self._dependency_relation = word_entry[CONLLU_FIELD_TO_IDX['deprel']]
        self._parent_token = None
        # check if there is dependency information
        if self._dependency_relation != '_':
            self._governor = int(self._governor)
        else:
            self._governor = None
            self._dependency_relation = None
        self._deps = None
        self._misc = None

    @property
    def dependency_relation(self):
        """ Access dependency relation of this word. Example: 'nmod'"""
        return self._dependency_relation

    @dependency_relation.setter
    def dependency_relation(self, value):
        """ Set the word's dependency relation value. Example: 'nmod'"""
        self._dependency_relation = value

    @property
    def lemma(self):
        """ Access lemma of this word. """
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        """ Set the word's lemma value. """
        self._lemma = value

    @property
    def governor(self):
        """ Access governor of this word. """
        return self._governor

    @governor.setter
    def governor(self, value):
        """ Set the word's governor value. """
        self._governor = value

    @property
    def pos(self):
        """ Access (treebank-specific) part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @pos.setter
    def pos(self, value):
        """ Set the word's (treebank-specific) part-of-speech value. Example: 'NNP'"""
        self._xpos = value

    @property
    def text(self):
        """ Access text of this word. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the word's text value. Example: 'The'"""
        self._text = value

    @property
    def xpos(self):
        """ Access treebank-specific part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @xpos.setter
    def xpos(self, value):
        """ Set the word's treebank-specific part-of-speech value. Example: 'NNP'"""
        self._xpos = value

    @property
    def upos(self):
        """ Access universal part-of-speech of this word. Example: 'DET'"""
        return self._upos

    @upos.setter
    def upos(self, value):
        """ Set the word's universal part-of-speech value. Example: 'DET'"""
        self._upos = value

    @property
    def feats(self):
        """ Access morphological features of this word. Example: 'Gender=Fem'"""
        return self._feats

    @feats.setter
    def feats(self, value):
        """ Set this word's morphological features. Example: 'Gender=Fem'"""
        self._feats = value

    @property
    def parent_token(self):
        """ Access the parent token of this word. """
        return self._parent_token

    @parent_token.setter
    def parent_token(self, value):
        """ Set this word's parent token. """
        self._parent_token = value

    @property
    def index(self):
        """ Access index of this word. """
        return self._index

    @index.setter
    def index(self, value):
        """ Set the word's index value. """
        self._index = value

    @property
    def deps(self):
        """ Access deps of this word. """
        return self._deps

    @deps.setter
    def deps(self, value):
        """ Set the word's deps value. """
        self._deps = value

    @property
    def misc(self):
        """ Access misc of this word. """
        return self._misc

    @misc.setter
    def misc(self, value):
        """ Set the word's misc value. """
        self._misc = value

    def __repr__(self):
        features = ['index', 'text', 'lemma', 'upos', 'xpos', 'feats', 'governor', 'dependency_relation']
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])

        return f"<{self.__class__.__name__} {feature_str}>"
