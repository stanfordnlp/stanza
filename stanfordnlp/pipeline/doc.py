"""
Basic data structures
"""

import io
import re

from stanfordnlp.models.common.conll import FIELD_TO_IDX as CONLLU_FIELD_TO_IDX

multi_word_token_line = re.compile("([0-9]+)\-([0-9]+)")


class Document:

    def __init__(self, text):
        self._text = text
        self._conll_file = None
        self._sentences = []

    @property
    def conll_file(self):
        """ Access the CoNLLFile of this document. """
        return self._conll_file

    @conll_file.setter
    def conll_file(self, value):
        """ Set the document's CoNLLFile value. """
        self._conll_file = value

    @property
    def text(self):
        """ Access text of this document. Example: 'This is a sentence.'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the document's text value. Example: 'This is a sentence.'"""
        self._text = value

    @property
    def sentences(self):
        """ Access list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value

    def load_annotations(self):
        """ Integrate info from the CoNLLFile instance. """
        self._sentences = [Sentence(token_list) for token_list in self.conll_file.sents]

    def write_conll_to_file(self, file_path):
        """ Write conll contents to file. """
        self.conll_file.write_conll(file_path)

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

    def __repr__(self):
        return f"<{self.__class__.__name__} index={self.index};words={self.words}>"

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

    def __repr__(self):
        features = ['index', 'text', 'lemma', 'upos', 'xpos', 'feats', 'governor', 'dependency_relation']
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])

        return f"<{self.__class__.__name__} {feature_str}>"
