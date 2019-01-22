"""
Basic data structures
"""

import re

from stanfordnlp.models.common.conll import FIELD_TO_IDX as CONLLU_FIELD_TO_IDX

multi_word_token_line = re.compile("[0-9]+\-[0-9]+")


class Document:

    def __init__(self, text):
        self._text = text
        self._conll_file = None
        self._sentences = []

    @property
    def conll_file(self):
        """ Access text of this document. Example: 'This is a sentence.'"""
        return self._conllu

    @conll_file.setter
    def conll_file(self, value):
        """ Set the document's text value. Example: 'This is a sentence.'"""
        self._conllu = value

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
        """ Access list of sentences for this document"""
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document"""
        self._sentences = value

    def load_annotations(self):
        """ Integrate info from conllu"""
        self._sentences = [Sentence(token_list) for token_list in self.conll_file.sents]

    def write_conll_to_file(self, file_path):
        """ write conll contents to file"""
        self.conll_file.write_conll(file_path)

class Sentence:

    def __init__(self, tokens):
        self._tokens = [Token(token_entry) for token_entry in tokens if self.is_token_line(token_entry)]
        self._dependencies = []
        # check if there is dependency info
        if not self.tokens[0].governor == '_':
            self.build_dependencies()

    def is_token_line(self, ln):
        if multi_word_token_line.match(ln[0]):
            return False
        else:
            return True

    @property
    def dependencies(self):
        """ Access list of tokens for this sentence"""
        return self._dependencies

    @dependencies.setter
    def tokens(self, value):
        """ Set the list of tokens for this sentence"""
        self._dependencies = value

    @property
    def tokens(self):
        """ Access list of tokens for this sentence"""
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        """ Set the list of tokens for this sentence"""
        self._tokens = value

    def build_dependencies(self):
        for tok in self.tokens:
            if tok.governor == 0:
                # make a token for the ROOT
                governor = Token(["_", "ROOT", "_", "_", "_", "_", "-1", "_", "_", "_", "_", "_"])
            else:
                # id is index in tokens list + 1
                governor = self.tokens[tok.governor-1]
            self.dependencies.append((governor, tok.dependency_relation, tok))

    def print_dependencies(self):
        for dep_edge in self.dependencies:
            print((dep_edge[2].word, dep_edge[0]._index, dep_edge[1]))

class Token:

    def __init__(self, token_entry):
        self._index = token_entry[CONLLU_FIELD_TO_IDX['id']]
        self._word = token_entry[CONLLU_FIELD_TO_IDX['word']]
        self._lemma = token_entry[CONLLU_FIELD_TO_IDX['lemma']]
        self._pos = token_entry[CONLLU_FIELD_TO_IDX['upos']]
        self._governor = token_entry[CONLLU_FIELD_TO_IDX['head']]
        # check if there is dependency information
        if self._governor != '_':
            self._governor = int(self._governor)
        self._dependency_relation = token_entry[CONLLU_FIELD_TO_IDX['deprel']]

    @property
    def dependency_relation(self):
        """ Access dependency relation of this token. Example: 'nmod'"""
        return self._dependency_relation

    @dependency_relation.setter
    def dependency_relation(self, value):
        """ Set the token's dependency relation value. Example: 'nmod'"""
        self._dependency_relation = value

    @property
    def lemma(self):
        """ Access lemma of this token. Example: 'NNP'"""
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        """ Set the token's lemma value. Example: 'NNP'"""
        self._lemma = value

    @property
    def governor(self):
        """ Access governor of this token. """
        return self._governor

    @governor.setter
    def governor(self, value):
        """ Set the token's governor value. """
        self._governor = value

    @property
    def pos(self):
        """ Access part-of-speech of this token. Example: 'NNP'"""
        return self._pos

    @pos.setter
    def pos(self, value):
        """ Set the token's part-of-speech value. Example: 'NNP'"""
        self._pos = value

    @property
    def word(self):
        """ Access text of this token. Example: 'The'"""
        return self._word

    @word.setter
    def word(self, value):
        """ Set the token's text value. Example: 'The'"""
        self._word = value
