"""
Basic data structures
"""

import io
import re

from stanfordnlp.models.common.conll import FIELD_TO_IDX

multi_word_token_line = re.compile("([0-9]+)\-([0-9]+)")


class Document:

    def __init__(self, sentences):
        self._sentences = []
        self._process_sentences(sentences)
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

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

    def _process_sentences(self, sentences):
        for tokens in sentences:
            self.sentences.append(Sentence(tokens))

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

    def to_dict(self):
        return [sentence.to_dict() for sentence in self.sentences]
                
class Sentence:

    def __init__(self, tokens):
        # tokens is a list of dict() containing both token entries and word entries
        self._tokens = []
        self._words = []
        self._process_tokens(tokens)
        self._dependencies = []
        # check if there is dependency info
        if self.words[0].deprel is not None:
            self.build_dependencies()

    def _process_tokens(self, tokens):
        st, en = -1, -1
        for entry in tokens:
            m = multi_word_token_line.match(entry.get('id'))
            if m: # if this token is a multi-word token
                st, en = int(m.group(1)), int(m.group(2))
                self._tokens.append(Token(entry))
            else: # else this token is a word
                new_word = Word(entry)
                self._words.append(new_word)
                idx = int(entry.get('id'))
                if idx <= en:
                    self._tokens[-1].words.append(new_word)
                else:
                    self.tokens.append(Token(entry, words=[new_word]))
                new_word.parent_token = self._tokens[-1]

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
            if word.head == 0:
                # make a word for the ROOT
                word_entry = {"id": "0", "text": "ROOT", "head": -1}
                head = Word(word_entry)
            else:
                # id is index in words list + 1
                head = self.words[word.head-1]
            self.dependencies.append((head, word.deprel, word))

    def print_dependencies(self, file=None):
        for dep_edge in self.dependencies:
            print((dep_edge[2].text, dep_edge[0].id, dep_edge[1]), file=file)

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

    def to_dict(self):
        ret = []
        for token in self.tokens:
            ret += token.to_dict()
        return ret
    

class Token:

    def __init__(self, token_entry, words=None):
        # token_entry is a dict() containing multiple fields (must include `id` and `text`)
        assert token_entry.get('id') and token_entry.get('text'), 'id and text should be included for the token'
        self._id = token_entry.get('id')
        self._text = token_entry.get('text')
        self._misc = token_entry.get('misc')
        self._words = words if words is not None else []

        if self._misc is not None: 
            self.init_from_misc()

    def init_from_misc(self):
        for item in self._misc.split('|'):
            key_value = item.split('=')
            if len(key_value) == 1: 
                # print(token_entry) # <COMMENT>: some key_value can not be splited, maybe data error?
                continue
            key, value = key_value
            if key in ['beginCharOffset', 'endCharOffset']:
                value = int(value)
            setattr(self, f'_{key}', value)

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
    def id(self):
        """ Access index of this token. """
        return self._id

    @id.setter
    def id(self, value):
        """ Set the token's id value. """
        self._id = value

    @property
    def text(self):
        """ Access text of this token. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the token's text value. Example: 'The'"""
        self._text = value

    @property
    def misc(self):
        """ Access misc of this word. """
        return self._misc

    @misc.setter
    def misc(self, value):
        """ Set the word's misc value. """
        self._misc = value

    @property
    def begin_char_offset(self):
        return self._beginCharOffset

    @property
    def end_char_offset(self):
        return self._endCharOffset

    def __repr__(self):
        return f"<{self.__class__.__name__} index={self.index};words={self.words};begin_char_offset={self.begin_char_offset};end_char_offset={self.end_char_offset}>"

    def to_dict(self, fields=['id', 'text', 'misc']):
        ret = []
        if len(self.words) != 1:
            token_dict = {}
            for field in fields:
                if getattr(self, field) is not None:
                    token_dict[field] = getattr(self, field)
            ret.append(token_dict)
        for word in self.words:
            ret.append(word.to_dict())
        return ret
  

class Word:

    def __init__(self, word_entry):
        # word_entry is a dict() containing multiple fields (must include `id` and `text`)
        assert word_entry.get('id') and word_entry.get('text'), 'id and text should be included for the word. {}'.format(word_entry)
        self._id = word_entry.get('id')
        self._text = word_entry.get('text')
        self._lemma = word_entry.get('lemma', None)
        self._upos = word_entry.get('upos', None)
        self._xpos = word_entry.get('xpos', None)
        self._feats = word_entry.get('feats', None)
        self._head = word_entry.get('head', None)
        self._deprel = word_entry.get('deprel', None)
        self._deps = word_entry.get('deps', None)
        self._misc = word_entry.get('misc', None)
        self._parent_token = None

    @property
    def deprel(self):
        """ Access dependency relation of this word. Example: 'nmod'"""
        return self._deprel

    @deprel.setter
    def deprel(self, value):
        """ Set the word's dependency relation value. Example: 'nmod'"""
        self._deprel = value

    @property
    def lemma(self):
        """ Access lemma of this word. """
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        """ Set the word's lemma value. """
        self._lemma = value

    @property
    def head(self):
        """ Access governor of this word. """
        return self._head

    @head.setter
    def head(self, value):
        """ Set the word's governor value. """
        self._head = value

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
    def id(self):
        """ Access index of this word. """
        return self._id

    @id.setter
    def id(self, value):
        """ Set the word's index value. """
        self._id = value

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
        features = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])

        return f"<{self.__class__.__name__} {feature_str}>"

    def to_dict(self, fields=['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']):
        word_dict = {}
        for field in fields:
            if getattr(self, field) is not None:
                word_dict[field] = getattr(self, field)
        return word_dict