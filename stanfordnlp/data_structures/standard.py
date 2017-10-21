"""
Module for standard NLP data structures: Token, Sentence, Document
"""

import json

class Token:
    """Class for representing tokens in documents"""

    def __init__(self, char_offsets, doc, doc_index, text):
        self._char_offsets = char_offsets
        self._doc = doc
        self._doc_index = doc_index
        self._embedding = None
        self._entity_mention = None
        self._ner = None
        self._pos = None
        self._starts_line = False
        self._sentence = None
        self._sentence_index = None
        self._text = text

    def to_dict(self):
        """Build a dictionary representation of the token"""
        return {"char_offsets": list(self.char_offsets), "text": self.text}

    @property
    def char_offsets(self):
        """Access the character offsets of this token. Example: (124,127)"""
        return self._char_offsets

    @char_offsets.setter
    def char_offsets(self, value):
        """Set the character offsets of this token. Example: (124,127)"""
        self._char_offsets = value

    @property
    def doc(self):
        """Access the document this token exists in."""
        return self._doc

    @doc.setter
    def doc(self, value):
        """Set the document this token exists in."""
        self._doc = value

    @property
    def doc_index(self):
        """If the document is viewed as a list of tokens, the index of this token in that list (starts at 0)"""
        return self._doc_index

    @doc_index.setter
    def doc_index(self, value):
        """Set the document index of this token"""
        self._doc_index = value

    @property
    def embedding(self):
        """The word embedding for this token"""
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        """Set the word embedding for this token"""
        self._embedding = value

    @property
    def entity_mention(self):
        """The entity mention this token belongs to, or None if it is not a part of an entity mention."""
        return self._entity_mention

    @entity_mention.setter
    def entity_mention(self, value):
        """Set the entity mention this token belongs to."""
        self._entity_mention = value

    @property
    def ner(self):
        """The named entity label of this token."""
        return self._ner

    @ner.setter
    def ner(self, value):
        """Set the named entity label for this token."""
        self._ner = value

    @property
    def pos(self):
        """The part of speech label of this token."""
        return self._pos

    @pos.setter
    def pos(self, value):
        """Set the part of speech label for this token."""
        self._pos = value

    @property
    def sentence(self):
        """The sentence this token belongs to."""
        return self._sentence

    @sentence.setter
    def sentence(self, value):
        """Set the sentence this token belongs to."""
        self._sentence = value

    @property
    def sentence_index(self):
        """If the sentence is viewed as a list of tokens, the index of this token in that list (starts at 0)"""
        return self._sentence_index

    @sentence_index.setter
    def sentence_index(self, value):
        """Set the sentence index of this token"""
        self._sentence_index = value

    @property
    def starts_line(self):
        """Does this token start a line ?"""
        return self._starts_line

    @starts_line.setter
    def starts_line(self, value):
        """Set whether or not this token starts a line"""
        self._starts_line = value

    @property
    def text(self):
        """ Access text of this token. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the token's text value. Example: 'The'"""
        self._text = value

class Sentence:
    """
    Class for representing sentences in a document
    """

    def __init__(self, doc, doc_index, tokens):
        # figure out sentence char offsets via tokens list
        sentence_char_start = tokens[0].char_offsets[0]
        sentence_char_end = tokens[-1].char_offsets[-1]
        self._char_offsets = (sentence_char_start, sentence_char_end)
        self._dep_parse = None
        self._doc = doc
        self._doc_index = doc_index
        self._entity_mentions = None
        self._tokens = tokens
        # determine text from char offsets
        self.text = doc.text[sentence_char_start:sentence_char_end]

    def to_dict(self):
        # convert tokens into dictionaries
        tokens_list = [token.to_dict() for token in self.tokens]
        return {"char_offsets": list(self.char_offsets), "text": self.text, "tokens": tokens_list}

    @property
    def char_offsets(self):
        """Access the character offsets of this sentence. Example: (124,127)"""
        return self._char_offsets

    @char_offsets.setter
    def char_offsets(self, value):
        """Set the character offsets of this sentence. Example: (124,127)"""
        self._char_offsets = value

    @property
    def dep_parse(self):
        """Access the dependency parse for this sentence."""
        return self._dep_parse

    @dep_parse.setter
    def dep_parse(self, value):
        """Set the dependency parse for this sentence."""
        self._dep_parse = value

    @property
    def doc(self):
        """Access the document this sentence exists in."""
        return self._doc

    @doc.setter
    def doc(self, value):
        """Set the document this sentence exists in."""
        self._doc = value

    @property
    def doc_index(self):
        """If the document is viewed as a list of sentences, the index of this sentence in that list (starts at 0)"""
        return self._document_index

    @doc_index.setter
    def doc_index(self, value):
        """Set the sentence index of this token"""
        self._doc_index = value

    @property
    def entity_mentions(self):
        """A list of entity mentions found in this sentence"""
        return self._entity_mentions

    @entity_mentions.setter
    def entity_mentions(self, value):
        """Set the list of entity mentions present in this sentence"""
        self._entity_mentions = value

    @property
    def tokens(self):
        """A list of tokens found in this sentence"""
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        """Set the list of tokens present in this sentence"""
        self._tokens = value

    @property
    def text(self):
        """ Access text of this sentence. Example: 'The dog chased the cat.'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the sentence's text value. Example: 'The dog chased the cat.'"""
        self._text = value

class Document:
    """
    Class for representing documents
    """

    def __init__(self, document_text, doc_id=None, document_date=None):
        self._doc_id = doc_id
        self._doc_date = document_date
        self._entity_mentions = None
        self._sentences = None
        self._tokens = None
        self._text = document_text

    def to_dict(self):
        sentences_list = [sentence.to_dict() for sentence in self.sentences]
        return {"text": self.text, "sentences": sentences_list}

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_file(cls, file_name, doc_id=None, doc_date=None):
        return cls(open(file_name).read(), doc_id, doc_date)

    @property
    def doc_id(self):
        """The document id for this document."""
        return self._document_id

    @doc_id.setter
    def doc_id(self, value):
        """Set the document id for this document."""
        self._doc_id = value

    @property
    def doc_date(self):
        """The doc date for this document."""
        return self._doc_date

    @doc_date.setter
    def doc_date(self, value):
        """Set the doc date for this document."""
        self._doc_date = value

    @property
    def entity_mentions(self):
        """A list of entity mentions found in this document"""
        return self._entity_mentions

    @entity_mentions.setter
    def entity_mentions(self, value):
        """Set the list of entity mentions present in this document"""
        self._entity_mentions = value

    @property
    def sentences(self):
        """A list of sentences found in this document"""
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """Set the list of sentences present in this document"""
        self._sentences = value

    @property
    def text(self):
        """The text of this document"""
        return self._text

    @text.setter
    def text(self, value):
        """Set the text of this document"""
        self._text = value

    @property
    def tokens(self):
        """A list of tokens found in this document"""
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        """Set the list of tokens present in this document"""
        self._tokens = value
