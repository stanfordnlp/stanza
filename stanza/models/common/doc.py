"""
Basic data structures
"""

import io
import re
import json

from stanza.models.ner.utils import decode_from_bioes

multi_word_token_id = re.compile(r"([0-9]+)-([0-9]+)")
multi_word_token_misc = re.compile(r".*MWT=Yes.*")

ID = 'id'
TEXT = 'text'
LEMMA = 'lemma'
UPOS = 'upos'
XPOS = 'xpos'
FEATS = 'feats'
HEAD = 'head'
DEPREL = 'deprel'
DEPS = 'deps'
MISC = 'misc'
NER = 'ner'
START_CHAR = 'start_char'
END_CHAR = 'end_char'
TYPE = 'type'
SENTIMENT = 'sentiment'

class Document:
    """ A document class that stores attributes of a document and carries a list of sentences.
    """

    def __init__(self, sentences, text=None):
        """ Construct a document given a list of sentences in the form of lists of CoNLL-U dicts.

        Args:
            sentences: a list of sentences, which being a list of token entry, in the form of a CoNLL-U dict.
            text: the raw text of the document.
        """
        self._sentences = []
        self._text = None
        self._num_tokens = 0
        self._num_words = 0

        self.text = text
        self._process_sentences(sentences)
        self._ents = []

    @property
    def text(self):
        """ Access the raw text for this document. """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the raw text for this document. """
        self._text = value

    @property
    def sentences(self):
        """ Access the list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value
    
    @property
    def num_tokens(self):
        """ Access the number of tokens for this document. """
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        """ Set the number of tokens for this document. """
        self._num_tokens = value

    @property
    def num_words(self):
        """ Access the number of words for this document. """
        return self._num_words

    @num_words.setter
    def num_words(self, value):
        """ Set the number of words for this document. """
        self._num_words = value

    @property
    def ents(self):
        """ Access the list of entities in this document. """
        return self._ents

    @ents.setter
    def ents(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    @property
    def entities(self):
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    def _process_sentences(self, sentences):
        self.sentences = []
        for tokens in sentences:
            self.sentences.append(Sentence(tokens, doc=self))
            begin_idx, end_idx = self.sentences[-1].tokens[0].start_char, self.sentences[-1].tokens[-1].end_char
            if all([self.text is not None, begin_idx is not None, end_idx is not None]): self.sentences[-1].text = self.text[begin_idx: end_idx]

        self.num_tokens = sum([len(sentence.tokens) for sentence in self.sentences])
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

    def get(self, fields, as_sentences=False, from_token=False):
        """ Get fields from a list of field names. 
        If only one field name (string or singleton list) is provided,
        return a list of that field; if more than one, return a list of list. 
        Note that all returned fields are after multi-word expansion.

        Args:
            fields: name of the fields as a list or a single string
            as_sentences: if True, return the fields as a list of sentences; otherwise as a whole list
            from_token: if True, get the fields from Token; otherwise from Word
        
        Returns:
            All requested fields.
        """
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."

        results = []
        for sentence in self.sentences:
            cursent = []
            # decide word or token
            if from_token:
                units = sentence.tokens
            else:
                units = sentence.words
            for unit in units:
                if len(fields) == 1:
                    cursent += [getattr(unit, fields[0])]
                else:
                    cursent += [[getattr(unit, field) for field in fields]]

            # decide whether append the results as a sentence or a whole list
            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

    def set(self, fields, contents, to_token=False, to_sentence=False):
        """Set fields based on contents. If only one field (string or
        singleton list) is provided, then a list of content will be
        expected; otherwise a list of list of contents will be expected.

        Args:
            fields: name of the fields as a list or a single string
            contents: field values to set; total length should be equal to number of words/tokens
            to_token: if True, set field values to tokens; otherwise to words

        """
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, (tuple, list)), "Must provide field names as a list."
        assert isinstance(contents, (tuple, list)), "Must provide contents as a list (one item per line)."
        assert len(fields) >= 1, "Must have at least one field."

        assert not to_sentence or not to_token, "Both to_token and to_sentence set to True, which is very confusing"

        if to_sentence:
            assert len(self.sentences) == len(contents), \
                "Contents must have the same length as the sentences"
            for sentence, content in zip(self.sentences, contents):
                if len(fields) == 1:
                    setattr(sentence, fields[0], content)
                else:
                    for field, piece in zip(fields, content):
                        setattr(sentence, field, piece)
        else:
            assert (to_token and self.num_tokens == len(contents)) or self.num_words == len(contents), \
                "Contents must have the same length as the original file."

            cidx = 0
            for sentence in self.sentences:
                # decide word or token
                if to_token:
                    units = sentence.tokens
                else:
                    units = sentence.words
                for unit in units:
                    if len(fields) == 1:
                        setattr(unit, fields[0], contents[cidx])
                    else:
                        for field, content in zip(fields, contents[cidx]):
                            setattr(unit, field, content)
                    cidx += 1

    def set_mwt_expansions(self, expansions):
        """ Extend the multi-word tokens annotated by tokenizer. A list of list of expansions
        will be expected for each multi-word token.
        """
        idx_e = 0
        for sentence in self.sentences:
            idx_w = 0
            for token in sentence.tokens:
                idx_w += 1
                m = multi_word_token_id.match(token.id)
                n = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                if not m and not n:
                    for word in token.words:
                        word.id = str(idx_w)
                        word.head, word.deprel = None, None # delete dependency information
                else:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    token.misc = None if token.misc == 'MWT=Yes' else '|'.join([x for x in token.misc.split('|') if x != 'MWT=Yes'])
                    token.id = f'{idx_w}-{idx_w_end}'
                    token.words = []
                    for i, e_word in enumerate(expanded):
                        token.words.append(Word({ID: str(idx_w + i), TEXT: e_word}))
                    idx_w = idx_w_end
            sentence._process_tokens(sentence.to_dict()) # reprocess to update sentence.words and sentence.dependencies
        self._process_sentences(self.to_dict()) # reprocess to update number of words
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation=False):
        """ Get the multi-word tokens. For training, return a list of
        (multi-word token, extended multi-word token); otherwise, return a list of
        multi-word token only.
        """
        expansions = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                m = multi_word_token_id.match(token.id)
                n = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                if m or n:
                    src = token.text
                    dst = ' '.join([word.text for word in token.words])
                    expansions.append([src, dst])
        if evaluation: expansions = [e[0] for e in expansions]
        return expansions

    def build_ents(self):
        """ Build the list of entities by iterating over all words. Return all entities as a list. """
        self.ents = []
        for s in self.sentences:
            s_ents = s.build_ents()
            self.ents += s_ents
        return self.ents

    def iter_words(self):
        """ An iterator that returns all of the words in this Document. """
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self):
        """ An iterator that returns all of the tokens in this Document. """
        for s in self.sentences:
            yield from s.tokens

    def to_dict(self):
        """ Dumps the whole document into a list of list of dictionary for each token in each sentence in the doc.
        """
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class Sentence:
    """ A sentence class that stores attributes of a sentence and carries a list of tokens.
    """

    def __init__(self, tokens, doc=None):
        """ Construct a setence given a list of tokens in the form of CoNLL-U dicts.
        """
        self._tokens = []
        self._words = []
        self._dependencies = []
        self._text = None
        self._ents = []
        self._doc = doc

        self._process_tokens(tokens)

    def _process_tokens(self, tokens):
        st, en = -1, -1
        self.tokens, self.words = [], []
        for i, entry in enumerate(tokens):
            if ID not in entry: # manually set a 1-based id for word if not exist
                entry[ID] = str(i+1)
            m = multi_word_token_id.match(entry.get(ID))
            n = multi_word_token_misc.match(entry.get(MISC)) if entry.get(MISC, None) is not None else None
            if m or n: # if this token is a multi-word token
                if m: st, en = int(m.group(1)), int(m.group(2))
                self.tokens.append(Token(entry))
            else: # else this token is a word
                new_word = Word(entry)
                self.words.append(new_word)
                idx = int(entry.get(ID))
                if idx <= en:
                    self.tokens[-1].words.append(new_word)
                else:
                    self.tokens.append(Token(entry, words=[new_word]))
                new_word.parent = self.tokens[-1]

        # check if there is dependency info
        is_complete_dependencies = all([word.head is not None and word.deprel is not None for word in self.words])
        is_complete_words = (len(self.words) >= len(self.tokens)) and (len(self.words) == int(self.words[-1].id))
        if is_complete_dependencies and is_complete_words: self.build_dependencies()
    
    @property
    def doc(self):
        """ Access the parent doc of this span. """
        return self._doc

    @doc.setter
    def doc(self, value):
        """ Set the parent doc of this span. """
        self._doc = value

    @property
    def text(self):
        """ Access the raw text for this sentence. """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the raw text for this sentence. """
        self._text = value

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
        """ Access the list of tokens for this sentence. """
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        """ Set the list of tokens for this sentence. """
        self._tokens = value

    @property
    def words(self):
        """ Access the list of words for this sentence. """
        return self._words

    @words.setter
    def words(self, value):
        """ Set the list of words for this sentence. """
        self._words = value
    
    @property
    def ents(self):
        """ Access the list of entities in this sentence. """
        return self._ents

    @ents.setter
    def ents(self, value):
        """ Set the list of entities in this sentence. """
        self._ents = value

    @property
    def entities(self):
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value):
        """ Set the list of entities in this sentence. """
        self._ents = value
    
    def build_ents(self):
        """ Build the list of entities by iterating over all tokens. Return all entities as a list. 
        
        Note that unlike other attributes, since NER requires raw text, the actual tagging are always
        performed at and attached to the `Token`s, instead of `Word`s.
        """
        self.ents = []
        tags = [w.ner for w in self.tokens]
        decoded = decode_from_bioes(tags)
        for e in decoded:
            ent_tokens = self.tokens[e['start']:e['end']+1]
            self.ents.append(Span(tokens=ent_tokens, type=e['type'], doc=self.doc, sent=self))
        return self.ents

    @property
    def sentiment(self):
        """ Returns the sentiment value for this sentence """
        return self._sentiment

    @sentiment.setter
    def sentiment(self, value):
        """ Set the sentiment value """
        self._sentiment = value

    def build_dependencies(self):
        """ Build the dependency graph for this sentence. Each dependency graph entry is
        a list of (head, deprel, word).
        """
        self.dependencies = []
        for word in self.words:
            if int(word.head) == 0:
                # make a word for the ROOT
                word_entry = {ID: "0", TEXT: "ROOT"}
                head = Word(word_entry)
            else:
                # id is index in words list + 1
                head = self.words[int(word.head) - 1]
                assert(int(word.head) == int(head.id))
            self.dependencies.append((head, word.deprel, word))

    def print_dependencies(self, file=None):
        """ Print the dependencies for this sentence. """
        for dep_edge in self.dependencies:
            print((dep_edge[2].text, dep_edge[0].id, dep_edge[1]), file=file)

    def dependencies_string(self):
        """ Dump the dependencies for this sentence into string. """
        dep_string = io.StringIO()
        self.print_dependencies(file=dep_string)
        return dep_string.getvalue().strip()

    def print_tokens(self, file=None):
        """ Print the tokens for this sentence. """
        for tok in self.tokens:
            print(tok.pretty_print(), file=file)

    def tokens_string(self):
        """ Dump the tokens for this sentence into string. """
        toks_string = io.StringIO()
        self.print_tokens(file=toks_string)
        return toks_string.getvalue().strip()

    def print_words(self, file=None):
        """ Print the words for this sentence. """
        for word in self.words:
            print(word.pretty_print(), file=file)

    def words_string(self):
        """ Dump the words for this sentence into string. """
        wrds_string = io.StringIO()
        self.print_words(file=wrds_string)
        return wrds_string.getvalue().strip()

    def to_dict(self):
        """ Dumps the sentence into a list of dictionary for each token in the sentence.
        """
        ret = []
        for token in self.tokens:
            ret += token.to_dict()
        return ret

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class Token:
    """ A token class that stores attributes of a token and carries a list of words. A token corresponds to a unit in the raw
    text. In some languages such as English, a token has a one-to-one mapping to a word, while in other languages such as French,
    a (multi-word) token might be expanded into multiple words that carry syntactic annotations.
    """

    def __init__(self, token_entry, words=None):
        """ Construct a token given a dictionary format token entry. Optionally link itself to the corresponding words.
        """
        assert token_entry.get(ID) and token_entry.get(TEXT), 'id and text should be included for the token'
        self._id, self._text, self._misc, self._words, self._start_char, self._end_char, self._ner = [None] * 7

        self.id = token_entry.get(ID)
        self.text = token_entry.get(TEXT)
        self.misc = token_entry.get(MISC, None)
        self.ner = token_entry.get(NER, None)
        self.words = words if words is not None else []

        if self.misc is not None:
            self.init_from_misc()

    def init_from_misc(self):
        """ Create attributes by parsing from the `misc` field.
        """
        for item in self._misc.split('|'):
            key_value = item.split('=', 1)
            if len(key_value) == 1: continue # some key_value can not be splited
            key, value = key_value
            if key in [START_CHAR, END_CHAR]:
                value = int(value)
            # set attribute
            attr = f'_{key}'
            if hasattr(self, attr):
                setattr(self, attr, value)

    @property
    def id(self):
        """ Access the index of this token. """
        return self._id

    @id.setter
    def id(self, value):
        """ Set the token's id value. """
        self._id = value

    @property
    def text(self):
        """ Access the text of this token. Example: 'The' """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the token's text value. Example: 'The' """
        self._text = value

    @property
    def misc(self):
        """ Access the miscellaneousness of this token. """
        return self._misc

    @misc.setter
    def misc(self, value):
        """ Set the token's miscellaneousness value. """
        self._misc = value if self._is_null(value) == False else None

    @property
    def words(self):
        """ Access the list of syntactic words underlying this token. """
        return self._words

    @words.setter
    def words(self, value):
        """ Set this token's list of underlying syntactic words. """
        self._words = value
        for w in self._words:
            w.parent = self

    @property
    def start_char(self):
        """ Access the start character index for this token in the raw text. """
        return self._start_char

    @property
    def end_char(self):
        """ Access the end character index for this token in the raw text. """
        return self._end_char
    
    @property
    def ner(self):
        """ Access the NER tag of this token. Example: 'B-ORG'"""
        return self._ner

    @ner.setter
    def ner(self, value):
        """ Set the token's NER tag. Example: 'B-ORG'"""
        self._ner = value if self._is_null(value) == False else None

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self, fields=[ID, TEXT, NER, MISC]):
        """ Dumps the token into a list of dictionary for this token with its extended words
        if the token is a multi-word token.
        """
        ret = []
        if multi_word_token_id.match(self.id):
            token_dict = {}
            for field in fields:
                if getattr(self, field) is not None:
                    token_dict[field] = getattr(self, field)
            ret.append(token_dict)
        for word in self.words:
            ret.append(word.to_dict())
        return ret

    def pretty_print(self):
        """ Print this token with its extended words in one line. """
        return f"<{self.__class__.__name__} id={self.id};words=[{', '.join([word.pretty_print() for word in self.words])}]>"

    def _is_null(self, value):
        return (value is None) or (value == '_')

class Word:
    """ A word class that stores attributes of a word.
    """

    def __init__(self, word_entry):
        """ Construct a word given a dictionary format word entry.
        """
        assert word_entry.get(ID) and word_entry.get(TEXT), 'id and text should be included for the word. {}'.format(word_entry)
        self._id, self._text, self._lemma, self._upos, self._xpos, self._feats, self._head, self._deprel, self._deps, \
            self._misc, self._parent = [None] * 11

        self.id = word_entry.get(ID)
        self.text = word_entry.get(TEXT)
        self.lemma = word_entry.get(LEMMA, None)
        self.upos = word_entry.get(UPOS, None)
        self.xpos = word_entry.get(XPOS, None)
        self.feats = word_entry.get(FEATS, None)
        self.head = word_entry.get(HEAD, None)
        self.deprel = word_entry.get(DEPREL, None)
        self.deps = word_entry.get(DEPS, None)
        self.misc = word_entry.get(MISC, None)

        if self.misc is not None:
            self.init_from_misc()

    def init_from_misc(self):
        """ Create attributes by parsing from the `misc` field.
        """
        for item in self._misc.split('|'):
            key_value = item.split('=', 1)
            if len(key_value) == 1: continue # some key_value can not be splited
            key, value = key_value
            # set attribute
            attr = f'_{key}'
            if hasattr(self, attr):
                setattr(self, attr, value)

    @property
    def id(self):
        """ Access the index of this word. """
        return self._id

    @id.setter
    def id(self, value):
        """ Set the word's index value. """
        self._id = value

    @property
    def text(self):
        """ Access the text of this word. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the word's text value. Example: 'The'"""
        self._text = value

    @property
    def lemma(self):
        """ Access the lemma of this word. """
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        """ Set the word's lemma value. """
        self._lemma = value if self._is_null(value) == False or self._text == '_' else None

    @property
    def upos(self):
        """ Access the universal part-of-speech of this word. Example: 'NOUN'"""
        return self._upos

    @upos.setter
    def upos(self, value):
        """ Set the word's universal part-of-speech value. Example: 'NOUN'"""
        self._upos = value if self._is_null(value) == False else None

    @property
    def xpos(self):
        """ Access the treebank-specific part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @xpos.setter
    def xpos(self, value):
        """ Set the word's treebank-specific part-of-speech value. Example: 'NNP'"""
        self._xpos = value if self._is_null(value) == False else None

    @property
    def feats(self):
        """ Access the morphological features of this word. Example: 'Gender=Fem'"""
        return self._feats

    @feats.setter
    def feats(self, value):
        """ Set this word's morphological features. Example: 'Gender=Fem'"""
        self._feats = value if self._is_null(value) == False else None

    @property
    def head(self):
        """ Access the id of the governer of this word. """
        return self._head

    @head.setter
    def head(self, value):
        """ Set the word's governor id value. """
        self._head = int(value) if self._is_null(value) == False else None

    @property
    def deprel(self):
        """ Access the dependency relation of this word. Example: 'nmod'"""
        return self._deprel

    @deprel.setter
    def deprel(self, value):
        """ Set the word's dependency relation value. Example: 'nmod'"""
        self._deprel = value if self._is_null(value) == False else None

    @property
    def deps(self):
        """ Access the dependencies of this word. """
        return self._deps

    @deps.setter
    def deps(self, value):
        """ Set the word's dependencies value. """
        self._deps = value if self._is_null(value) == False else None

    @property
    def misc(self):
        """ Access the miscellaneousness of this word. """
        return self._misc

    @misc.setter
    def misc(self, value):
        """ Set the word's miscellaneousness value. """
        self._misc = value if self._is_null(value) == False else None

    @property
    def parent(self):
        """ Access the parent token of this word. In the case of a multi-word token, a token can be the parent of
        multiple words. Note that this should return a reference to the parent token object.
        """
        return self._parent

    @parent.setter
    def parent(self, value):
        """ Set this word's parent token. In the case of a multi-word token, a token can be the parent of
        multiple words. Note that value here should be a reference to the parent token object.
        """
        self._parent = value

    @property
    def pos(self):
        """ Access the universal part-of-speech of this word. Example: 'NOUN'"""
        return self._upos

    @pos.setter
    def pos(self, value):
        """ Set the word's universal part-of-speech value. Example: 'NOUN'"""
        self._upos = value if self._is_null(value) == False else None

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self, fields=[ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]):
        """ Dumps the word into a dictionary.
        """
        word_dict = {}
        for field in fields:
            if getattr(self, field) is not None:
                word_dict[field] = getattr(self, field)
        return word_dict

    def pretty_print(self):
        """ Print the word in one line. """
        features = [ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL]
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])
        return f"<{self.__class__.__name__} {feature_str}>"

    def _is_null(self, value):
        return (value is None) or (value == '_')


class Span:
    """ A span class that stores attributes of a textual span. A span can be typed.
    A range of objects (e.g., entity mentions) can be represented as spans.
    """

    def __init__(self, span_entry=None, tokens=None, type=None, doc=None, sent=None):
        """ Construct a span given a span entry or a list of tokens. A valid reference to a doc
        must be provided to construct a span (otherwise the text of the span cannot be initialized).
        """
        assert span_entry is not None or (tokens is not None and type is not None), \
                'Either a span_entry or a token list needs to be provided to construct a span.'
        assert doc is not None, 'A parent doc must be provided to construct a span.'
        self._text, self._type, self._start_char, self._end_char = [None] * 4
        self._tokens = []
        self._words = []
        self._doc = doc
        self._sent = sent

        if span_entry is not None:
            self.init_from_entry(span_entry)

        if tokens is not None:
            self.init_from_tokens(tokens, type)

    def init_from_entry(self, span_entry):
        self.text = span_entry.get(TEXT, None)
        self.type = span_entry.get(TYPE, None)
        self.start_char = span_entry.get(START_CHAR, None)
        self.end_char = span_entry.get(END_CHAR, None)

    def init_from_tokens(self, tokens, type):
        assert isinstance(tokens, list), 'Tokens must be provided as a list to construct a span.'
        assert len(tokens) > 0, "Tokens of a span cannot be an empty list."
        self.tokens = tokens
        self.type = type
        # load start and end char offsets from tokens
        self.start_char = self.tokens[0].start_char
        self.end_char = self.tokens[-1].end_char
        # assume doc is already provided and not None
        self.text = self.doc.text[self.start_char:self.end_char]
        # collect the words of the span following tokens
        self.words = [w for t in tokens for w in t.words]

    @property
    def doc(self):
        """ Access the parent doc of this span. """
        return self._doc

    @doc.setter
    def doc(self, value):
        """ Set the parent doc of this span. """
        self._doc = value

    @property
    def text(self):
        """ Access the text of this span. Example: 'Stanford University'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the span's text value. Example: 'Stanford University'"""
        self._text = value

    @property
    def tokens(self):
        """ Access reference to a list of tokens that correspond to this span. """
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        """ Set the span's list of tokens. """
        self._tokens = value
    
    @property
    def words(self):
        """ Access reference to a list of words that correspond to this span. """
        return self._words

    @words.setter
    def words(self, value):
        """ Set the span's list of words. """
        self._words = value

    @property
    def type(self):
        """ Access the type of this span. Example: 'PERSON'"""
        return self._type

    @type.setter
    def type(self, value):
        """ Set the type of this span. """
        self._type = value

    @property
    def start_char(self):
        """ Access the start character offset of this span. """
        return self._start_char

    @start_char.setter
    def start_char(self, value):
        """ Set the start character offset of this span. """
        self._start_char = value

    @property
    def end_char(self):
        """ Access the end character offset of this span. """
        return self._end_char

    @end_char.setter
    def end_char(self, value):
        """ Set the end character offset of this span. """
        self._end_char = value

    def to_dict(self):
        """ Dumps the span into a dictionary. """
        attrs = ['text', 'type', 'start_char', 'end_char']
        span_dict = dict([(attr_name, getattr(self, attr_name)) for attr_name in attrs])
        return span_dict

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def pretty_print(self):
        """ Print the span in one line. """
        span_dict = self.to_dict()
        feature_str = ";".join(["{}={}".format(k,v) for k,v in span_dict.items()])
        return f"<{self.__class__.__name__} {feature_str}>"
