"""
Basic data structures
"""

import io
from itertools import repeat
import re
import json
import pickle
import warnings

from enum import Enum

import networkx as nx

from stanza.models.common.stanza_object import StanzaObject
from stanza.models.common.utils import misc_to_space_after, space_after_to_misc, misc_to_space_before, space_before_to_misc
from stanza.models.ner.utils import decode_from_bioes
from stanza.models.constituency import tree_reader
from stanza.models.coref.coref_chain import CorefMention, CorefChain, CorefAttachment

class MWTProcessingType(Enum):
    FLATTEN = 0 # flatten the current token into one ID instead of MWT
    PROCESS = 1 # process the current token as an MWT and expand it as such
    SKIP = 2 # do nothing on this token, simply increment IDs

multi_word_token_id = re.compile(r"([0-9]+)-([0-9]+)")
multi_word_token_misc = re.compile(r".*MWT=Yes.*")

MEXP = 'manual_expansion'
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
MULTI_NER = 'multi_ner'     # will represent tags from multiple NER models
START_CHAR = 'start_char'
END_CHAR = 'end_char'
TYPE = 'type'
SENTIMENT = 'sentiment'
CONSTITUENCY = 'constituency'
COREF_CHAINS = 'coref_chains'

# field indices when converting the document to conll
FIELD_TO_IDX = {ID: 0, TEXT: 1, LEMMA: 2, UPOS: 3, XPOS: 4, FEATS: 5, HEAD: 6, DEPREL: 7, DEPS: 8, MISC: 9}
FIELD_NUM = len(FIELD_TO_IDX)

class DocJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CorefMention):
            return obj.__dict__
        if isinstance(obj, CorefAttachment):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)

class Document(StanzaObject):
    """ A document class that stores attributes of a document and carries a list of sentences.
    """

    def __init__(self, sentences, text=None, comments=None, empty_sentences=None):
        """ Construct a document given a list of sentences in the form of lists of CoNLL-U dicts.

        Args:
            sentences: a list of sentences, which being a list of token entry, in the form of a CoNLL-U dict.
            text: the raw text of the document.
            comments: A list of list of strings to use as comments on the sentences, either None or the same length as sentences
        """
        self._sentences = []
        self._lang = None
        self._text = text
        self._num_tokens = 0
        self._num_words = 0

        self._process_sentences(sentences, comments, empty_sentences)
        self._ents = []
        self._coref = []
        if self._text is not None:
            self.build_ents()
            self.mark_whitespace()

    def mark_whitespace(self):
        for sentence in self._sentences:
            # TODO: pairwise, once we move to minimum 3.10
            for prev_token, next_token in zip(sentence.tokens[:-1], sentence.tokens[1:]):
                whitespace = self._text[prev_token.end_char:next_token.start_char]
                prev_token.spaces_after = whitespace
        for prev_sentence, next_sentence in zip(self._sentences[:-1], self._sentences[1:]):
            prev_token = prev_sentence.tokens[-1]
            next_token = next_sentence.tokens[0]
            whitespace = self._text[prev_token.end_char:next_token.start_char]
            prev_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[-1].tokens) > 0:
            final_token = self._sentences[-1].tokens[-1]
            whitespace = self._text[final_token.end_char:]
            final_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[0].tokens) > 0:
            first_token = self._sentences[0].tokens[0]
            whitespace = self._text[:first_token.start_char]
            first_token.spaces_before = whitespace


    @property
    def lang(self):
        """ Access the language of this document """
        return self._lang

    @lang.setter
    def lang(self, value):
        """ Set the language of this document """
        self._lang = value

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

    def _process_sentences(self, sentences, comments=None, empty_sentences=None):
        self.sentences = []
        if empty_sentences is None:
            empty_sentences = repeat([])
        for sent_idx, (tokens, empty_words) in enumerate(zip(sentences, empty_sentences)):
            try:
                sentence = Sentence(tokens, doc=self, empty_words=empty_words)
            except IndexError as e:
                raise IndexError("Could not process document at sentence %d" % sent_idx) from e
            except ValueError as e:
                tokens = ["|%s|" % t for t in tokens]
                tokens = ", ".join(tokens)
                raise ValueError("Could not process document at sentence %d\n  Raw tokens: %s" % (sent_idx, tokens)) from e
            self.sentences.append(sentence)
            begin_idx, end_idx = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
            if all((self.text is not None, begin_idx is not None, end_idx is not None)): sentence.text = self.text[begin_idx: end_idx]
            sentence.index = sent_idx

        self._count_words()

        # Add a #text comment to each sentence in a doc if it doesn't already exist
        if not comments:
            comments = [[] for x in self.sentences]
        else:
            comments = [list(x) for x in comments]
        for sentence, sentence_comments in zip(self.sentences, comments):
            # the space after text can occur in treebanks such as the Naija-NSC treebank,
            # which extensively uses `# text_en =` and `# text_ortho`
            if sentence.text and not any(comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text=") for comment in sentence_comments):
                # split/join to handle weird whitespace, especially newlines
                sentence_comments.append("# text = " + ' '.join(sentence.text.split()))
            elif not sentence.text:
                for comment in sentence_comments:
                    if comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text="):
                        sentence.text = comment.split("=", 1)[-1].strip()
                        break

            for comment in sentence_comments:
                sentence.add_comment(comment)

            # look for sent_id in the comments
            # if it's there, overwrite the sent_idx id from above
            for comment in sentence_comments:
                if comment.startswith("# sent_id"):
                    sentence.sent_id = comment.split("=", 1)[-1].strip()
                    break
            else:
                # no sent_id found.  add a comment with our enumerated id
                # setting the sent_id on the sentence will automatically add the comment
                sentence.sent_id = str(sentence.index)

    def _count_words(self):
        """
        Count the number of tokens and words
        """
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

    def set_mwt_expansions(self, expansions,
                           fake_dependencies=False,
                           process_manual_expanded=None):
        """ Extend the multi-word tokens annotated by tokenizer. A list of list of expansions
        will be expected for each multi-word token. Use `process_manual_expanded` to limit
        processing for tokens marked manually expanded:

        There are two types of MWT expansions: those with `misc`: `MWT=True`, and those with
        `manual_expansion`: True. The latter of which means that it is an expansion which the
        user manually specified through a postprocessor; the former means that it is a MWT
        which the detector picked out, but needs to be automatically expanded.

        process_manual_expanded = None - default; doesn't process manually expanded tokens
                                = True - process only manually expanded tokens (with `manual_expansion`: True)
                                = False - process only tokens explicitly tagged as MWT (`misc`: `MWT=True`)
        """

        idx_e = 0
        for sentence in self.sentences:
            idx_w = 0
            for token in sentence.tokens:
                idx_w += 1
                is_multi = (len(token.id) > 1)
                is_mwt = (multi_word_token_misc.match(token.misc) if token.misc is not None else None)
                is_manual_expansion = token.manual_expansion

                perform_mwt_processing = MWTProcessingType.FLATTEN

                if (process_manual_expanded and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_mwt):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.SKIP
                elif (process_manual_expanded==None and (is_mwt or is_multi)):
                    perform_mwt_processing = MWTProcessingType.PROCESS

                if perform_mwt_processing == MWTProcessingType.FLATTEN:
                    for word in token.words:
                        token.id = (idx_w, )
                        # delete dependency information
                        word.deps = None
                        word.head, word.deprel = None, None
                        word.id = idx_w
                elif perform_mwt_processing == MWTProcessingType.PROCESS:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    # in the event the MWT annotator only split the
                    # Token into a single Word, we preserve its text
                    # otherwise the Token's text is different from its
                    # only Word's text
                    if len(expanded) == 1:
                        expanded = [token.text]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    if token.misc:  # None can happen when using a prebuilt doc
                        token.misc = None if token.misc == 'MWT=Yes' else '|'.join([x for x in token.misc.split('|') if x != 'MWT=Yes'])
                    token.id = (idx_w, idx_w_end) if len(expanded) > 1 else (idx_w,)
                    token.words = []
                    for i, e_word in enumerate(expanded):
                        token.words.append(Word(sentence, {ID: idx_w + i, TEXT: e_word}))
                    idx_w = idx_w_end
                elif perform_mwt_processing == MWTProcessingType.SKIP:
                    token.id = tuple(orig_id + idx_e for orig_id in token.id)
                    for i in token.words:
                        i.id += idx_e
                    idx_w = token.id[-1]
                    token.manual_expansion = None

            # reprocess the words using the new tokens
            sentence.words = []
            for token in sentence.tokens:
                token.sent = sentence
                for word in token.words:
                    word.sent = sentence
                    word.parent = token
                    sentence.words.append(word)
                if token.start_char is not None and token.end_char is not None and "".join(word.text for word in token.words) == token.text:
                    start_char = token.start_char
                    for word in token.words:
                        end_char = start_char + len(word.text)
                        word.start_char = start_char
                        word.end_char = end_char
                        start_char = end_char

            if fake_dependencies:
                sentence.build_fake_dependencies()
            else:
                sentence.rebuild_dependencies()

        self._count_words() # update number of words & tokens
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation=False):
        """ Get the multi-word tokens. For training, return a list of
        (multi-word token, extended multi-word token); otherwise, return a list of
        multi-word token only. By default doesn't skip already expanded tokens, but
        `skip_already_expanded` will return only tokens marked as MWT.
        """
        expansions = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                is_multi = (len(token.id) > 1)
                is_mwt = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                is_manual_expansion = token.manual_expansion
                if (is_multi and not is_manual_expansion) or is_mwt:
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

    def sort_features(self):
        """ Sort the features on all the words... useful for prototype treebanks, for example """
        for sentence in self.sentences:
            for word in sentence.words:
                if not word.feats:
                    continue
                pieces = word.feats.split("|")
                pieces = sorted(pieces)
                word.feats = "|".join(pieces)

    def iter_words(self):
        """ An iterator that returns all of the words in this Document. """
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self):
        """ An iterator that returns all of the tokens in this Document. """
        for s in self.sentences:
            yield from s.tokens

    def sentence_comments(self):
        """ Returns a list of list of comments for the sentences """
        return [[comment for comment in sentence.comments] for sentence in self.sentences]

    @property
    def coref(self):
        """
        Access the coref lists of the document
        """
        return self._coref

    @coref.setter
    def coref(self, chains):
        """ Set the document's coref lists """
        self._coref = chains
        self._attach_coref_mentions(chains)

    def _attach_coref_mentions(self, chains):
        for sentence in self.sentences:
            for word in sentence.words:
                word.coref_chains = []

        for chain in chains:
            for mention_idx, mention in enumerate(chain.mentions):
                sentence = self.sentences[mention.sentence]
                for word_idx in range(mention.start_word, mention.end_word):
                    is_start = word_idx == mention.start_word
                    is_end = word_idx == mention.end_word - 1
                    is_representative = mention_idx == chain.representative_index
                    attachment = CorefAttachment(chain, is_start, is_end, is_representative)
                    sentence.words[word_idx].coref_chains.append(attachment)

    def reindex_sentences(self, start_index):
        for sent_id, sentence in zip(range(start_index, start_index + len(self.sentences)), self.sentences):
            sentence.sent_id = str(sent_id)

    def to_dict(self):
        """ Dumps the whole document into a list of list of dictionary for each token in each sentence in the doc.
        """
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec == 'c':
            return "\n\n".join("{:c}".format(s) for s in self.sentences)
        elif spec == 'C':
            return "\n\n".join("{:C}".format(s) for s in self.sentences)
        else:
            return str(self)

    def to_serialized(self):
        """ Dumps the whole document including text to a byte array containing a list of list of dictionaries for each token in each sentence in the doc.
        """
        return pickle.dumps((self.text, self.to_dict(), self.sentence_comments()))

    @classmethod
    def from_serialized(cls, serialized_string):
        """ Create and initialize a new document from a serialized string generated by Document.to_serialized_string():
        """
        stuff = pickle.loads(serialized_string)
        if not isinstance(stuff, tuple):
            raise TypeError("Serialized data was not a tuple when building a Document")
        if len(stuff) == 2:
            text, sentences = pickle.loads(serialized_string)
            doc = cls(sentences, text)
        else:
            text, sentences, comments = pickle.loads(serialized_string)
            doc = cls(sentences, text, comments)
        return doc


class Sentence(StanzaObject):
    """ A sentence class that stores attributes of a sentence and carries a list of tokens.
    """

    def __init__(self, tokens, doc=None, empty_words=None):
        """ Construct a sentence given a list of tokens in the form of CoNLL-U dicts.
        """
        self._tokens = []
        self._words = []
        self._dependencies = []
        self._text = None
        self._ents = []
        self._doc = doc
        self._constituency = None
        self._sentiment = None
        # comments are a list of comment lines occurring before the
        # sentence in a CoNLL-U file.  Can be empty
        self._comments = []
        self._doc_id = None

        # enhanced_dependencies represents the DEPS column
        # this is a networkx MultiDiGraph
        # with edges from the parent to the dependent
        # however, we set it to None until needed, as it is somewhat slow
        self._enhanced_dependencies = None
        self._process_tokens(tokens)

        if empty_words is not None:
            self._empty_words = [Word(self, entry) for entry in empty_words]
        else:
            self._empty_words = []

    def _process_tokens(self, tokens):
        st, en = -1, -1
        self.tokens, self.words = [], []
        for i, entry in enumerate(tokens):
            if ID not in entry: # manually set a 1-based id for word if not exist
                entry[ID] = (i+1, )
            if isinstance(entry[ID], int):
                entry[ID] = (entry[ID], )
            if len(entry.get(ID)) > 1: # if this token is a multi-word token
                st, en = entry[ID]
                self.tokens.append(Token(self, entry))
            else: # else this token is a word
                new_word = Word(self, entry)
                if len(self.words) > 0 and self.words[-1].id == new_word.id:
                    # this can happen in the following context:
                    # a document was created with MWT=Yes to mark that a token should be split
                    # and then there was an MWT "expansion" with a single word after that token
                    # we replace the Word in the Token assuming that the expansion token might
                    # have more information than the Token dict did
                    # note that a single word MWT like that can be detected with something like
                    #   multi_word_token_misc.match(entry.get(MISC)) if entry.get(MISC, None)
                    self.words[-1] = new_word
                    self.tokens[-1].words[-1] = new_word
                    continue
                self.words.append(new_word)
                idx = entry.get(ID)[0]
                if idx <= en:
                    self.tokens[-1].words.append(new_word)
                else:
                    self.tokens.append(Token(self, entry, words=[new_word]))
                new_word.parent = self.tokens[-1]

        # put all of the whitespace annotations (if any) on the Tokens instead of the Words
        for token in self.tokens:
            token.consolidate_whitespace()
        self.rebuild_dependencies()

    def has_enhanced_dependencies(self):
        """
        Whether or not the enhanced dependencies are part of this sentence
        """
        return self._enhanced_dependencies is not None and len(self._enhanced_dependencies) > 0

    @property
    def index(self):
        """
        Access the index of this sentence within the doc.

        If multiple docs were processed together,
        the sentence index will continue counting across docs.
        """
        return self._index

    @index.setter
    def index(self, value):
        """ Set the sentence's index value. """
        self._index = value

    @property
    def id(self):
        """
        Access the index of this sentence within the doc.

        If multiple docs were processed together,
        the sentence index will continue counting across docs.
        """
        warnings.warn("Use of sentence.id is deprecated.  Please use sentence.index instead", stacklevel=2)
        return self._index

    @id.setter
    def id(self, value):
        """ Set the sentence's index value. """
        warnings.warn("Use of sentence.id is deprecated.  Please use sentence.index instead", stacklevel=2)
        self._index = value

    @property
    def sent_id(self):
        """ conll-style sent_id  Will be set from index if unknown """
        return self._sent_id

    @sent_id.setter
    def sent_id(self, value):
        """ Set the sentence's sent_id value. """
        self._sent_id = value
        sent_id_comment = "# sent_id = " + str(value)
        for comment_idx, comment in enumerate(self._comments):
            if comment.startswith("# sent_id = "):
                self._comments[comment_idx] = sent_id_comment
                break
        else: # this is intended to be a for/else loop
            self._comments.append(sent_id_comment)

    @property
    def doc_id(self):
        """ conll-style doc_id  Can be left blank if unknown """
        return self._doc_id

    @doc_id.setter
    def doc_id(self, value):
        """ Set the sentence's doc_id value. """
        self._doc_id = value
        doc_id_comment = "# doc_id = " + str(value)
        for comment_idx, comment in enumerate(self._comments):
            if comment.startswith("# doc_id = "):
                self._comments[comment_idx] = doc_id_comment
                break
        else: # this is intended to be a for/else loop
            self._comments.append(doc_id_comment)

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
    def empty_words(self):
        """ Access the list of words for this sentence. """
        return self._empty_words

    @empty_words.setter
    def empty_words(self, value):
        """ Set the list of words for this sentence. """
        self._empty_words = value

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
        sentiment_comment = "# sentiment = " + str(value)
        for comment_idx, comment in enumerate(self._comments):
            if comment.startswith("# sentiment = "):
                self._comments[comment_idx] = sentiment_comment
                break
        else: # this is intended to be a for/else loop
            self._comments.append(sentiment_comment)

    @property
    def constituency(self):
        """ Returns the constituency tree for this sentence """
        return self._constituency

    @constituency.setter
    def constituency(self, value):
        """
        Set the constituency tree

        This incidentally updates the #constituency comment if it already exists,
        or otherwise creates a new comment # constituency = ...
        """
        self._constituency = value
        constituency_comment = "# constituency = " + str(value)
        constituency_comment = constituency_comment.replace("\n", "*NL*").replace("\r", "")
        for comment_idx, comment in enumerate(self._comments):
            if comment.startswith("# constituency = "):
                self._comments[comment_idx] = constituency_comment
                break
        else: # this is intended to be a for/else loop
            self._comments.append(constituency_comment)


    @property
    def comments(self):
        """ Returns CoNLL-style comments for this sentence """
        return self._comments

    def add_comment(self, comment):
        """ Adds a single comment to this sentence.

        If the comment does not already have # at the start, it will be added.
        """
        if not comment.startswith("#"):
            comment = "# " + comment
        if comment.startswith("# constituency ="):
            _, tree_text = comment.split("=", 1)
            tree = tree_reader.read_trees(tree_text)
            if len(tree) > 1:
                raise ValueError("Multiple constituency trees for one sentence: %s" % tree_text)
            self._constituency = tree[0]
            self._comments = [x for x in self._comments if not x.startswith("# constituency =")]
        elif comment.startswith("# sentiment ="):
            _, sentiment = comment.split("=", 1)
            sentiment = int(sentiment.strip())
            self._sentiment = sentiment
            self._comments = [x for x in self._comments if not x.startswith("# sentiment =")]
        elif comment.startswith("# sent_id ="):
            _, sent_id = comment.split("=", 1)
            sent_id = sent_id.strip()
            self._sent_id = sent_id
            self._comments = [x for x in self._comments if not x.startswith("# sent_id =")]
        elif comment.startswith("# doc_id ="):
            _, doc_id = comment.split("=", 1)
            doc_id = doc_id.strip()
            self._doc_id = doc_id
            self._comments = [x for x in self._comments if not x.startswith("# doc_id =")]
        self._comments.append(comment)

    def rebuild_dependencies(self):
        # rebuild dependencies if there is dependency info
        is_complete_dependencies = all(word.head is not None and word.deprel is not None for word in self.words)
        is_complete_words = (len(self.words) >= len(self.tokens)) and (len(self.words) == self.words[-1].id)
        if is_complete_dependencies and is_complete_words: self.build_dependencies()

    def build_dependencies(self):
        """ Build the dependency graph for this sentence. Each dependency graph entry is
        a list of (head, deprel, word).
        """
        self.dependencies = []
        for word in self.words:
            if word.head == 0:
                # make a word for the ROOT
                word_entry = {ID: 0, TEXT: "ROOT"}
                head = Word(self, word_entry)
            else:
                # id is index in words list + 1
                try:
                    head = self.words[word.head - 1]
                except IndexError as e:
                    raise IndexError("Word head {} is not a valid word index for word {}".format(word.head, word.id)) from e
                if word.head != head.id:
                    raise ValueError("Dependency tree is incorrectly constructed")
            self.dependencies.append((head, word.deprel, word))

    def build_fake_dependencies(self):
        self.dependencies = []
        for word_idx, word in enumerate(self.words):
            word.head = word_idx   # note that this goes one previous to the index
            word.deprel = "root" if word_idx == 0 else "dep"
            word.deps = "%d:%s" % (word.head, word.deprel)
            self.dependencies.append((word_idx, word.deprel, word))

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
        empty_idx = 0
        for token_idx, token in enumerate(self.tokens):
            while empty_idx < len(self._empty_words) and self._empty_words[empty_idx].id[0] < token.id[0]:
                ret.append(self._empty_words[empty_idx].to_dict())
                empty_idx += 1
            ret += token.to_dict()
        for empty_word in self._empty_words[empty_idx:]:
            ret.append(empty_word.to_dict())
        return ret

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec != 'c' and spec != 'C':
            return str(self)

        pieces = []
        empty_idx = 0
        for token_idx, token in enumerate(self.tokens):
            while empty_idx < len(self._empty_words) and self._empty_words[empty_idx].id[0] < token.id[0]:
                pieces.append(self._empty_words[empty_idx].to_conll_text())
                empty_idx += 1
            pieces.append(token.to_conll_text())
        for empty_word in self._empty_words[empty_idx:]:
            pieces.append(empty_word.to_conll_text())

        if spec == 'c':
            return "\n".join(pieces)
        elif spec == 'C':
            tokens = "\n".join(pieces)
            if len(self.comments) > 0:
                text = "\n".join(self.comments)
                return text + "\n" + tokens
            return tokens

def init_from_misc(unit):
    """Create attributes by parsing from the `misc` field.

    Also, remove start_char, end_char, and any other values we can set
    from the misc field if applicable, so that we don't repeat ourselves
    """
    remaining_values = []
    for item in unit._misc.split('|'):
        key_value = item.split('=', 1)
        if len(key_value) == 2:
            # some key_value can not be split
            key, value = key_value
            # start & end char are kept as ints
            if key in (START_CHAR, END_CHAR):
                value = int(value)
            # set attribute
            attr = f'_{key}'
            if hasattr(unit, attr):
                setattr(unit, attr, value)
                continue
            elif key == NER:
                # special case skipping NER for Words, since there is no Word NER field
                continue
        remaining_values.append(item)
    unit._misc = "|".join(remaining_values)


def dict_to_conll_text(token_dict, id_connector="-"):
    token_conll = ['_' for i in range(FIELD_NUM)]
    misc = []
    for key in token_dict:
        if key == START_CHAR or key == END_CHAR:
            misc.append("{}={}".format(key, token_dict[key]))
        elif key == NER:
            # TODO: potentially need to escape =|\ in the NER
            misc.append("{}={}".format(key, token_dict[key]))
        elif key == COREF_CHAINS:
            chains = token_dict[key]
            if len(chains) > 0:
                misc_chains = []
                for chain in chains:
                    if chain.is_start and chain.is_end:
                        coref_position = "unit-"
                    elif chain.is_start:
                        coref_position = "start-"
                    elif chain.is_end:
                        coref_position = "end-"
                    else:
                        coref_position = "middle-"
                    is_representative = "repr-" if chain.is_representative else ""
                    misc_chains.append("%s%sid%d" % (coref_position, is_representative, chain.chain.index))
                misc.append("{}={}".format(key, ",".join(misc_chains)))
        elif key == MISC:
            # avoid appending a blank misc entry.
            # otherwise the resulting misc field in the conll doc will wind up being blank text
            # TODO: potentially need to escape =|\ in the MISC as well
            if token_dict[key]:
                misc.append(token_dict[key])
        elif key == ID:
            token_conll[FIELD_TO_IDX[key]] = id_connector.join([str(x) for x in token_dict[key]]) if isinstance(token_dict[key], tuple) else str(token_dict[key])
        elif key in FIELD_TO_IDX:
            token_conll[FIELD_TO_IDX[key]] = str(token_dict[key])
    if misc:
        token_conll[FIELD_TO_IDX[MISC]] = "|".join(misc)
    else:
        token_conll[FIELD_TO_IDX[MISC]] = '_'
    # when a word (not mwt token) without head is found, we insert dummy head as required by the UD eval script
    if '-' not in token_conll[FIELD_TO_IDX[ID]] and '.' not in token_conll[FIELD_TO_IDX[ID]] and HEAD not in token_dict:
        token_conll[FIELD_TO_IDX[HEAD]] = str(int(token_dict[ID] if isinstance(token_dict[ID], int) else token_dict[ID][0]) - 1) # evaluation script requires head: int
    return "\t".join(token_conll)


class Token(StanzaObject):
    """ A token class that stores attributes of a token and carries a list of words. A token corresponds to a unit in the raw
    text. In some languages such as English, a token has a one-to-one mapping to a word, while in other languages such as French,
    a (multi-word) token might be expanded into multiple words that carry syntactic annotations.
    """

    def __init__(self, sentence, token_entry, words=None):
        """
        Construct a token given a dictionary format token entry. Optionally link itself to the corresponding words.
        The owning sentence must be passed in.
        """
        self._id = token_entry.get(ID)
        self._text = token_entry.get(TEXT)
        if not self._id:
            raise ValueError('id not included for the token')
        if not self._text:
            raise ValueError('text not included for the token')
        self._misc = token_entry.get(MISC, None)
        self._ner = token_entry.get(NER, None)
        self._multi_ner = token_entry.get(MULTI_NER, None)
        self._words = words if words is not None else []
        self._start_char = token_entry.get(START_CHAR, None)
        self._end_char = token_entry.get(END_CHAR, None)
        self._sent = sentence
        self._mexp = token_entry.get(MEXP, None)
        self._spaces_before = ""
        self._spaces_after = " "

        if self._misc is not None:
            init_from_misc(self)

    @property
    def id(self):
        """ Access the index of this token. """
        return self._id

    @id.setter
    def id(self, value):
        """ Set the token's id value. """
        self._id = value

    @property
    def manual_expansion(self):
        """ Access the whether this token was manually expanded. """
        return self._mexp

    @manual_expansion.setter
    def manual_expansion(self, value):
        """ Set the whether this token was manually expanded. """
        self._mexp = value

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

    def consolidate_whitespace(self):
        """
        Remove whitespace misc annotations from the Words and mark the whitespace on the Tokens
        """
        found_after = False
        found_before = False
        num_words = len(self.words)
        for word_idx, word in enumerate(self.words):
            misc = word.misc
            if not misc:
                continue
            pieces = misc.split("|")
            if word_idx == 0:
                if any(piece.startswith("SpacesBefore=") for piece in pieces):
                    self.spaces_before = misc_to_space_before(misc)
                    found_before = True
            else:
                if any(piece.startswith("SpacesBefore=") for piece in pieces):
                    warnings.warn("Found a SpacesBefore MISC annotation on a Word that was not the first Word in a Token")
            if word_idx == num_words - 1:
                if any(piece.startswith("SpaceAfter=") or piece.startswith("SpacesAfter=") for piece in pieces):
                    self.spaces_after = misc_to_space_after(misc)
                    found_after = True
            else:
                if any(piece.startswith("SpaceAfter=") or piece.startswith("SpacesAfter=") for piece in pieces):
                    unexpected_space_after = misc_to_space_after(misc)
                    if unexpected_space_after == "":
                        warnings.warn("Unexpected SpaceAfter=No annotation on a word in the middle of an MWT")
                    else:
                        warnings.warn("Unexpected SpacesAfter on a word in the middle on an MWT")
            pieces = [x for x in pieces if not x.startswith("SpacesAfter=") and not x.startswith("SpaceAfter=") and not x.startswith("SpacesBefore=")]
            word.misc = "|".join(pieces)

        misc = self.misc
        if misc:
            pieces = misc.split("|")
            if any(piece.startswith("SpacesBefore=") for piece in pieces):
                spaces_before = misc_to_space_before(misc)
                if found_before:
                    if spaces_before != self.spaces_before:
                        warnings.warn("Found conflicting SpacesBefore on a token and its word!")
                else:
                    self.spaces_before = spaces_before
            if any(piece.startswith("SpaceAfter=") or piece.startswith("SpacesAfter=") for piece in pieces):
                spaces_after = misc_to_space_after(misc)
                if found_after:
                    if spaces_after != self.spaces_after:
                        warnings.warn("Found conflicting SpaceAfter / SpacesAfter on a token and its word!")
                else:
                    self.spaces_after = spaces_after
            pieces = [x for x in pieces if not x.startswith("SpacesAfter=") and not x.startswith("SpaceAfter=") and not x.startswith("SpacesBefore=")]
            self.misc = "|".join(pieces)

    @property
    def spaces_before(self):
        """ SpacesBefore for the token. Translated from the MISC fields """
        return self._spaces_before

    @spaces_before.setter
    def spaces_before(self, value):
        self._spaces_before = value

    @property
    def spaces_after(self):
        """ SpaceAfter or SpacesAfter for the token.  Translated from the MISC field """
        return self._spaces_after

    @spaces_after.setter
    def spaces_after(self, value):
        self._spaces_after = value

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

    @property
    def multi_ner(self):
        """ Access the MULTI_NER tag of this token. Example: '(B-ORG, B-DISEASE)'"""
        return self._multi_ner

    @multi_ner.setter
    def multi_ner(self, value):
        """ Set the token's MULTI_NER tag. Example: '(B-ORG, B-DISEASE)'"""
        self._multi_ner = value if self._is_null(value) == False else None

    @property
    def sent(self):
        """ Access the pointer to the sentence that this token belongs to. """
        return self._sent

    @sent.setter
    def sent(self, value):
        """ Set the pointer to the sentence that this token belongs to. """
        self._sent = value

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec == 'C':
            return "\n".join(self.to_conll_text())
        elif spec == 'P':
            return self.pretty_print()
        else:
            return str(self)

    def to_conll_text(self):
        return "\n".join(dict_to_conll_text(x) for x in self.to_dict())

    def to_dict(self, fields=[ID, TEXT, MISC, START_CHAR, END_CHAR, NER, MULTI_NER, MEXP]):
        """ Dumps the token into a list of dictionary for this token with its extended words
        if the token is a multi-word token.
        """
        ret = []
        if len(self.id) > 1:
            token_dict = {}
            for field in fields:
                if getattr(self, field) is not None:
                    token_dict[field] = getattr(self, field)
            if MISC in fields:
                spaces_after = self.spaces_after
                if spaces_after is not None and spaces_after != ' ':
                    space_misc = space_after_to_misc(spaces_after)
                    if token_dict.get(MISC):
                        token_dict[MISC] = token_dict[MISC] + "|" + space_misc
                    else:
                        token_dict[MISC] = space_misc

                spaces_before = self.spaces_before
                if spaces_before is not None and spaces_before != '':
                    space_misc = space_before_to_misc(spaces_before)
                    if token_dict.get(MISC):
                        token_dict[MISC] = token_dict[MISC] + "|" + space_misc
                    else:
                        token_dict[MISC] = space_misc

            ret.append(token_dict)
        for word in self.words:
            word_dict = word.to_dict()
            if len(self.id) == 1 and NER in fields and getattr(self, NER) is not None: # propagate NER label to Word if it is a single-word token
                word_dict[NER] = getattr(self, NER)
            if len(self.id) == 1 and MULTI_NER in fields and getattr(self, MULTI_NER) is not None: # propagate MULTI_NER label to Word if it is a single-word token
                word_dict[MULTI_NER] = getattr(self, MULTI_NER)
            if len(self.id) == 1 and MISC in fields:
                spaces_after = self.spaces_after
                if spaces_after is not None and spaces_after != ' ':
                    space_misc = space_after_to_misc(spaces_after)
                    if word_dict.get(MISC):
                        word_dict[MISC] = word_dict[MISC] + "|" + space_misc
                    else:
                        word_dict[MISC] = space_misc

                spaces_before = self.spaces_before
                if spaces_before is not None and spaces_before != '':
                    space_misc = space_before_to_misc(spaces_before)
                    if word_dict.get(MISC):
                        word_dict[MISC] = word_dict[MISC] + "|" + space_misc
                    else:
                        word_dict[MISC] = space_misc
            ret.append(word_dict)
        return ret

    def pretty_print(self):
        """ Print this token with its extended words in one line. """
        return f"<{self.__class__.__name__} id={'-'.join([str(x) for x in self.id])};words=[{', '.join([word.pretty_print() for word in self.words])}]>"

    def _is_null(self, value):
        return (value is None) or (value == '_')

    def is_mwt(self):
        return len(self.words) > 1

class Word(StanzaObject):
    """ A word class that stores attributes of a word.
    """

    def __init__(self, sentence, word_entry):
        """ Construct a word given a dictionary format word entry.
        """
        self._id = word_entry.get(ID, None)
        if isinstance(self._id, tuple):
            if len(self._id) == 1:
                self._id = self._id[0]
        self._text = word_entry.get(TEXT, None)

        assert self._id is not None and self._text is not None, 'id and text should be included for the word. {}'.format(word_entry)

        self._lemma = word_entry.get(LEMMA, None)
        self._upos = word_entry.get(UPOS, None)
        self._xpos = word_entry.get(XPOS, None)
        self._feats = word_entry.get(FEATS, None)
        self._head = word_entry.get(HEAD, None)
        self._deprel = word_entry.get(DEPREL, None)
        self._misc = word_entry.get(MISC, None)
        self._start_char = word_entry.get(START_CHAR, None)
        self._end_char = word_entry.get(END_CHAR, None)
        self._parent = None
        self._sent = sentence
        self._mexp = word_entry.get(MEXP, None)
        self._coref_chains = None

        if self._misc is not None:
            init_from_misc(self)

        # use the setter, which will go up to the sentence and set the
        # dependencies on that graph
        self.deps = word_entry.get(DEPS, None)

    @property
    def manual_expansion(self):
        """ Access the whether this token was manually expanded. """
        return self._mexp

    @manual_expansion.setter
    def manual_expansion(self, value):
        """ Set the whether this token was manually expanded. """
        self._mexp = value

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
        """ Access the id of the governor of this word. """
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
        graph = self._sent._enhanced_dependencies
        if graph is None or not graph.has_node(self.id):
            return None

        data = []
        predecessors = sorted(list(graph.predecessors(self.id)), key=lambda x: x if isinstance(x, tuple) else (x,))
        for parent in predecessors:
            deps = sorted(list(graph.get_edge_data(parent, self.id)))
            for dep in deps:
                if isinstance(parent, int):
                    data.append("%d:%s" % (parent, dep))
                else:
                    data.append("%d.%d:%s" % (parent[0], parent[1], dep))
        if not data:
            return None

        return "|".join(data)

    @deps.setter
    def deps(self, value):
        """ Set the word's dependencies value. """
        graph = self._sent._enhanced_dependencies
        # if we don't have a graph, and we aren't trying to set any actual
        # dependencies, we can save the time of doing anything else
        if graph is None and value is None:
            return

        if graph is None:
            graph = nx.MultiDiGraph()
            self._sent._enhanced_dependencies = graph
        # need to make a new list: cannot iterate and delete at the same time
        if graph.has_node(self.id):
            in_edges = list(graph.in_edges(self.id))
            graph.remove_edges_from(in_edges)

        if value is None:
            return

        if isinstance(value, str):
            value = value.split("|")
        if all(isinstance(x, str) for x in value):
            value = [x.split(":", maxsplit=1) for x in value]
        for parent, dep in value:
            # we have to match the format of the IDs.  since the IDs
            # of the words are int if they aren't empty words, we need
            # to convert single int IDs into int instead of tuple
            parent = tuple(map(int, parent.split(".", maxsplit=1)))
            if len(parent) == 1:
                parent = parent[0]
            graph.add_edge(parent, self.id, dep)

    @property
    def misc(self):
        """ Access the miscellaneousness of this word. """
        return self._misc

    @misc.setter
    def misc(self, value):
        """ Set the word's miscellaneousness value. """
        self._misc = value if self._is_null(value) == False else None

    @property
    def start_char(self):
        """ Access the start character index for this token in the raw text. """
        return self._start_char

    @start_char.setter
    def start_char(self, value):
        self._start_char = value

    @property
    def end_char(self):
        """ Access the end character index for this token in the raw text. """
        return self._end_char

    @end_char.setter
    def end_char(self, value):
        self._end_char = value

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

    @property
    def coref_chains(self):
        """
        coref_chains points to a list of CorefChain namedtuple, which has a list of mentions and a representative mention.

        Useful for disambiguating words such as "him" (in languages where coref is available)

        Theoretically it is possible for multiple corefs to occur at the same word.  For example,
          "Chris Manning's NLP Group"
        could have "Chris Manning" and "Chris Manning's NLP Group" as overlapping entities
        """
        return self._coref_chains

    @coref_chains.setter
    def coref_chains(self, chain):
        """ Set the backref for the coref chains """
        self._coref_chains = chain

    @property
    def sent(self):
        """ Access the pointer to the sentence that this word belongs to. """
        return self._sent

    @sent.setter
    def sent(self, value):
        """ Set the pointer to the sentence that this word belongs to. """
        self._sent = value

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec == 'C':
            return self.to_conll_text()
        elif spec == 'P':
            return self.pretty_print()
        else:
            return str(self)

    def to_conll_text(self):
        """
        Turn a word into a conll representation (10 column tab separated)
        """
        token_dict = self.to_dict()
        return dict_to_conll_text(token_dict, '.')

    def to_dict(self, fields=[ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, START_CHAR, END_CHAR, MEXP, COREF_CHAINS]):
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


class Span(StanzaObject):
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
        if self.doc is not None and self.doc.text is not None:
            self.text = self.doc.text[self.start_char:self.end_char]
        elif tokens[0].sent is tokens[-1].sent:
            sentence = tokens[0].sent
            text_start = tokens[0].start_char - sentence.tokens[0].start_char
            text_end = tokens[-1].end_char - sentence.tokens[0].start_char
            self.text = sentence.text[text_start:text_end]
        else:
            # TODO: do any spans ever cross sentences?
            raise RuntimeError("Document text does not exist, and the span tested crosses two sentences, so it is impossible to extract the entity text!")
        # collect the words of the span following tokens
        self.words = [w for t in tokens for w in t.words]
        # set the sentence back-pointer to point to the sentence of the first token
        self.sent = tokens[0].sent

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

    @property
    def sent(self):
        """ Access the pointer to the sentence that this span belongs to. """
        return self._sent

    @sent.setter
    def sent(self, value):
        """ Set the pointer to the sentence that this span belongs to. """
        self._sent = value

    def to_dict(self):
        """ Dumps the span into a dictionary. """
        attrs = ['text', 'type', 'start_char', 'end_char']
        span_dict = dict([(attr_name, getattr(self, attr_name)) for attr_name in attrs])
        return span_dict

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def pretty_print(self):
        """ Print the span in one line. """
        span_dict = self.to_dict()
        feature_str = ";".join(["{}={}".format(k,v) for k,v in span_dict.items()])
        return f"<{self.__class__.__name__} {feature_str}>"
