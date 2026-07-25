"""
Basic data structures
"""

from __future__ import annotations

import io
import re
import json
import pickle
import warnings

from enum import Enum
from typing import Iterable, Iterator, Optional, Sequence, TextIO, Type, TypedDict, TypeVar, Union

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
LINE_NUMBER = 'line_number'
MORPHEMES = 'morphemes'

# field indices when converting the document to conll
FIELD_TO_IDX = {ID: 0, TEXT: 1, LEMMA: 2, UPOS: 3, XPOS: 4, FEATS: 5, HEAD: 6, DEPREL: 7, DEPS: 8, MISC: 9}
FIELD_NUM = len(FIELD_TO_IDX)

DEFAULT_OUTPUT_FIELDS = [ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, START_CHAR, END_CHAR, NER, MULTI_NER, MEXP, COREF_CHAINS, MORPHEMES]
NO_OFFSETS_OUTPUT_FIELDS = [ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, NER, MULTI_NER, MEXP, COREF_CHAINS, MORPHEMES]

PretokenizedText = list[list[str]]
DocumentText = Union[str, PretokenizedText]
CommentLines = list[str]
DocumentComments = Sequence[CommentLines]
# ``Sequence[str]`` would also accept a bare string and make field iteration
# operate on individual characters.
FieldNames = Union[list[str], tuple[str, ...]]
SentenceId = Union[str, int]
TokenEntryId = int
TokenEntryIdComponents = Sequence[TokenEntryId]
TokenId = Union[tuple[int], tuple[int, int]]
EmptyWordId = tuple[int, int]
WordId = Union[int, tuple[int, int]]
# Likewise, a bare string is not a collection of independent NER layers.
MultiNerTags = Union[list[str], tuple[str, ...]]
DependencyParts = tuple[str, str]
Dependencies = Union[Sequence[str], Sequence[DependencyParts]]
DependencyGovernor = Union["Word", int]
DependencyEdge = tuple[DependencyGovernor, str, "Word"]
DependencyEdges = list[DependencyEdge]
_JSONValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Sequence["_JSONValue"],
    dict[str, "_JSONValue"],
]
_DocumentT = TypeVar("_DocumentT", bound="Document")


class _RequiredTokenEntry(TypedDict):
    text: str


class _OptionalTokenEntryFields(TypedDict, total=False):
    lemma: Optional[str]
    upos: Optional[str]
    xpos: Optional[str]
    feats: Optional[str]
    head: Optional[int]
    deprel: Optional[str]
    deps: Optional[Union[str, Dependencies]]
    misc: Optional[str]
    ner: Optional[str]
    multi_ner: Optional[MultiNerTags]
    start_char: Optional[int]
    end_char: Optional[int]
    manual_expansion: Optional[bool]
    coref_chains: Optional[list[CorefAttachment]]
    morphemes: Optional[list[str]]


class TokenEntry(_RequiredTokenEntry, _OptionalTokenEntryFields, total=False):
    id: Union[TokenEntryId, TokenEntryIdComponents]


class _RequiredEmptyWordEntry(TypedDict):
    id: EmptyWordId
    text: str


class EmptyWordEntry(_RequiredEmptyWordEntry, _OptionalTokenEntryFields):
    pass


def _require_token_entry(entry, location: str) -> TokenEntry:
    """Build a precise entry from the closed fields used by default output.

    Token and Word allow properties to be added at runtime, so their
    field-selected ``to_dict`` methods deliberately retain a dynamic return
    boundary.  Sentence and Document use the default field set and construct a
    closed TokenEntry here instead of asserting or casting that dynamic value.
    """
    try:
        validated = _validate_serialized_token_entry(entry, location)
    except TypeError as error:
        raise ValueError(
            f"{location} did not return a valid default token dictionary"
        ) from error

    coref_chains = entry.get(COREF_CHAINS)
    if coref_chains is not None:
        if (not isinstance(coref_chains, list)
                or any(
                    not isinstance(attachment, CorefAttachment)
                    for attachment in coref_chains
                )):
            raise ValueError(
                f"{location} did not return valid coreference attachments"
            )
        validated[COREF_CHAINS] = list(coref_chains)
    elif COREF_CHAINS in entry:
        validated[COREF_CHAINS] = None
    return validated


class SpanDict(TypedDict):
    text: Optional[str]
    type: Optional[str]
    start_char: Optional[int]
    end_char: Optional[int]


class SpanInput(TypedDict, total=False):
    text: Optional[str]
    type: Optional[str]
    start_char: Optional[int]
    end_char: Optional[int]


SpanEntry = Union[SpanInput, SpanDict]


class _SerializedDocumentPayload(TypedDict):
    text: Optional[DocumentText]
    sentences: list[list[TokenEntry]]
    comments: list[list[str]]
    empty_sentences: list[list[EmptyWordEntry]]


def _deserialize_document_text(value: _JSONValue) -> Optional[DocumentText]:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        pretokenized_text: PretokenizedText = []
        for sentence in value:
            if not isinstance(sentence, list):
                raise TypeError(
                    "Serialized Document text must be a string, null, "
                    "or a list of lists of strings"
                )
            serialized_sentence: list[str] = []
            for token in sentence:
                if not isinstance(token, str):
                    raise TypeError(
                        "Serialized Document text must be a string, null, "
                        "or a list of lists of strings"
                    )
                serialized_sentence.append(token)
            pretokenized_text.append(serialized_sentence)
        return pretokenized_text
    raise TypeError(
        "Serialized Document text must be a string, null, "
        "or a list of lists of strings"
    )


def _validate_serialized_token_entry(
        value: _JSONValue,
        location: str,
    ) -> TokenEntry:
    if not isinstance(value, dict):
        raise TypeError(f"Serialized {location} must be a dict")

    text = value.get(TEXT)
    if not isinstance(text, str):
        raise TypeError(f"Serialized {location}.{TEXT} must be a string")
    entry: TokenEntry = {TEXT: text}

    entry_id = value.get(ID)
    if entry_id is not None:
        if isinstance(entry_id, bool):
            raise TypeError(f"Serialized {location}.{ID} must contain integers")
        if isinstance(entry_id, int):
            entry[ID] = entry_id
        elif isinstance(entry_id, (list, tuple)) and len(entry_id) == 1:
            first_id = entry_id[0]
            if not isinstance(first_id, int) or isinstance(first_id, bool):
                raise TypeError(
                    f"Serialized {location}.{ID} must contain integers"
                )
            entry[ID] = (first_id,)
        elif isinstance(entry_id, (list, tuple)) and len(entry_id) == 2:
            first_id, second_id = entry_id
            if (not isinstance(first_id, int)
                    or isinstance(first_id, bool)
                    or not isinstance(second_id, int)
                    or isinstance(second_id, bool)):
                raise TypeError(
                    f"Serialized {location}.{ID} must contain integers"
                )
            entry[ID] = (first_id, second_id)
        else:
            raise TypeError(
                f"Serialized {location}.{ID} must be an integer "
                "or a one- or two-integer list"
            )

    lemma = value.get(LEMMA)
    if lemma is not None and not isinstance(lemma, str):
        raise TypeError(f"Serialized {location}.{LEMMA} must be a string or null")
    if LEMMA in value:
        entry[LEMMA] = lemma

    upos = value.get(UPOS)
    if upos is not None and not isinstance(upos, str):
        raise TypeError(f"Serialized {location}.{UPOS} must be a string or null")
    if UPOS in value:
        entry[UPOS] = upos

    xpos = value.get(XPOS)
    if xpos is not None and not isinstance(xpos, str):
        raise TypeError(f"Serialized {location}.{XPOS} must be a string or null")
    if XPOS in value:
        entry[XPOS] = xpos

    feats = value.get(FEATS)
    if feats is not None and not isinstance(feats, str):
        raise TypeError(f"Serialized {location}.{FEATS} must be a string or null")
    if FEATS in value:
        entry[FEATS] = feats

    deprel = value.get(DEPREL)
    if deprel is not None and not isinstance(deprel, str):
        raise TypeError(f"Serialized {location}.{DEPREL} must be a string or null")
    if DEPREL in value:
        entry[DEPREL] = deprel

    misc = value.get(MISC)
    if misc is not None and not isinstance(misc, str):
        raise TypeError(f"Serialized {location}.{MISC} must be a string or null")
    if MISC in value:
        entry[MISC] = misc

    ner = value.get(NER)
    if ner is not None and not isinstance(ner, str):
        raise TypeError(f"Serialized {location}.{NER} must be a string or null")
    if NER in value:
        entry[NER] = ner

    head = value.get(HEAD)
    if (head is not None
            and (not isinstance(head, int) or isinstance(head, bool))):
        raise TypeError(f"Serialized {location}.{HEAD} must be an integer or null")
    if HEAD in value:
        entry[HEAD] = head

    start_char = value.get(START_CHAR)
    if (start_char is not None
            and (not isinstance(start_char, int)
                 or isinstance(start_char, bool))):
        raise TypeError(
            f"Serialized {location}.{START_CHAR} must be an integer or null"
        )
    if START_CHAR in value:
        entry[START_CHAR] = start_char

    end_char = value.get(END_CHAR)
    if (end_char is not None
            and (not isinstance(end_char, int)
                 or isinstance(end_char, bool))):
        raise TypeError(
            f"Serialized {location}.{END_CHAR} must be an integer or null"
        )
    if END_CHAR in value:
        entry[END_CHAR] = end_char

    manual_expansion = value.get(MEXP)
    if manual_expansion is not None and not isinstance(manual_expansion, bool):
        raise TypeError(f"Serialized {location}.{MEXP} must be a boolean or null")
    if MEXP in value:
        entry[MEXP] = manual_expansion

    multi_ner = value.get(MULTI_NER)
    if multi_ner is not None:
        if not isinstance(multi_ner, (list, tuple)):
            raise TypeError(
                f"Serialized {location}.{MULTI_NER} must be a list of strings or null"
            )
        serialized_tags: list[str] = []
        for tag in multi_ner:
            if not isinstance(tag, str):
                raise TypeError(
                    f"Serialized {location}.{MULTI_NER} must be a list of strings or null"
                )
            serialized_tags.append(tag)
        entry[MULTI_NER] = serialized_tags
    elif MULTI_NER in value:
        entry[MULTI_NER] = None

    morphemes = value.get(MORPHEMES)
    if morphemes is not None:
        if not isinstance(morphemes, (list, tuple)):
            raise TypeError(
                f"Serialized {location}.{MORPHEMES} must be a list of strings or null"
            )
        serialized_morphemes: list[str] = []
        for morpheme in morphemes:
            if not isinstance(morpheme, str):
                raise TypeError(
                    f"Serialized {location}.{MORPHEMES} must be a list of strings or null"
                )
            serialized_morphemes.append(morpheme)
        entry[MORPHEMES] = serialized_morphemes
    elif MORPHEMES in value:
        entry[MORPHEMES] = None

    dependencies = value.get(DEPS)
    if isinstance(dependencies, str):
        entry[DEPS] = dependencies
    elif dependencies is not None:
        if not isinstance(dependencies, (list, tuple)):
            raise TypeError(
                f"Serialized {location}.{DEPS} must be a string, sequence, or null"
            )
        if all(isinstance(dependency, str) for dependency in dependencies):
            string_dependencies: list[str] = []
            for dependency in dependencies:
                if isinstance(dependency, str):
                    string_dependencies.append(dependency)
            entry[DEPS] = string_dependencies
        else:
            pair_dependencies: list[DependencyParts] = []
            for dependency in dependencies:
                if (not isinstance(dependency, (list, tuple))
                        or len(dependency) != 2):
                    raise TypeError(
                        f"Serialized {location}.{DEPS} must contain strings "
                        "or two-string sequences"
                    )
                parent, relation = dependency
                if not isinstance(parent, str) or not isinstance(relation, str):
                    raise TypeError(
                        f"Serialized {location}.{DEPS} must contain strings "
                        "or two-string lists"
                    )
                pair_dependencies.append((parent, relation))
            entry[DEPS] = pair_dependencies
    elif DEPS in value:
        entry[DEPS] = None

    return entry


def _require_empty_word_entry(entry, location: str) -> EmptyWordEntry:
    validated = _require_token_entry(entry, location)
    entry_id = validated.get(ID)
    if entry_id is None:
        raise ValueError(f"{location} did not include an empty-word ID")
    normalized_id = _normalize_token_id(entry_id)
    if len(normalized_id) != 2:
        raise ValueError(
            f"{location} did not include a two-component empty-word ID"
        )

    empty_entry: EmptyWordEntry = {
        ID: (normalized_id[0], normalized_id[1]),
        TEXT: validated[TEXT],
    }
    if LEMMA in validated:
        empty_entry[LEMMA] = validated[LEMMA]
    if UPOS in validated:
        empty_entry[UPOS] = validated[UPOS]
    if XPOS in validated:
        empty_entry[XPOS] = validated[XPOS]
    if FEATS in validated:
        empty_entry[FEATS] = validated[FEATS]
    if HEAD in validated:
        empty_entry[HEAD] = validated[HEAD]
    if DEPREL in validated:
        empty_entry[DEPREL] = validated[DEPREL]
    if DEPS in validated:
        empty_entry[DEPS] = validated[DEPS]
    if MISC in validated:
        empty_entry[MISC] = validated[MISC]
    if NER in validated:
        empty_entry[NER] = validated[NER]
    if MULTI_NER in validated:
        empty_entry[MULTI_NER] = validated[MULTI_NER]
    if START_CHAR in validated:
        empty_entry[START_CHAR] = validated[START_CHAR]
    if END_CHAR in validated:
        empty_entry[END_CHAR] = validated[END_CHAR]
    if MEXP in validated:
        empty_entry[MEXP] = validated[MEXP]
    if COREF_CHAINS in validated:
        empty_entry[COREF_CHAINS] = validated[COREF_CHAINS]
    if MORPHEMES in validated:
        empty_entry[MORPHEMES] = validated[MORPHEMES]
    return empty_entry


def _deserialize_sentences(
        value: _JSONValue,
        field_name: str,
    ) -> list[list[TokenEntry]]:
    if not isinstance(value, list):
        raise TypeError(f"Serialized Document {field_name} must be a list")

    sentences: list[list[TokenEntry]] = []
    for sentence_idx, sentence in enumerate(value):
        if not isinstance(sentence, list):
            raise TypeError(
                f"Serialized Document {field_name}[{sentence_idx}] must be a list"
            )
        sentences.append([
            _validate_serialized_token_entry(
                entry,
                f"{field_name}[{sentence_idx}][{entry_idx}]",
            )
            for entry_idx, entry in enumerate(sentence)
        ])
    return sentences


def _deserialize_empty_sentences(
        value: _JSONValue,
    ) -> list[list[EmptyWordEntry]]:
    if not isinstance(value, list):
        raise TypeError("Serialized Document empty_sentences must be a list")

    sentences: list[list[EmptyWordEntry]] = []
    for sentence_idx, sentence in enumerate(value):
        if not isinstance(sentence, list):
            raise TypeError(
                f"Serialized Document empty_sentences[{sentence_idx}] must be a list"
            )
        empty_words: list[EmptyWordEntry] = []
        for entry_idx, entry in enumerate(sentence):
            location = f"empty_sentences[{sentence_idx}][{entry_idx}]"
            try:
                empty_words.append(
                    _require_empty_word_entry(entry, f"Serialized {location}")
                )
            except ValueError as error:
                raise TypeError(str(error)) from error
        sentences.append(empty_words)
    return sentences


def _deserialize_comments(value: _JSONValue) -> Optional[list[list[str]]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("Serialized Document comments must be a list or null")

    comments: list[list[str]] = []
    for sentence_idx, sentence_comments in enumerate(value):
        if not isinstance(sentence_comments, list):
            raise TypeError(
                "Serialized Document comments"
                f"[{sentence_idx}] must be a list of strings"
            )
        serialized_comments: list[str] = []
        for comment in sentence_comments:
            if not isinstance(comment, str):
                raise TypeError(
                    "Serialized Document comments"
                    f"[{sentence_idx}] must be a list of strings"
                )
            serialized_comments.append(comment)
        comments.append(serialized_comments)
    return comments


def _normalize_token_id(
        token_id: Union[TokenEntryId, TokenEntryIdComponents],
    ) -> TokenId:
    if isinstance(token_id, int):
        return (token_id,)

    normalized_id = tuple(token_id)
    if len(normalized_id) == 1:
        return (normalized_id[0],)
    if len(normalized_id) == 2:
        return normalized_id[0], normalized_id[1]
    raise ValueError(f"Token IDs must contain one or two integers, got {token_id!r}")


def _split_legacy_empty_sentences(
        sentences: list[list[TokenEntry]],
    ) -> tuple[list[list[TokenEntry]], list[list[EmptyWordEntry]]]:
    """Separate decimal empty-node IDs from MWT ranges in old JSON payloads."""
    regular_sentences: list[list[TokenEntry]] = []
    empty_sentences: list[list[EmptyWordEntry]] = []

    for sentence in sentences:
        regular_tokens: list[TokenEntry] = []
        empty_words: list[EmptyWordEntry] = []
        seen_word_ids: set[int] = set()
        for entry in sentence:
            entry_id = entry.get(ID)
            normalized_id = (
                None if entry_id is None else _normalize_token_id(entry_id)
            )
            if (normalized_id is not None
                    and len(normalized_id) == 2
                    and (normalized_id[0] == 0
                         or normalized_id[1] <= normalized_id[0]
                         or normalized_id[0] in seen_word_ids)):
                empty_words.append(
                    _require_empty_word_entry(entry, "Legacy empty word")
                )
            else:
                regular_tokens.append(entry)
                if normalized_id is not None and len(normalized_id) == 1:
                    seen_word_ids.add(normalized_id[0])
        regular_sentences.append(regular_tokens)
        empty_sentences.append(empty_words)

    return regular_sentences, empty_sentences


def _restore_empty_words(
        document: _DocumentT,
        empty_sentences: Sequence[Sequence[EmptyWordEntry]],
    ) -> _DocumentT:
    if len(document.sentences) != len(empty_sentences):
        raise TypeError(
            "Serialized Document empty_sentences must have the same length "
            "as sentences"
        )
    for sentence, empty_words in zip(document.sentences, empty_sentences):
        sentence.empty_words = [
            Word(sentence, entry)
            for entry in empty_words
        ]
    return document


def _empty_word_id(word: Word) -> tuple[int, int]:
    word_id = word.id
    if isinstance(word_id, int):
        raise ValueError(f"Empty words must have a two-part ID, got {word_id!r}")
    return word_id


class DocJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CorefMention):
            return obj.__dict__
        if isinstance(obj, CorefAttachment):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)

class RestrictedUnpickler(pickle.Unpickler):
    # Stanza Document serialization only ever produces tuples, lists, dicts,
    # and scalar primitives. No custom classes are needed.
    SAFE_CLASSES = frozenset({
        ('builtins', 'tuple'),
        ('builtins', 'list'),
        ('builtins', 'dict'),
    })

    def find_class(self, module, name):
        if (module, name) not in self.SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Blocked unsafe global: {module}.{name}"
            )
        return super().find_class(module, name)

class Document(StanzaObject):
    """ A document class that stores attributes of a document and carries a list of sentences.
    """

    def __init__(self, sentences: Sequence[Sequence[TokenEntry]], text: Optional[DocumentText] = None,
                 comments: Optional[DocumentComments] = None,
                 empty_sentences: Optional[Sequence[Sequence[EmptyWordEntry]]] = None) -> None:
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

    def mark_whitespace(self) -> None:
        if not isinstance(self._text, str):
            return
        text = self._text
        for sentence in self._sentences:
            # TODO: pairwise, once we move to minimum 3.10
            for prev_token, next_token in zip(sentence.tokens[:-1], sentence.tokens[1:]):
                whitespace = text[prev_token.end_char:next_token.start_char]
                prev_token.spaces_after = whitespace
        for prev_sentence, next_sentence in zip(self._sentences[:-1], self._sentences[1:]):
            prev_token = prev_sentence.tokens[-1]
            next_token = next_sentence.tokens[0]
            whitespace = text[prev_token.end_char:next_token.start_char]
            prev_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[-1].tokens) > 0:
            final_token = self._sentences[-1].tokens[-1]
            whitespace = text[final_token.end_char:]
            final_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[0].tokens) > 0:
            first_token = self._sentences[0].tokens[0]
            whitespace = text[:first_token.start_char]
            first_token.spaces_before = whitespace


    @property
    def lang(self) -> Optional[str]:
        """ Access the language of this document """
        return self._lang

    @lang.setter
    def lang(self, value: Optional[str]) -> None:
        """ Set the language of this document """
        self._lang = value

    @property
    def text(self) -> Optional[DocumentText]:
        """ Access the raw text for this document. """
        return self._text

    @text.setter
    def text(self, value: Optional[DocumentText]) -> None:
        """ Set the raw text for this document. """
        self._text = value

    @property
    def sentences(self) -> list[Sentence]:
        """ Access the list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value: list[Sentence]) -> None:
        """ Set the list of tokens for this document. """
        self._sentences = value

    @property
    def num_tokens(self) -> int:
        """ Access the number of tokens for this document. """
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value: int) -> None:
        """ Set the number of tokens for this document. """
        self._num_tokens = value

    @property
    def num_words(self) -> int:
        """ Access the number of words for this document. """
        return self._num_words

    @num_words.setter
    def num_words(self, value: int) -> None:
        """ Set the number of words for this document. """
        self._num_words = value

    @property
    def ents(self) -> list[Span]:
        """ Access the list of entities in this document. """
        return self._ents

    @ents.setter
    def ents(self, value: list[Span]) -> None:
        """ Set the list of entities in this document. """
        self._ents = value

    @property
    def entities(self) -> list[Span]:
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value: list[Span]) -> None:
        """ Set the list of entities in this document. """
        self._ents = value

    def _process_sentences(self, sentences: Sequence[Sequence[TokenEntry]],
                           comments: Optional[DocumentComments] = None,
                           empty_sentences: Optional[Sequence[Sequence[EmptyWordEntry]]] = None) -> None:
        self.sentences = []
        sentence_entries = [list(sentence) for sentence in sentences]
        if empty_sentences is None:
            sentence_entries, inferred_empty_sentences = (
                _split_legacy_empty_sentences(sentence_entries)
            )
            empty_sentence_iter: Iterable[Sequence[EmptyWordEntry]] = (
                inferred_empty_sentences
            )
        else:
            if len(empty_sentences) != len(sentence_entries):
                raise ValueError(
                    "empty_sentences must have the same length as sentences"
                )
            empty_sentence_iter = empty_sentences
        for sent_idx, (tokens, empty_words) in enumerate(zip(sentence_entries, empty_sentence_iter)):
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
            document_text = self.text
            if isinstance(document_text, str) and begin_idx is not None and end_idx is not None:
                sentence.text = document_text[begin_idx:end_idx]
            sentence.index = sent_idx

        self._count_words()

        # Add a #text comment to each sentence in a doc if it doesn't already exist
        if not comments:
            comments = [[] for x in self.sentences]
        else:
            if len(comments) != len(self.sentences):
                raise ValueError(
                    "comments must have the same length as sentences"
                )
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

            # look for speaker in the comments
            for comment in sentence_comments:
                if comment.startswith("# speaker"):
                    sentence.speaker = comment.split("=", 1)[-1].strip()
                    break
            else:
                sentence.speaker = None

    def _count_words(self) -> None:
        """
        Count the number of tokens and words
        """
        self.num_tokens = sum([len(sentence.tokens) for sentence in self.sentences])
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

    def get(self, fields: Union[str, Sequence[str]],
            as_sentences: bool = False, from_token: bool = False):
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
        assert isinstance(fields, Sequence), "Must provide field names as a sequence."
        fields = list(fields)
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

    def set(self, fields: Union[str, Sequence[str]], contents,
            to_token: bool = False, to_sentence: bool = False) -> None:
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
        assert isinstance(fields, Sequence), "Must provide field names as a sequence."
        fields = list(fields)
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

    def set_mwt_expansions(self, expansions: Sequence[str],
                           fake_dependencies: bool = False,
                           process_manual_expanded: Optional[bool] = None) -> None:
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
                    token.id = _normalize_token_id([orig_id + idx_e for orig_id in token.id])
                    for token_word in token.words:
                        if not isinstance(token_word.id, int):
                            raise ValueError(
                                "Words in an unexpanded token must have integer IDs"
                            )
                        token_word.id = token_word.id + idx_e
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
                if len(token.words) == 1:
                    token.words[0].start_char = token.start_char
                    token.words[0].end_char = token.end_char
                elif token.start_char is not None and token.end_char is not None:
                    search_string = "^%s$" % ("\\s*".join("(%s)" % re.escape(word.text) for word in token.words))
                    match = re.compile(search_string).match(token.text)
                    if match:
                        for word_idx, word in enumerate(token.words):
                            word.start_char = match.start(word_idx+1) + token.start_char
                            word.end_char = match.end(word_idx+1) + token.start_char

            if fake_dependencies:
                sentence.build_fake_dependencies()
            else:
                sentence.rebuild_dependencies()

        self._count_words() # update number of words & tokens
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation: bool = False) -> Union[list[list[str]], list[str]]:
        """ Get the multi-word tokens. For training, return a list of
        (multi-word token, extended multi-word token); otherwise, return a list of
        multi-word token only. By default doesn't skip already expanded tokens, but
        `skip_already_expanded` will return only tokens marked as MWT.
        """
        expansions: list[list[str]] = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                is_multi = (len(token.id) > 1)
                is_mwt = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                is_manual_expansion = token.manual_expansion
                if (is_multi and not is_manual_expansion) or is_mwt:
                    src = token.text
                    dst = ' '.join([word.text for word in token.words])
                    expansions.append([src, dst])
        if evaluation:
            return [expansion[0] for expansion in expansions]
        return expansions

    def build_ents(self) -> list[Span]:
        """ Build the list of entities by iterating over all words. Return all entities as a list. """
        self.ents = []
        for s in self.sentences:
            s_ents = s.build_ents()
            self.ents += s_ents
        return self.ents

    def sort_features(self) -> None:
        """ Sort the features on all the words... useful for prototype treebanks, for example """
        for sentence in self.sentences:
            for word in sentence.words:
                if not word.feats:
                    continue
                pieces = word.feats.split("|")
                pieces = sorted(pieces, key=str.casefold)
                word.feats = "|".join(pieces)

    def iter_words(self) -> Iterator[Word]:
        """ An iterator that returns all of the words in this Document. """
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self) -> Iterator[Token]:
        """ An iterator that returns all of the tokens in this Document. """
        for s in self.sentences:
            yield from s.tokens

    def sentence_comments(self) -> list[list[str]]:
        """ Returns a list of list of comments for the sentences """
        return [[comment for comment in sentence.comments] for sentence in self.sentences]

    @property
    def coref(self) -> list[CorefChain]:
        """
        Access the coref lists of the document
        """
        return self._coref

    @coref.setter
    def coref(self, chains: list[CorefChain]) -> None:
        """ Set the document's coref lists """
        self._coref = chains
        self._attach_coref_mentions(chains)

    def _attach_coref_mentions(self, chains: Sequence[CorefChain]) -> None:
        for sentence in self.sentences:
            for word in sentence.all_words:
                word.coref_chains = []

        for chain in chains:
            for mention_idx, mention in enumerate(chain.mentions):
                sentence = self.sentences[mention.sentence]
                if isinstance(mention.start_word, tuple):
                    attachment = CorefAttachment(chain, True, True, False)
                    empty_word = sentence._empty_words[mention.start_word[1]-1]
                    empty_word_chains = empty_word.coref_chains
                    if empty_word_chains is None:
                        raise RuntimeError("Coreference attachments were not initialized")
                    empty_word_chains.append(attachment)
                else:
                    end_word = mention.end_word
                    if not isinstance(end_word, int):
                        raise ValueError(
                            "A regular coreference mention must end at an integer word index"
                        )
                    for word_idx in range(mention.start_word, end_word):
                        is_start = word_idx == mention.start_word
                        is_end = word_idx == end_word - 1
                        is_representative = mention_idx == chain.representative_index
                        attachment = CorefAttachment(chain, is_start, is_end, is_representative)
                        word = sentence.words[word_idx]
                        word_chains = word.coref_chains
                        if word_chains is None:
                            raise RuntimeError("Coreference attachments were not initialized")
                        word_chains.append(attachment)

    def reindex_sentences(self, start_index: int) -> None:
        for sent_id, sentence in zip(range(start_index, start_index + len(self.sentences)), self.sentences):
            sentence.sent_id = str(sent_id)

    def to_dict(self) -> list[list[TokenEntry]]:
        """ Dumps the whole document into a list of list of dictionary for each token in each sentence in the doc.
        """
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec: str) -> str:
        if spec and spec[0] in ('c', 'C'):
            spec = "{:%s}" % spec
            return "\n\n".join(spec.format(s) for s in self.sentences)
        else:
            return str(self)

    def to_serialized(self) -> bytes:
        """Dumps the whole document including text to a UTF-8 encoded JSON byte string.

        The format is a dict with keys 'text', 'sentences', 'comments', and
        'empty_sentences'. Empty words are kept separate from regular tokens
        because their two-part IDs cannot always be distinguished from MWT IDs.
        Old pickle-format blobs can still be loaded by from_serialized() but
        will produce a DeprecationWarning.
        """
        sentences: list[list[TokenEntry]] = [
            [
                _require_token_entry(entry, "Token.to_dict()")
                for token in sentence.tokens
                for entry in token.to_dict()
            ]
            for sentence in self.sentences
        ]
        empty_sentences: list[list[EmptyWordEntry]] = [
            [
                _require_empty_word_entry(word.to_dict(), "Word.to_dict()")
                for word in sentence.empty_words
            ]
            for sentence in self.sentences
        ]
        payload: _SerializedDocumentPayload = {
            "text": self.text,
            "sentences": sentences,
            "comments": self.sentence_comments(),
            "empty_sentences": empty_sentences,
        }
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def from_serialized(cls: Type[_DocumentT], serialized_string: bytes) -> _DocumentT:
        """Create and initialize a new document from a serialized string.

        Accepts both the current JSON format produced by to_serialized() and
        the legacy pickle format. Pickle-format blobs will trigger a
        DeprecationWarning; support for them will be removed in a future release.
        """
        # Detect JSON by checking for a leading b'{' (UTF-8, no BOM).
        # Pickle blobs always begin with 0x80 (protocol opcode), never '{'.
        if serialized_string[:1] == b"{":
            payload_value: _JSONValue = json.loads(serialized_string)
            if not isinstance(payload_value, dict):
                raise TypeError(
                    "Serialized JSON data was not a dict when building a Document"
                )
            serialized_text = _deserialize_document_text(payload_value.get("text"))
            serialized_sentences = _deserialize_sentences(
                payload_value.get("sentences"),
                "sentences",
            )
            if any(len(sentence) == 0 for sentence in serialized_sentences):
                raise TypeError(
                    "Serialized Document sentences cannot contain an empty sentence"
                )
            serialized_comments = _deserialize_comments(
                payload_value.get("comments")
            )
            if "empty_sentences" in payload_value:
                empty_sentences = payload_value.get("empty_sentences")
                serialized_empty_sentences = (
                    None
                    if empty_sentences is None
                    else _deserialize_empty_sentences(empty_sentences)
                )
            else:
                (
                    serialized_sentences,
                    serialized_empty_sentences,
                ) = _split_legacy_empty_sentences(serialized_sentences)
            if (serialized_comments
                    and len(serialized_comments) != len(serialized_sentences)):
                raise TypeError(
                    "Serialized Document comments must have the same length "
                    "as sentences"
                )
            if (serialized_empty_sentences is not None
                    and len(serialized_empty_sentences)
                    != len(serialized_sentences)):
                raise TypeError(
                    "Serialized Document empty_sentences must have the same "
                    "length as sentences"
                )
            if serialized_comments is not None:
                document = cls(
                    serialized_sentences,
                    serialized_text,
                    serialized_comments,
                )
            else:
                document = cls(serialized_sentences, serialized_text)
            if serialized_empty_sentences is not None:
                return _restore_empty_words(
                    document,
                    serialized_empty_sentences,
                )
            return document

        # Legacy pickle path.
        warnings.warn(
            "Loading a Document from a pickle-format serialized string is deprecated "
            "and will be removed in a future release. Re-save using Document.to_serialized(), "
            "which now produces a JSON-format byte string.",
            DeprecationWarning,
            stacklevel=2,
        )
        stuff = RestrictedUnpickler(io.BytesIO(serialized_string)).load()
        if not isinstance(stuff, tuple):
            raise TypeError(
                "Serialized pickle data was not a tuple when building a Document"
            )
        if len(stuff) == 2:
            text_value, sentences_value = stuff
            comments_value = None
            has_comments = False
        elif len(stuff) == 3:
            text_value, sentences_value, comments_value = stuff
            has_comments = True
        else:
            raise TypeError(
                "Serialized pickle data must contain two or three items"
            )

        text = _deserialize_document_text(text_value)
        sentences = _deserialize_sentences(sentences_value, "sentences")
        comments = _deserialize_comments(comments_value)
        sentences, empty_sentences = _split_legacy_empty_sentences(sentences)
        if has_comments:
            document = cls(sentences, text, comments)
        else:
            document = cls(sentences, text)
        return _restore_empty_words(document, empty_sentences)


class Sentence(StanzaObject):
    """ A sentence class that stores attributes of a sentence and carries a list of tokens.
    """

    def __init__(self, tokens: Sequence[TokenEntry], doc: Optional[Document] = None,
                 empty_words: Optional[Sequence[EmptyWordEntry]] = None) -> None:
        """ Construct a sentence given a list of tokens in the form of CoNLL-U dicts.
        """
        self._tokens = []
        self._words = []
        self._dependencies: DependencyEdges = []
        self._text = None
        self._ents = []
        self._doc = doc
        self._constituency = None
        self._sentiment = None
        self._index = 0
        self._sent_id = "0"
        self._speaker = None
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
            self._empty_words = [
                Word(
                    self,
                    _require_empty_word_entry(entry, "Empty word"),
                )
                for entry in empty_words
            ]
        else:
            self._empty_words = []

    def _process_tokens(self, tokens: Sequence[TokenEntry]) -> None:
        st, en = -1, -1
        self.tokens, self.words = [], []
        for i, entry in enumerate(tokens):
            entry_id = entry.get(ID)
            if entry_id is None: # manually set a 1-based id for word if not exist
                normalized_id: TokenId = (i+1,)
            else:
                normalized_id = _normalize_token_id(entry_id)
            entry[ID] = normalized_id

            if len(normalized_id) > 1: # if this token is a multi-word token
                st, en = normalized_id[0], normalized_id[1]
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
                idx = normalized_id[0]
                if idx <= en:
                    self.tokens[-1].words.append(new_word)
                else:
                    self.tokens.append(Token(self, entry, words=[new_word]))
                new_word.parent = self.tokens[-1]

        # put all of the whitespace annotations (if any) on the Tokens instead of the Words
        for token in self.tokens:
            token.consolidate_whitespace()
        self.rebuild_dependencies()

    def has_enhanced_dependencies(self) -> bool:
        """
        Whether or not the enhanced dependencies are part of this sentence
        """
        return self._enhanced_dependencies is not None and len(self._enhanced_dependencies) > 0

    @property
    def enhanced_dependencies(self) -> nx.MultiDiGraph:
        """
        Returns the enhanced_dependencies graph.

        Creates an empty one if one currently does not exist.
        """
        graph = self._enhanced_dependencies
        if graph is None:
            graph = nx.MultiDiGraph()
            self._enhanced_dependencies = graph
        return graph

    @property
    def index(self) -> int:
        """
        Access the index of this sentence within the doc.

        If multiple docs were processed together,
        the sentence index will continue counting across docs.
        """
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        """ Set the sentence's index value. """
        self._index = value

    @property
    def id(self) -> int:
        """
        Access the index of this sentence within the doc.

        If multiple docs were processed together,
        the sentence index will continue counting across docs.
        """
        warnings.warn("Use of sentence.id is deprecated.  Please use sentence.index instead", stacklevel=2)
        return self._index

    @id.setter
    def id(self, value: int) -> None:
        """ Set the sentence's index value. """
        warnings.warn("Use of sentence.id is deprecated.  Please use sentence.index instead", stacklevel=2)
        self._index = value

    @property
    def sent_id(self) -> SentenceId:
        """ conll-style sent_id  Will be set from index if unknown """
        return self._sent_id

    @sent_id.setter
    def sent_id(self, value: SentenceId) -> None:
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
    def speaker(self) -> Optional[str]:
        """ conll-style speaker - adopt the EN GUM formatting """
        return self._speaker

    @speaker.setter
    def speaker(self, value: Optional[str]) -> None:
        """ Set the sentence's speaker value. """
        self._speaker = value
        speaker_comment = "# speaker = " + str(value)
        if not value:
            for comment_idx, comment in enumerate(self._comments):
                if comment.startswith("# speaker = "):
                    self._comments.pop(comment_idx)
                    break
        else:
            for comment_idx, comment in enumerate(self._comments):
                if comment.startswith("# speaker = "):
                    self._comments[comment_idx] = speaker_comment
                    break
            else: # this is intended to be a for/else loop
                self._comments.append(speaker_comment)

    @property
    def doc_id(self) -> Optional[str]:
        """ conll-style doc_id  Can be left blank if unknown """
        return self._doc_id

    @doc_id.setter
    def doc_id(self, value: Optional[str]) -> None:
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
    def doc(self) -> Optional[Document]:
        """ Access the parent doc of this span. """
        return self._doc

    @doc.setter
    def doc(self, value: Optional[Document]) -> None:
        """ Set the parent doc of this span. """
        self._doc = value

    @property
    def text(self) -> Optional[str]:
        """ Access the raw text for this sentence. """
        return self._text

    @text.setter
    def text(self, value: Optional[str]) -> None:
        """ Set the raw text for this sentence. """
        self._text = value

    @property
    def dependencies(self) -> DependencyEdges:
        """ Access list of dependencies for this sentence. """
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value: DependencyEdges) -> None:
        """ Set the list of dependencies for this sentence. """
        self._dependencies = value

    @property
    def tokens(self) -> list[Token]:
        """ Access the list of tokens for this sentence. """
        return self._tokens

    @tokens.setter
    def tokens(self, value: list[Token]) -> None:
        """ Set the list of tokens for this sentence. """
        self._tokens = value

    @property
    def words(self) -> list[Word]:
        """ Access the list of words for this sentence. """
        return self._words

    @words.setter
    def words(self, value: list[Word]) -> None:
        """ Set the list of words for this sentence. """
        self._words = value

    @property
    def empty_words(self) -> list[Word]:
        """ Access the list of words for this sentence. """
        return self._empty_words

    @empty_words.setter
    def empty_words(self, value: list[Word]) -> None:
        """ Set the list of words for this sentence. """
        self._empty_words = value

    @property
    def all_words(self) -> list[Word]:
        """ Access the list of words + empty words for this sentence. """
        words = self._words
        empty_words = self._empty_words

        all_words = sorted(words + empty_words,
                           key=lambda x:(x.id,) if isinstance(x.id, int) else x.id)

        return all_words

    @property
    def ents(self) -> list[Span]:
        """ Access the list of entities in this sentence. """
        return self._ents

    @ents.setter
    def ents(self, value: list[Span]) -> None:
        """ Set the list of entities in this sentence. """
        self._ents = value

    @property
    def entities(self) -> list[Span]:
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value: list[Span]) -> None:
        """ Set the list of entities in this sentence. """
        self._ents = value

    def build_ents(self) -> list[Span]:
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
    def sentiment(self) -> Optional[Union[int, str]]:
        """ Returns the sentiment value for this sentence """
        return self._sentiment

    @sentiment.setter
    def sentiment(self, value: Optional[Union[int, str]]) -> None:
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
    def constituency(self) -> Optional[Union[tree_reader.Tree, str]]:
        """ Returns the constituency tree for this sentence """
        return self._constituency

    @constituency.setter
    def constituency(self, value: Optional[Union[tree_reader.Tree, str]]) -> None:
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
    def comments(self) -> list[str]:
        """ Returns CoNLL-style comments for this sentence """
        return self._comments

    def add_comment(self, comment: str) -> None:
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

    def rebuild_dependencies(self) -> None:
        # rebuild dependencies if there is dependency info
        is_complete_dependencies = all(word.head is not None and word.deprel is not None for word in self.words)
        is_complete_words = (len(self.words) >= len(self.tokens)) and (len(self.words) == self.words[-1].id)
        if is_complete_dependencies and is_complete_words: self.build_dependencies()

    def build_dependencies(self) -> None:
        """ Build the dependency graph for this sentence. Each dependency graph entry is
        a list of (head, deprel, word).
        """
        self.dependencies = []
        for word in self.words:
            word_head = word.head
            if word_head is None or word.deprel is None:
                raise ValueError(
                    f"Cannot build dependencies for word {word.id} without "
                    "a head and dependency relation"
                )
            if word_head == 0:
                # make a word for the ROOT
                word_entry: TokenEntry = {"id": 0, "text": "ROOT"}
                head = Word(self, word_entry)
            else:
                # id is index in words list + 1
                try:
                    head = self.words[word_head - 1]
                except IndexError as e:
                    raise IndexError("Word head {} is not a valid word index for word {}".format(word_head, word.id)) from e
                if word_head != head.id:
                    raise ValueError("Dependency tree is incorrectly constructed")
            self.dependencies.append((head, word.deprel, word))

    def build_fake_dependencies(self) -> None:
        self.dependencies = []
        for word_idx, word in enumerate(self.words):
            word.head = word_idx   # note that this goes one previous to the index
            word.deprel = "root" if word_idx == 0 else "dep"
            word.deps = "%d:%s" % (word.head, word.deprel)
            self.dependencies.append((word_idx, word.deprel, word))

    def print_dependencies(self, file: Optional[TextIO] = None) -> None:
        """ Print the dependencies for this sentence. """
        for dep_edge in self.dependencies:
            governor = dep_edge[0]
            governor_id = governor if isinstance(governor, int) else governor.id
            print((dep_edge[2].text, governor_id, dep_edge[1]), file=file)

    def dependencies_string(self) -> str:
        """ Dump the dependencies for this sentence into string. """
        dep_string = io.StringIO()
        self.print_dependencies(file=dep_string)
        return dep_string.getvalue().strip()

    def get_roots(self) -> list[Word]:
        """ Return a list of root(s) from a sentence """
        roots = []
        for word in self.words:
            if word.head == 0:
                roots.append(word)
        return roots

    def print_tokens(self, file: Optional[TextIO] = None) -> None:
        """ Print the tokens for this sentence. """
        for tok in self.tokens:
            print(tok.pretty_print(), file=file)

    def tokens_string(self) -> str:
        """ Dump the tokens for this sentence into string. """
        toks_string = io.StringIO()
        self.print_tokens(file=toks_string)
        return toks_string.getvalue().strip()

    def print_words(self, file: Optional[TextIO] = None) -> None:
        """ Print the words for this sentence. """
        for word in self.words:
            print(word.pretty_print(), file=file)

    def words_string(self) -> str:
        """ Dump the words for this sentence into string. """
        wrds_string = io.StringIO()
        self.print_words(file=wrds_string)
        return wrds_string.getvalue().strip()

    def to_dict(self) -> list[TokenEntry]:
        """ Dumps the sentence into a list of dictionary for each token in the sentence.
        """
        ret: list[TokenEntry] = []
        empty_idx = 0
        for token_idx, token in enumerate(self.tokens):
            while (empty_idx < len(self._empty_words)
                   and _empty_word_id(self._empty_words[empty_idx])[0] < token.id[0]):
                ret.append(_require_token_entry(
                    self._empty_words[empty_idx].to_dict(),
                    "Word.to_dict()",
                ))
                empty_idx += 1
            ret += [
                _require_token_entry(entry, "Token.to_dict()")
                for entry in token.to_dict()
            ]
        for empty_word in self._empty_words[empty_idx:]:
            ret.append(_require_token_entry(
                empty_word.to_dict(),
                "Word.to_dict()",
            ))
        return ret

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec: str) -> str:
        if not spec:
            return str(self)
        if not spec[0] == 'c' and not spec[0] == 'C':
            return str(self)
        if "-o" in spec:
            fields = NO_OFFSETS_OUTPUT_FIELDS
        else:
            fields = DEFAULT_OUTPUT_FIELDS

        pieces = []
        empty_idx = 0
        for token_idx, token in enumerate(self.tokens):
            while (empty_idx < len(self._empty_words)
                   and _empty_word_id(self._empty_words[empty_idx])[0] < token.id[0]):
                pieces.append(self._empty_words[empty_idx].to_conll_text(fields))
                empty_idx += 1
            pieces.append(token.to_conll_text(fields))
        for empty_word in self._empty_words[empty_idx:]:
            pieces.append(empty_word.to_conll_text(fields))

        if spec[0] == 'c':
            return "\n".join(pieces)
        elif spec[0] == 'C':
            tokens = "\n".join(pieces)
            if len(self.comments) > 0:
                text = "\n".join(self.comments)
                return text + "\n" + tokens
            return tokens
        return str(self)

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
            if key in (START_CHAR, END_CHAR, LINE_NUMBER):
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
    if token_dict.get(MISC):
        # avoid appending a blank misc entry.
        # otherwise the resulting misc field in the conll doc will wind up being blank text
        # TODO: potentially need to escape =|\ in the MISC as well
        misc.append(token_dict[MISC])

    # for other items meant to be in the MISC field,
    # we try to operate on those columns in a deterministic order
    # so that the output doesn't change based on the order of keys
    # in the token_dict
    for key in [START_CHAR, END_CHAR, NER]:
        if key in token_dict:
            misc.append("{}={}".format(key, token_dict[key]))

    if COREF_CHAINS in token_dict:
        chains = token_dict[COREF_CHAINS]
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
            misc.append("{}={}".format(COREF_CHAINS, ",".join(misc_chains)))

    if MORPHEMES in token_dict and token_dict[MORPHEMES]:
        misc.append("Morphemes={}".format(",".join(token_dict[MORPHEMES])))

    for key in token_dict.keys():
        if key == ID:
            token_conll[FIELD_TO_IDX[key]] = id_connector.join([str(x) for x in token_dict[key]]) if isinstance(token_dict[key], tuple) else str(token_dict[key])
        elif key == FEATS:
            feats = token_dict[key]
            if feats:
                pieces = feats.split("|")
                pieces = sorted(pieces, key=str.casefold)
                feats = "|".join(pieces)
            token_conll[FIELD_TO_IDX[key]] = str(feats)
        elif key in FIELD_TO_IDX:
            token_conll[FIELD_TO_IDX[key]] = str(token_dict[key])
        elif key == LINE_NUMBER:
            # skip this when converting back for now
            pass
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

    def __init__(self, sentence: Sentence, token_entry: TokenEntry,
                 words: Optional[list[Word]] = None) -> None:
        """
        Construct a token given a dictionary format token entry. Optionally link itself to the corresponding words.
        The owning sentence must be passed in.
        """
        entry_id = token_entry.get(ID)
        self._text = token_entry.get(TEXT)
        if not entry_id:
            raise ValueError('id not included for the token')
        if not self._text:
            raise ValueError('text not included for the token')
        self._id = _normalize_token_id(entry_id)
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
        self._line_number = None

        if self._misc is not None:
            init_from_misc(self)

    @property
    def id(self) -> TokenId:
        """ Access the index of this token. """
        return self._id

    @id.setter
    def id(self, value: TokenId) -> None:
        """ Set the token's id value. """
        self._id = value

    @property
    def manual_expansion(self) -> Optional[bool]:
        """ Access the whether this token was manually expanded. """
        return self._mexp

    @manual_expansion.setter
    def manual_expansion(self, value: Optional[bool]) -> None:
        """ Set the whether this token was manually expanded. """
        self._mexp = value

    @property
    def text(self) -> str:
        """ Access the text of this token. Example: 'The' """
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """ Set the token's text value. Example: 'The' """
        self._text = value

    @property
    def misc(self) -> Optional[str]:
        """ Access the miscellaneousness of this token. """
        return self._misc

    @misc.setter
    def misc(self, value: Optional[str]) -> None:
        """ Set the token's miscellaneousness value. """
        self._misc = value if self._is_null(value) == False else None

    def consolidate_whitespace(self) -> None:
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
    def spaces_before(self) -> str:
        """ SpacesBefore for the token. Translated from the MISC fields """
        return self._spaces_before

    @spaces_before.setter
    def spaces_before(self, value: str) -> None:
        self._spaces_before = value

    @property
    def spaces_after(self) -> str:
        """ SpaceAfter or SpacesAfter for the token.  Translated from the MISC field """
        return self._spaces_after

    @spaces_after.setter
    def spaces_after(self, value: str) -> None:
        self._spaces_after = value

    @property
    def words(self) -> list[Word]:
        """ Access the list of syntactic words underlying this token. """
        return self._words

    @words.setter
    def words(self, value: list[Word]) -> None:
        """ Set this token's list of underlying syntactic words. """
        self._words = value
        for w in self._words:
            w.parent = self

    @property
    def line_number(self) -> Optional[int]:
        """ Access the line number from the original document, if set """
        return self._line_number

    @property
    def start_char(self) -> Optional[int]:
        """ Access the start character index for this token in the raw text. """
        return self._start_char

    @property
    def end_char(self) -> Optional[int]:
        """ Access the end character index for this token in the raw text. """
        return self._end_char

    @property
    def ner(self) -> Optional[str]:
        """ Access the NER tag of this token. Example: 'B-ORG'"""
        return self._ner

    @ner.setter
    def ner(self, value: Optional[str]) -> None:
        """ Set the token's NER tag. Example: 'B-ORG'"""
        self._ner = value if self._is_null(value) == False else None

    @property
    def multi_ner(self) -> Optional[MultiNerTags]:
        """ Access the MULTI_NER tag of this token. Example: '(B-ORG, B-DISEASE)'"""
        return self._multi_ner

    @multi_ner.setter
    def multi_ner(self, value: Optional[MultiNerTags]) -> None:
        """ Set the token's MULTI_NER tag. Example: '(B-ORG, B-DISEASE)'"""
        self._multi_ner = value if self._is_null(value) == False else None

    @property
    def sent(self) -> Sentence:
        """ Access the pointer to the sentence that this token belongs to. """
        return self._sent

    @sent.setter
    def sent(self, value: Sentence) -> None:
        """ Set the pointer to the sentence that this token belongs to. """
        self._sent = value

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec: str) -> str:
        if spec == 'C':
            return "\n".join(self.to_conll_text(DEFAULT_OUTPUT_FIELDS))
        elif spec == 'P':
            return self.pretty_print()
        else:
            return str(self)

    def to_conll_text(self, fields: FieldNames = DEFAULT_OUTPUT_FIELDS) -> str:
        return "\n".join(dict_to_conll_text(x) for x in self.to_dict(fields))

    def to_dict(
        self,
        fields: FieldNames = DEFAULT_OUTPUT_FIELDS,
    ):
        """ Dumps the token into a list of dictionary for this token with its extended words
        if the token is a multi-word token.
        """
        ret = []
        if len(self.id) > 1:
            token_dict = {}
            for field in fields:
                if getattr(self, field, None) is not None:
                    token_dict[field] = getattr(self, field)
            if MISC in fields:
                needs_sorting = False

                spaces_after = self.spaces_after
                if spaces_after is not None and spaces_after != ' ':
                    space_misc = space_after_to_misc(spaces_after)
                    current_misc = token_dict.get(MISC)
                    if isinstance(current_misc, str) and current_misc:
                        token_dict[MISC] = current_misc + "|" + space_misc
                        needs_sorting = True
                    else:
                        token_dict[MISC] = space_misc

                spaces_before = self.spaces_before
                if spaces_before is not None and spaces_before != '':
                    space_misc = space_before_to_misc(spaces_before)
                    current_misc = token_dict.get(MISC)
                    if isinstance(current_misc, str) and current_misc:
                        token_dict[MISC] = current_misc + "|" + space_misc
                        needs_sorting = True
                    else:
                        token_dict[MISC] = space_misc
                if needs_sorting:
                    current_misc = token_dict.get(MISC)
                    if isinstance(current_misc, str):
                        pieces = sorted(current_misc.split("|"))
                        token_dict[MISC] = "|".join(pieces)

            ret.append(token_dict)
        for word in self.words:
            word_dict = word.to_dict(fields)
            if len(self.id) == 1 and NER in fields and getattr(self, NER) is not None: # propagate NER label to Word if it is a single-word token
                word_dict[NER] = getattr(self, NER)
            if len(self.id) == 1 and MULTI_NER in fields and getattr(self, MULTI_NER) is not None: # propagate MULTI_NER label to Word if it is a single-word token
                word_dict[MULTI_NER] = getattr(self, MULTI_NER)
            if len(self.id) == 1 and MISC in fields:
                needs_sorting = False

                spaces_after = self.spaces_after
                if spaces_after is not None and spaces_after != ' ':
                    space_misc = space_after_to_misc(spaces_after)
                    current_misc = word_dict.get(MISC)
                    if isinstance(current_misc, str) and current_misc:
                        word_dict[MISC] = current_misc + "|" + space_misc
                        needs_sorting = True
                    else:
                        word_dict[MISC] = space_misc

                spaces_before = self.spaces_before
                if spaces_before is not None and spaces_before != '':
                    space_misc = space_before_to_misc(spaces_before)
                    current_misc = word_dict.get(MISC)
                    if isinstance(current_misc, str) and current_misc:
                        word_dict[MISC] = current_misc + "|" + space_misc
                        needs_sorting = True
                    else:
                        word_dict[MISC] = space_misc
                if needs_sorting:
                    current_misc = word_dict.get(MISC)
                    if isinstance(current_misc, str):
                        pieces = sorted(current_misc.split("|"))
                        word_dict[MISC] = "|".join(pieces)
            ret.append(word_dict)
        return ret

    def pretty_print(self) -> str:
        """ Print this token with its extended words in one line. """
        return f"<{self.__class__.__name__} id={'-'.join([str(x) for x in self.id])};words=[{', '.join([word.pretty_print() for word in self.words])}]>"

    def _is_null(self, value) -> bool:
        return (value is None) or (value == '_')

    def is_mwt(self) -> bool:
        return len(self.words) > 1

class Word(StanzaObject):
    """ A word class that stores attributes of a word.
    """

    def __init__(
            self,
            sentence: Sentence,
            word_entry: Union[TokenEntry, EmptyWordEntry],
        ) -> None:
        """ Construct a word given a dictionary format word entry.
        """
        entry_id = word_entry.get(ID, None)
        self._text = word_entry.get(TEXT, None)
        assert entry_id is not None and self._text is not None, 'id and text should be included for the word. {}'.format(word_entry)
        normalized_id = _normalize_token_id(entry_id)
        self._id = normalized_id[0] if len(normalized_id) == 1 else normalized_id

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
        self._line_number = None
        self._morphemes = word_entry.get(MORPHEMES, None)

        if self._misc is not None:
            init_from_misc(self)

        # use the setter, which will go up to the sentence and set the
        # dependencies on that graph
        self.deps = word_entry.get(DEPS, None)

    @property
    def manual_expansion(self) -> Optional[bool]:
        """ Access the whether this token was manually expanded. """
        return self._mexp

    @manual_expansion.setter
    def manual_expansion(self, value: Optional[bool]) -> None:
        """ Set the whether this token was manually expanded. """
        self._mexp = value

    @property
    def id(self) -> WordId:
        """ Access the index of this word. """
        return self._id

    @id.setter
    def id(self, value: WordId) -> None:
        """ Set the word's index value. """
        self._id = value

    @property
    def text(self) -> str:
        """ Access the text of this word. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """ Set the word's text value. Example: 'The'"""
        self._text = value

    @property
    def lemma(self) -> Optional[str]:
        """ Access the lemma of this word. """
        return self._lemma

    @lemma.setter
    def lemma(self, value: Optional[str]) -> None:
        """ Set the word's lemma value. """
        self._lemma = value if self._is_null(value) == False or self._text == '_' else None

    @property
    def upos(self) -> Optional[str]:
        """ Access the universal part-of-speech of this word. Example: 'NOUN'"""
        return self._upos

    @upos.setter
    def upos(self, value: Optional[str]) -> None:
        """ Set the word's universal part-of-speech value. Example: 'NOUN'"""
        self._upos = value if self._is_null(value) == False else None

    @property
    def xpos(self) -> Optional[str]:
        """ Access the treebank-specific part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @xpos.setter
    def xpos(self, value: Optional[str]) -> None:
        """ Set the word's treebank-specific part-of-speech value. Example: 'NNP'"""
        self._xpos = value if self._is_null(value) == False else None

    @property
    def feats(self) -> Optional[str]:
        """ Access the morphological features of this word. Example: 'Gender=Fem'"""
        return self._feats

    @feats.setter
    def feats(self, value: Optional[str]) -> None:
        """ Set this word's morphological features. Example: 'Gender=Fem'"""
        self._feats = value if self._is_null(value) == False else None

    @property
    def head(self) -> Optional[int]:
        """ Access the id of the governor of this word. """
        return self._head

    @head.setter
    def head(self, value: Union[int, str, None]) -> None:
        """ Set the word's governor id value. """
        self._head = None if value is None or value == '_' else int(value)

    @property
    def deprel(self) -> Optional[str]:
        """ Access the dependency relation of this word. Example: 'nmod'"""
        return self._deprel

    @deprel.setter
    def deprel(self, value: Optional[str]) -> None:
        """ Set the word's dependency relation value. Example: 'nmod'"""
        self._deprel = value if self._is_null(value) == False else None

    @property
    def deps(self) -> Optional[str]:
        """ Access the dependencies of this word. """
        graph = self._sent._enhanced_dependencies
        if graph is None or not graph.has_node(self.id):
            return None

        data: list[str] = []
        predecessor_nodes: list[Union[int, tuple[int, int]]] = []
        for predecessor in graph.predecessors(self.id):
            if isinstance(predecessor, int) and not isinstance(predecessor, bool):
                predecessor_nodes.append(predecessor)
            elif isinstance(predecessor, tuple) and len(predecessor) == 2:
                first, second = predecessor
                if (not isinstance(first, int)
                        or isinstance(first, bool)
                        or not isinstance(second, int)
                        or isinstance(second, bool)):
                    raise TypeError(
                        f"Unexpected enhanced dependency node {predecessor!r}"
                    )
                predecessor_nodes.append((first, second))
            else:
                raise TypeError(
                    f"Unexpected enhanced dependency node {predecessor!r}"
                )
        predecessors = sorted(
            predecessor_nodes,
            key=lambda x: x if isinstance(x, tuple) else (x,),
        )
        for parent in predecessors:
            edge_data = graph.get_edge_data(parent, self.id)
            if edge_data is None:
                continue
            dependencies: list[str] = []
            for dependency in edge_data:
                if not isinstance(dependency, str):
                    raise TypeError(
                        f"Unexpected enhanced dependency relation {dependency!r}"
                    )
                dependencies.append(dependency)
            deps = sorted(dependencies)
            for dep in deps:
                if isinstance(parent, int):
                    data.append("%d:%s" % (parent, dep))
                else:
                    data.append("%d.%d:%s" % (parent[0], parent[1], dep))
        if not data:
            return None

        return "|".join(data)

    @deps.setter
    def deps(self, value: Optional[Union[str, Dependencies]]) -> None:
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
            dependency_values: Sequence[Union[str, DependencyParts]] = value.split("|")
        else:
            dependency_values = value

        normalized_dependencies: list[DependencyParts] = []
        for dependency in dependency_values:
            if isinstance(dependency, str):
                pieces = dependency.split(":", maxsplit=1)
                if len(pieces) != 2:
                    raise ValueError(
                        f"Enhanced dependency must contain a relation: {dependency!r}"
                    )
                normalized_dependencies.append((pieces[0], pieces[1]))
            else:
                normalized_dependencies.append(dependency)

        for parent_text, dep in normalized_dependencies:
            # we have to match the format of the IDs.  since the IDs
            # of the words are int if they aren't empty words, we need
            # to convert single int IDs into int instead of tuple
            parent_parts = tuple(map(int, parent_text.split(".", maxsplit=1)))
            parent: WordId
            if len(parent_parts) == 1:
                parent = parent_parts[0]
            else:
                parent = (parent_parts[0], parent_parts[1])
            graph.add_edge(parent, self.id, dep)

    @property
    def misc(self) -> Optional[str]:
        """ Access the miscellaneousness of this word. """
        return self._misc

    @misc.setter
    def misc(self, value: Optional[str]) -> None:
        """ Set the word's miscellaneousness value. """
        self._misc = value if self._is_null(value) == False else None

    @property
    def line_number(self) -> Optional[int]:
        """ Access the line number from the original document, if set """
        return self._line_number

    @property
    def start_char(self) -> Optional[int]:
        """ Access the start character index for this token in the raw text. """
        return self._start_char

    @start_char.setter
    def start_char(self, value: Optional[int]) -> None:
        self._start_char = value

    @property
    def end_char(self) -> Optional[int]:
        """ Access the end character index for this token in the raw text. """
        return self._end_char

    @end_char.setter
    def end_char(self, value: Optional[int]) -> None:
        self._end_char = value

    @property
    def parent(self) -> Optional[Token]:
        """ Access the parent token of this word. In the case of a multi-word token, a token can be the parent of
        multiple words. Note that this should return a reference to the parent token object.
        """
        return self._parent

    @parent.setter
    def parent(self, value: Optional[Token]) -> None:
        """ Set this word's parent token. In the case of a multi-word token, a token can be the parent of
        multiple words. Note that value here should be a reference to the parent token object.
        """
        self._parent = value

    @property
    def pos(self) -> Optional[str]:
        """ Access the universal part-of-speech of this word. Example: 'NOUN'"""
        return self._upos

    @pos.setter
    def pos(self, value: Optional[str]) -> None:
        """ Set the word's universal part-of-speech value. Example: 'NOUN'"""
        self._upos = value if self._is_null(value) == False else None

    @property
    def coref_chains(self) -> Optional[list[CorefAttachment]]:
        """
        coref_chains points to a list of CorefChain namedtuple, which has a list of mentions and a representative mention.

        Useful for disambiguating words such as "him" (in languages where coref is available)

        Theoretically it is possible for multiple corefs to occur at the same word.  For example,
          "Chris Manning's NLP Group"
        could have "Chris Manning" and "Chris Manning's NLP Group" as overlapping entities
        """
        return self._coref_chains

    @coref_chains.setter
    def coref_chains(self, chain: Optional[list[CorefAttachment]]) -> None:
        """ Set the backref for the coref chains """
        self._coref_chains = chain

    @property
    def morphemes(self) -> Optional[list[str]]:
        """Access morpheme segments produced by the morphseg processor."""
        return self._morphemes

    @morphemes.setter
    def morphemes(self, value: Optional[list[str]]) -> None:
        self._morphemes = value

    @property
    def sent(self) -> Sentence:
        """ Access the pointer to the sentence that this word belongs to. """
        return self._sent

    @sent.setter
    def sent(self, value: Sentence) -> None:
        """ Set the pointer to the sentence that this word belongs to. """
        self._sent = value

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec: str) -> str:
        if spec == 'C':
            return self.to_conll_text(DEFAULT_OUTPUT_FIELDS)
        elif spec == 'P':
            return self.pretty_print()
        else:
            return str(self)

    def to_conll_text(self, fields: FieldNames = DEFAULT_OUTPUT_FIELDS) -> str:
        """
        Turn a word into a conll representation (10 column tab separated)
        """
        token_dict = self.to_dict(fields)
        return dict_to_conll_text(token_dict, '.')

    def to_dict(
        self,
        fields: FieldNames = DEFAULT_OUTPUT_FIELDS,
    ):
        """ Dumps the word into a dictionary.
        """
        word_dict = {}
        for field in fields:
            if getattr(self, field, None) is not None:
                word_dict[field] = getattr(self, field)
        return word_dict

    def pretty_print(self) -> str:
        """ Print the word in one line. """
        features = [ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL]
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])
        return f"<{self.__class__.__name__} {feature_str}>"

    def _is_null(self, value) -> bool:
        return (value is None) or (value == '_')


class Span(StanzaObject):
    """ A span class that stores attributes of a textual span. A span can be typed.
    A range of objects (e.g., entity mentions) can be represented as spans.
    """

    def __init__(self, span_entry: Optional[SpanEntry] = None,
                 tokens: Optional[list[Token]] = None,
                 type: Optional[str] = None, doc: Optional[Document] = None,
                 sent: Optional[Sentence] = None) -> None:
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

    def init_from_entry(self, span_entry: SpanEntry) -> None:
        self.text = span_entry.get(TEXT, None)
        self.type = span_entry.get(TYPE, None)
        self.start_char = span_entry.get(START_CHAR, None)
        self.end_char = span_entry.get(END_CHAR, None)

    def init_from_tokens(self, tokens: list[Token], type: Optional[str]) -> None:
        assert isinstance(tokens, list), 'Tokens must be provided as a list to construct a span.'
        assert len(tokens) > 0, "Tokens of a span cannot be an empty list."
        self.tokens = tokens
        self.type = type
        # load start and end char offsets from tokens
        self.start_char = self.tokens[0].start_char
        self.end_char = self.tokens[-1].end_char
        document = self.doc
        document_text = document.text if document is not None else None
        if isinstance(document_text, str):
            self.text = document_text[self.start_char:self.end_char]
        elif tokens[0].sent is tokens[-1].sent:
            sentence = tokens[0].sent
            sentence_text = sentence.text
            if (tokens[-1].end_char is not None
                    and tokens[0].start_char is not None
                    and sentence.tokens[0].start_char is not None
                    and isinstance(sentence_text, str)):
                text_start = tokens[0].start_char - sentence.tokens[0].start_char
                text_end = tokens[-1].end_char - sentence.tokens[0].start_char
                self.text = sentence_text[text_start:text_end]
            else:
                text = []
                for token in tokens:
                    text.append(token.text)
                    text.append(token.spaces_after)
                self.text = "".join(text[:-1])
        else:
            # TODO: do any spans ever cross sentences?
            raise RuntimeError("Document text does not exist, and the span tested crosses two sentences, so it is impossible to extract the entity text!")
        # collect the words of the span following tokens
        self.words = [w for t in tokens for w in t.words]
        # set the sentence back-pointer to point to the sentence of the first token
        self.sent = tokens[0].sent

    @property
    def doc(self) -> Optional[Document]:
        """ Access the parent doc of this span. """
        return self._doc

    @doc.setter
    def doc(self, value: Optional[Document]) -> None:
        """ Set the parent doc of this span. """
        self._doc = value

    @property
    def text(self) -> Optional[str]:
        """ Access the text of this span. Example: 'Stanford University'"""
        return self._text

    @text.setter
    def text(self, value: Optional[str]) -> None:
        """ Set the span's text value. Example: 'Stanford University'"""
        self._text = value

    @property
    def tokens(self) -> list[Token]:
        """ Access reference to a list of tokens that correspond to this span. """
        return self._tokens

    @tokens.setter
    def tokens(self, value: list[Token]) -> None:
        """ Set the span's list of tokens. """
        self._tokens = value

    @property
    def words(self) -> list[Word]:
        """ Access reference to a list of words that correspond to this span. """
        return self._words

    @words.setter
    def words(self, value: list[Word]) -> None:
        """ Set the span's list of words. """
        self._words = value

    @property
    def type(self) -> Optional[str]:
        """ Access the type of this span. Example: 'PERSON'"""
        return self._type

    @type.setter
    def type(self, value: Optional[str]) -> None:
        """ Set the type of this span. """
        self._type = value

    @property
    def start_char(self) -> Optional[int]:
        """ Access the start character offset of this span. """
        return self._start_char

    @start_char.setter
    def start_char(self, value: Optional[int]) -> None:
        """ Set the start character offset of this span. """
        self._start_char = value

    @property
    def end_char(self) -> Optional[int]:
        """ Access the end character offset of this span. """
        return self._end_char

    @end_char.setter
    def end_char(self, value: Optional[int]) -> None:
        """ Set the end character offset of this span. """
        self._end_char = value

    @property
    def sent(self) -> Optional[Sentence]:
        """ Access the pointer to the sentence that this span belongs to. """
        return self._sent

    @sent.setter
    def sent(self, value: Optional[Sentence]) -> None:
        """ Set the pointer to the sentence that this span belongs to. """
        self._sent = value

    def to_dict(self) -> SpanDict:
        """ Dumps the span into a dictionary. """
        return {
            "text": self.text,
            "type": self.type,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def pretty_print(self) -> str:
        """ Print the span in one line. """
        span_dict = self.to_dict()
        feature_str = ";".join(["{}={}".format(k,v) for k,v in span_dict.items()])
        return f"<{self.__class__.__name__} {feature_str}>"
