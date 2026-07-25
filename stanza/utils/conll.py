"""
Utility functions for the loading and conversion of CoNLL-format files.
"""
import os
import io
from collections.abc import Iterable, Sequence
from typing import BinaryIO, Literal, Optional, TextIO, TypedDict, Union
from zipfile import ZipFile, ZipInfo

from stanza.models.common.doc import (
    Document,
    EmptyWordEntry,
    EmptyWordId,
    TokenEntry,
    TokenId,
)
from stanza.models.common.doc import ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
from stanza.models.common.doc import FIELD_TO_IDX, FIELD_NUM
from stanza.models.common.doc import LINE_NUMBER


CoNLLRow = list[str]
CoNLLSentence = list[CoNLLRow]
CoNLLDocument = list[CoNLLSentence]
TokenDocument = list[list[TokenEntry]]
EmptyWordDocument = list[list[EmptyWordEntry]]
SentenceComments = list[list[str]]
PathInput = Union[
    str,
    bytes,
    os.PathLike[str],
    os.PathLike[bytes],
]
ArchiveMember = Union[str, ZipInfo]
ZipSource = Union[PathInput, BinaryIO]
OutputTarget = Union[PathInput, TextIO]
IdConnector = Literal["-", "."]


class _RequiredRawTokenEntry(TypedDict):
    id: str
    text: str


class _RawTokenEntry(_RequiredRawTokenEntry, total=False):
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str


def _parse_token_id(value: str, connector: IdConnector) -> TokenId:
    pieces = value.split(connector, maxsplit=1)
    if len(pieces) == 1:
        return (int(pieces[0]),)
    return int(pieces[0]), int(pieces[1])


_NormalizedTokenEntry = Union[TokenEntry, EmptyWordEntry]


def _copy_optional_token_fields(
        raw_entry: _RawTokenEntry,
        entry: _NormalizedTokenEntry,
    ) -> None:
    if LEMMA in raw_entry:
        entry[LEMMA] = raw_entry[LEMMA]
    if UPOS in raw_entry:
        entry[UPOS] = raw_entry[UPOS]
    if XPOS in raw_entry:
        entry[XPOS] = raw_entry[XPOS]
    if FEATS in raw_entry:
        entry[FEATS] = raw_entry[FEATS]
    if HEAD in raw_entry:
        entry[HEAD] = raw_entry[HEAD]
    if DEPREL in raw_entry:
        entry[DEPREL] = raw_entry[DEPREL]
    if DEPS in raw_entry:
        entry[DEPS] = raw_entry[DEPS]
    if MISC in raw_entry:
        entry[MISC] = raw_entry[MISC]


def _normalized_token_entry(
        raw_entry: _RawTokenEntry,
        token_id: TokenId,
    ) -> TokenEntry:
    entry: TokenEntry = {
        ID: token_id,
        TEXT: raw_entry[TEXT],
    }
    _copy_optional_token_fields(raw_entry, entry)
    return entry


def _normalized_empty_word_entry(
        raw_entry: _RawTokenEntry,
        token_id: EmptyWordId,
    ) -> EmptyWordEntry:
    entry: EmptyWordEntry = {
        ID: token_id,
        TEXT: raw_entry[TEXT],
    }
    _copy_optional_token_fields(raw_entry, entry)
    return entry


class CoNLLError(ValueError):
    pass

class CoNLL:

    @staticmethod
    def load_conll(
            f: Iterable[str],
            ignore_gapping: bool = True,
            keep_line_numbers: bool = False,
        ) -> tuple[CoNLLDocument, SentenceComments]:
        """ Load the file or string into the CoNLL-U format data.
        Input: file or string reader, where the data is in CoNLL-U format.
        Output: a tuple whose first element is a list of list of list for each token in each sentence in the data,
        where the innermost list represents all fields of a token; and whose second element is a list of lists for each
        comment in each sentence in the data.
        """
        # f is open() or io.StringIO()
        doc: CoNLLDocument = []
        sent: CoNLLSentence = []
        doc_comments: SentenceComments = []
        sent_comments: list[str] = []
        for line_idx, line in enumerate(f):
            # leave whitespace such as NBSP, in case it is meaningful in the conll-u doc
            line = line.lstrip().rstrip(' \n\r\t')
            if len(line) == 0:
                if len(sent) > 0:
                    doc.append(sent)
                    sent = []
                    doc_comments.append(sent_comments)
                    sent_comments = []
            else:
                if line.startswith('#'): # read comment line
                    sent_comments.append(line)
                    continue
                array = line.split('\t')
                if ignore_gapping and '.' in array[0]:
                    continue
                if len(array) != FIELD_NUM:
                    raise CoNLLError(f"Cannot parse CoNLL line {line_idx+1}: expecting {FIELD_NUM} fields, {len(array)} found at line {line_idx}\n  {array}")
                if keep_line_numbers:
                    if array[-1] == "_":
                        array[-1] = "%s=%d" % (LINE_NUMBER, line_idx)
                    else:
                        array[-1] = "%s|%s=%d" % (array[-1], LINE_NUMBER, line_idx)
                sent.append(array)
        if len(sent) > 0:
            doc.append(sent)
            doc_comments.append(sent_comments)
        return doc, doc_comments

    @staticmethod
    def convert_conll(
            doc_conll: Sequence[Sequence[Sequence[str]]],
        ) -> tuple[TokenDocument, EmptyWordDocument]:
        """ Convert the CoNLL-U format input data to a dictionary format output data.
        Input: list of token fields loaded from the CoNLL-U format data, where the outmost list represents a list of sentences, and the inside list represents all fields of a token.
        Output: a list of list of dictionaries for each token in each sentence in the document.
        """
        doc_dict: TokenDocument = []
        doc_empty: EmptyWordDocument = []
        for sent_idx, sent_conll in enumerate(doc_conll):
            sent_dict: list[TokenEntry] = []
            sent_empty: list[EmptyWordEntry] = []
            for token_idx, token_conll in enumerate(sent_conll):
                try:
                    raw_entry = CoNLL.convert_conll_token(token_conll)
                    raw_id = raw_entry[ID]
                    connector: IdConnector = "." if "." in raw_id else "-"
                    token_id = _parse_token_id(raw_id, connector)
                except ValueError as e:
                    raise CoNLLError("Could not process sentence %d token %d:\n%s\n%s" % (sent_idx, token_idx, token_conll, str(e))) from e
                if connector == ".":
                    if len(token_id) != 2:
                        raise CoNLLError(
                            f"Empty word ID {raw_id!r} did not have two components"
                        )
                    sent_empty.append(_normalized_empty_word_entry(
                        raw_entry,
                        (token_id[0], token_id[1]),
                    ))
                else:
                    sent_dict.append(
                        _normalized_token_entry(raw_entry, token_id)
                    )
            doc_dict.append(sent_dict)
            doc_empty.append(sent_empty)
        return doc_dict, doc_empty

    @staticmethod
    def convert_dict(
            doc_dict: Iterable[Sequence[TokenEntry]],
        ) -> CoNLLDocument:
        """ Convert the dictionary format input data to the CoNLL-U format output data.

        This is the reverse function of `convert_conll`, but does not include sentence level annotations or comments.

        Can call this on a Document using `CoNLL.convert_dict(doc.to_dict())`

        Input: dictionary format data, which is a list of list of dictionaries for each token in each sentence in the data.
        Output: CoNLL-U format data as a list of list of list for each token in each sentence in the data.
        """
        sentences_input = list(doc_dict)
        if len(sentences_input) == 0:
            return []
        doc = Document(sentences_input)
        text = "{:c}".format(doc)
        sentences = text.split("\n\n")
        doc_conll = [[x.split("\t") for x in sentence.split("\n")] for sentence in sentences]
        return doc_conll

    @staticmethod
    def convert_conll_token(
            token_conll: Sequence[str],
        ) -> _RawTokenEntry:
        """ Convert the CoNLL-U format input token to the dictionary format output token.
        Input: a list of all CoNLL-U fields for the token.
        Output: a dictionary that maps from field name to value.
        """
        if len(token_conll) != FIELD_NUM:
            raise CoNLLError(
                f"Expected {FIELD_NUM} CoNLL-U fields, got {len(token_conll)}"
            )

        text = token_conll[FIELD_TO_IDX[TEXT]]
        token_dict: _RawTokenEntry = {
            ID: token_conll[FIELD_TO_IDX[ID]],
            TEXT: text,
        }

        lemma = token_conll[FIELD_TO_IDX[LEMMA]]
        if lemma != "_" or text == "_":
            token_dict[LEMMA] = lemma

        upos = token_conll[FIELD_TO_IDX[UPOS]]
        if upos != "_":
            token_dict[UPOS] = upos

        xpos = token_conll[FIELD_TO_IDX[XPOS]]
        if xpos != "_":
            token_dict[XPOS] = xpos

        feats = token_conll[FIELD_TO_IDX[FEATS]]
        if feats not in ("", "_"):
            token_dict[FEATS] = feats

        head = token_conll[FIELD_TO_IDX[HEAD]]
        if head != "_":
            token_dict[HEAD] = int(head)

        deprel = token_conll[FIELD_TO_IDX[DEPREL]]
        if deprel != "_":
            token_dict[DEPREL] = deprel

        deps = token_conll[FIELD_TO_IDX[DEPS]]
        if deps != "_":
            token_dict[DEPS] = deps

        misc = token_conll[FIELD_TO_IDX[MISC]]
        if misc != "_":
            token_dict[MISC] = misc

        return token_dict

    @staticmethod
    def conll2dict(
            input_file: Optional[Union[PathInput, ZipInfo]] = None,
            input_str: Optional[str] = None,
            ignore_gapping: bool = True,
            zip_file: Optional[ZipSource] = None,
            keep_line_numbers: bool = False,
        ) -> tuple[TokenDocument, SentenceComments, EmptyWordDocument]:
        """ Load the CoNLL-U format data from file or string into lists of dictionaries.
        """
        if (input_file is None) == (input_str is None):
            raise ValueError("Specify exactly one of input_file and input_str")
        if zip_file is not None and input_file is None:
            raise ValueError("input_file must be provided when zip_file is set")

        if input_str is not None:
            infile = io.StringIO(input_str)
            doc_conll, doc_comments = CoNLL.load_conll(infile, ignore_gapping, keep_line_numbers)
        elif zip_file is not None:
            if input_file is None:
                raise RuntimeError("input_file validation failed")
            if isinstance(zip_file, (str, bytes, os.PathLike)):
                archive_source: ZipSource = os.fsdecode(zip_file)
            else:
                archive_source = zip_file
            with ZipFile(archive_source) as zin:
                if isinstance(input_file, ZipInfo):
                    archive_member: ArchiveMember = input_file
                else:
                    member_path = os.fspath(input_file)
                    if not isinstance(member_path, str):
                        raise TypeError(
                            "A zip archive member name must be a string or ZipInfo"
                        )
                    archive_member = member_path
                with zin.open(archive_member) as fin:
                    doc_conll, doc_comments = CoNLL.load_conll(io.TextIOWrapper(fin, encoding="utf-8"), ignore_gapping, keep_line_numbers)
        else:
            if input_file is None:
                raise RuntimeError("input_file validation failed")
            if isinstance(input_file, ZipInfo):
                raise TypeError(
                    "A ZipInfo input_file requires the zip_file argument"
                )
            with open(input_file, encoding='utf-8') as fin:
                doc_conll, doc_comments = CoNLL.load_conll(fin, ignore_gapping, keep_line_numbers)

        doc_dict, doc_empty = CoNLL.convert_conll(doc_conll)
        return doc_dict, doc_comments, doc_empty

    @staticmethod
    def conll2doc(
            input_file: Optional[Union[PathInput, ZipInfo]] = None,
            input_str: Optional[str] = None,
            ignore_gapping: bool = True,
            zip_file: Optional[ZipSource] = None,
            keep_line_numbers: bool = False,
        ) -> Document:
        doc_dict, doc_comments, doc_empty = CoNLL.conll2dict(input_file, input_str, ignore_gapping, zip_file=zip_file, keep_line_numbers=keep_line_numbers)
        return Document(doc_dict, text=None, comments=doc_comments, empty_sentences=doc_empty)

    @staticmethod
    def conll2multi_docs(
            input_file: Optional[Union[PathInput, ZipInfo]] = None,
            input_str: Optional[str] = None,
            ignore_gapping: bool = True,
            zip_file: Optional[ZipSource] = None,
        ) -> list[Document]:
        doc_dict, doc_comments, doc_empty = CoNLL.conll2dict(input_file, input_str, ignore_gapping, zip_file=zip_file)

        docs: list[Document] = []
        current_doc: list[list[TokenEntry]] = []
        current_comments: SentenceComments = []
        current_empty: EmptyWordDocument = []
        current_doc_id: Optional[str] = None
        for sentence, comments, empty in zip(doc_dict, doc_comments, doc_empty):
            for comment in comments:
                if comment.startswith("# doc_id =") or comment.startswith("# newdoc id ="):
                    doc_id = comment.split("=", maxsplit=1)[1]
                    if len(current_doc) == 0:
                        current_doc_id = doc_id
                    elif doc_id != current_doc_id:
                        new_doc = Document(current_doc, text=None, comments=current_comments, empty_sentences=current_empty)
                        if current_doc_id != None:
                            for i in new_doc.sentences:
                                i.doc_id = current_doc_id.strip()
                        docs.append(new_doc)
                        current_doc_id = doc_id
                    else:
                        continue
                    current_doc = [sentence]
                    current_comments = [comments]
                    current_empty = [empty]
                    break
            else: # no comments defined a new doc_id, so just add it to the current document
                current_doc.append(sentence)
                current_comments.append(comments)
                current_empty.append(empty)
        if len(current_doc) > 0:
            new_doc = Document(current_doc, text=None, comments=current_comments, empty_sentences=current_empty)
            if current_doc_id != None:
                for i in new_doc.sentences:
                    i.doc_id = current_doc_id.strip()
            docs.append(new_doc)

        return docs

    @staticmethod
    def dict2conll(
            doc_dict: Sequence[Sequence[TokenEntry]],
            filename: OutputTarget,
        ) -> None:
        """
        Convert the dictionary format input data to the CoNLL-U format output data and write to a file.
        """
        doc = Document(doc_dict)
        CoNLL.write_doc2conll(doc, filename)


    @staticmethod
    def write_doc2conll(
            doc: Document,
            filename: OutputTarget,
            mode: str = 'w',
            encoding: str = 'utf-8',
        ) -> None:
        """
        Writes the doc as a conll file to the given file.

        If passed a string, that filename will be opened.  Otherwise, filename.write() will be called.

        Note that the output needs an extra \n\n at the end to be a legal output file
        """
        if isinstance(filename, (str, bytes, os.PathLike)):
            with open(filename, mode, encoding=encoding) as outfile:
                outfile.write("{:C}\n\n".format(doc))
        else:
            filename.write("{:C}\n\n".format(doc))
