"""
Utility functions for the loading and conversion of CoNLL-format files.
"""
import os
import io
import re
import warnings
from zipfile import ZipFile

from stanza.models.common.doc import Document
from stanza.models.common.doc import ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, NER, START_CHAR, END_CHAR
from stanza.models.common.doc import FIELD_TO_IDX, FIELD_NUM
from stanza.models.common.doc import LINE_NUMBER
from stanza.models.common.utils import misc_to_space_after, misc_to_space_before

class CoNLLError(ValueError):
    pass

# MISC annotations which describe whitespace.  These are consumed when
# reconstructing the text of a document, then dropped from the MISC field
# so that the Document can regenerate them from spaces_before / spaces_after
# instead of emitting them twice
SPACE_MISC_PREFIXES = ("spaceafter=", "spacesafter=", "spacesbefore=")

TEXT_COMMENT_PREFIXES = ("# text =", "#text =", "# text=", "#text=")

def group_sentence_units(sent_dict):
    """Group a flat list of CoNLL-U dicts into (token, words) pairs.

    For an MWT, the token is the range line and words are the lines it covers.
    For a regular token, the token and its single word are the same dict, so
    the words list is empty to signal there is nothing extra to annotate.
    """
    units = []
    idx = 0
    while idx < len(sent_dict):
        token = sent_dict[idx]
        idx += 1
        if len(token[ID]) > 1:
            end = token[ID][1]
            words = []
            while idx < len(sent_dict) and len(sent_dict[idx][ID]) == 1 and sent_dict[idx][ID][0] <= end:
                words.append(sent_dict[idx])
                idx += 1
            units.append((token, words))
        else:
            units.append((token, []))
    return units

def unit_space_after(token, words):
    """SpaceAfter / SpacesAfter for a token, preferring the annotation on its final word

    Matches the precedence used by Token.consolidate_whitespace.  UD marks the
    annotation on the range line of an MWT, but some treebanks put it on the
    last word instead, so both are accepted
    """
    for entry in (words[-1] if words else token, token):
        misc = entry.get(MISC)
        if misc and any(piece.lower().startswith(("spaceafter=", "spacesafter=")) for piece in misc.split("|")):
            return misc_to_space_after(misc)
    return " "

def unit_space_before(token, words):
    """SpacesBefore for a token, preferring the annotation on its first word"""
    for entry in (words[0] if words else token, token):
        misc = entry.get(MISC)
        if misc and any(piece.lower().startswith("spacesbefore=") for piece in misc.split("|")):
            return misc_to_space_before(misc)
    return ""

def sentence_text_comment(comments):
    """Return the `# text = ` value for a sentence, or None if there isn't one"""
    for comment in comments:
        if comment.startswith(TEXT_COMMENT_PREFIXES):
            return comment.split("=", 1)[1].strip()
    return None

def align_units_to_text(units, text):
    """Find each token in the sentence's `# text` comment.

    Returns a list of (start, end) offsets relative to the start of the text,
    or None if the tokens cannot be laid over the text.  We require the gaps
    between tokens to be whitespace so that a token matching later in the
    sentence by coincidence is treated as a failure rather than silently
    producing wrong offsets.
    """
    pos = 0
    spans = []
    for token, _ in units:
        form = token[TEXT]
        idx = text.find(form, pos)
        if idx < 0 or text[pos:idx].strip():
            return None
        spans.append((idx, idx + len(form)))
        pos = idx + len(form)
    if text[pos:].strip():
        return None
    return spans

def annotate_mwt_words(token, words, start_char):
    """Split a token's span across the words of an MWT, if the words line up

    Same approach as Document.set_mwt_expansions: words which are not a
    substring split of the token, such as `del` -> `de el`, are left alone
    """
    if not words:
        return
    if len(words) == 1:
        words[0][START_CHAR] = token[START_CHAR]
        words[0][END_CHAR] = token[END_CHAR]
        return
    search_string = "^%s$" % ("\\s*".join("(%s)" % re.escape(word[TEXT]) for word in words))
    match = re.compile(search_string).match(token[TEXT])
    if not match:
        return
    for word_idx, word in enumerate(words):
        word[START_CHAR] = match.start(word_idx + 1) + start_char
        word[END_CHAR] = match.end(word_idx + 1) + start_char

class CoNLL:

    @staticmethod
    def load_conll(f, ignore_gapping=True, keep_line_numbers=False):
        """ Load the file or string into the CoNLL-U format data.
        Input: file or string reader, where the data is in CoNLL-U format.
        Output: a tuple whose first element is a list of list of list for each token in each sentence in the data,
        where the innermost list represents all fields of a token; and whose second element is a list of lists for each
        comment in each sentence in the data.
        """
        # f is open() or io.StringIO()
        doc, sent = [], []
        doc_comments, sent_comments = [], []
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
                    if array[-1] == "_" or array[-1] is None:
                        array[-1] = "%s=%d" % (LINE_NUMBER, line_idx)
                    else:
                        array[-1] = "%s|%s=%d" % (array[-1], LINE_NUMBER, line_idx)
                sent += [array]
        if len(sent) > 0:
            doc.append(sent)
            doc_comments.append(sent_comments)
        return doc, doc_comments

    @staticmethod
    def convert_conll(doc_conll):
        """ Convert the CoNLL-U format input data to a dictionary format output data.
        Input: list of token fields loaded from the CoNLL-U format data, where the outmost list represents a list of sentences, and the inside list represents all fields of a token.
        Output: a list of list of dictionaries for each token in each sentence in the document.
        """
        doc_dict = []
        doc_empty = []
        for sent_idx, sent_conll in enumerate(doc_conll):
            sent_dict = []
            sent_empty = []
            for token_idx, token_conll in enumerate(sent_conll):
                try:
                    token_dict = CoNLL.convert_conll_token(token_conll)
                except ValueError as e:
                    raise CoNLLError("Could not process sentence %d token %d:\n%s\n%s" % (sent_idx, token_idx, token_conll, str(e))) from e
                if '.' in token_dict[ID]:
                    token_dict[ID] = tuple(int(x) for x in token_dict[ID].split(".", maxsplit=1))
                    sent_empty.append(token_dict)
                else:
                    try:
                        token_dict[ID] = tuple(int(x) for x in token_dict[ID].split("-", maxsplit=1))
                    except ValueError as e:
                        raise CoNLLError("Could not process ID %s at sent_idx %d, token_idx %d\nEntire token dict:\n%s" % (token_dict[ID], sent_idx, token_idx, token_dict)) from e
                    sent_dict.append(token_dict)
            doc_dict.append(sent_dict)
            doc_empty.append(sent_empty)
        return doc_dict, doc_empty

    @staticmethod
    def convert_dict(doc_dict):
        """ Convert the dictionary format input data to the CoNLL-U format output data.

        This is the reverse function of `convert_conll`, but does not include sentence level annotations or comments.

        Can call this on a Document using `CoNLL.convert_dict(doc.to_dict())`

        Input: dictionary format data, which is a list of list of dictionaries for each token in each sentence in the data.
        Output: CoNLL-U format data as a list of list of list for each token in each sentence in the data.
        """
        doc = Document(doc_dict)
        text = "{:c}".format(doc)
        sentences = text.split("\n\n")
        doc_conll = [[x.split("\t") for x in sentence.split("\n")] for sentence in sentences]
        return doc_conll

    @staticmethod
    def convert_conll_token(token_conll):
        """ Convert the CoNLL-U format input token to the dictionary format output token.
        Input: a list of all CoNLL-U fields for the token.
        Output: a dictionary that maps from field name to value.
        """
        token_dict = {}
        for field, field_idx in FIELD_TO_IDX.items():
            value = token_conll[field_idx]
            if value == '' and field is FEATS:
                continue
            elif value != '_':
                if field is HEAD:
                    token_dict[field] = int(value)
                else:
                    token_dict[field] = value
        # special case if text is '_'
        if token_conll[FIELD_TO_IDX[TEXT]] == '_':
            token_dict[TEXT] = token_conll[FIELD_TO_IDX[TEXT]]
            token_dict[LEMMA] = token_conll[FIELD_TO_IDX[LEMMA]]
        return token_dict

    @staticmethod
    def conll2dict(input_file=None, input_str=None, ignore_gapping=True, zip_file=None, keep_line_numbers=False):
        """ Load the CoNLL-U format data from file or string into lists of dictionaries.
        """
        assert any([input_file, input_str]) and not all([input_file, input_str]), 'either use input file or input string'
        if zip_file: assert input_file, 'must provide input_file if zip_file is set'

        if input_str:
            infile = io.StringIO(input_str)
            doc_conll, doc_comments = CoNLL.load_conll(infile, ignore_gapping, keep_line_numbers)
        elif zip_file:
            with ZipFile(zip_file) as zin:
                with zin.open(input_file) as fin:
                    doc_conll, doc_comments = CoNLL.load_conll(io.TextIOWrapper(fin, encoding="utf-8"), ignore_gapping, keep_line_numbers)
        else:
            with open(input_file, encoding='utf-8') as fin:
                doc_conll, doc_comments = CoNLL.load_conll(fin, ignore_gapping, keep_line_numbers)

        doc_dict, doc_empty = CoNLL.convert_conll(doc_conll)
        return doc_dict, doc_comments, doc_empty

    @staticmethod
    def add_char_offsets(doc_dict, doc_comments=None, use_text_comments=True):
        """Add start_char & end_char to the tokens and words of a document, and return the text they refer to.

        CoNLL-U files do not record character offsets, so both the offsets and
        the text they index into have to be rebuilt from the tokens.  Where a
        sentence has a `# text` comment, that is the surface form the treebank
        itself claims, so we lay the tokens over it and use the resulting
        positions.  Otherwise, and for any sentence where the tokens cannot be
        aligned with the comment, the text is rebuilt by concatenating the
        tokens using their SpaceAfter / SpacesAfter annotations.

        The whitespace annotations are read but left in the MISC column, and
        spaces_before / spaces_after are deliberately not set on the tokens.
        The Document rewrites MISC from those fields when it has them, which
        would both duplicate the annotations and re-sort the rest of the MISC
        column of any token which has one.

        Modifies doc_dict in place.  Returns the text of the document.
        """
        if doc_comments is None:
            doc_comments = [[] for _ in doc_dict]

        text_pieces = []
        offset = 0
        previous_after = None
        for sent_idx, (sent_dict, comments) in enumerate(zip(doc_dict, doc_comments)):
            units = group_sentence_units(sent_dict)
            if not units:
                continue

            spaces_before = unit_space_before(*units[0])
            spaces_after = [unit_space_after(token, words) for token, words in units]

            sent_text = sentence_text_comment(comments) if use_text_comments else None
            spans = align_units_to_text(units, sent_text) if sent_text is not None else None
            if sent_text is not None and spans is None:
                warnings.warn("Could not align the tokens of sentence %d with its `# text` comment.  Rebuilding that sentence from its tokens instead" % sent_idx)
            if spans is not None:
                # the text comment and the SpaceAfter annotations are required to
                # agree, but treebanks do get this wrong.  the text wins, since
                # that is what the offsets have to index into, but it is worth
                # saying so - the alternative is silently rewriting the MISC
                for unit_idx, (start, end) in enumerate(spans):
                    gap = sent_text[end:spans[unit_idx+1][0]] if unit_idx + 1 < len(spans) else None
                    if gap is not None and gap != spaces_after[unit_idx]:
                        warnings.warn("Sentence %d has whitespace after token %d which does not match its MISC annotation.  Using the `# text` comment" % (sent_idx, unit_idx + 1))
                        break
            if spans is None:
                pieces = []
                spans = []
                local = 0
                for unit_idx, (token, words) in enumerate(units):
                    form = token[TEXT]
                    spans.append((local, local + len(form)))
                    pieces.append(form)
                    local += len(form)
                    if unit_idx < len(units) - 1:
                        pieces.append(spaces_after[unit_idx])
                        local += len(spaces_after[unit_idx])
                sent_text = "".join(pieces)

            # whitespace between the end of the previous sentence and this one.
            # a treebank marks the gap on one side or the other, not both
            if previous_after is None:
                separator = spaces_before
            elif previous_after != " ":
                separator = previous_after
            else:
                separator = spaces_before or " "
            text_pieces.append(separator)
            offset += len(separator)

            for (token, words), (start, end) in zip(units, spans):
                token[START_CHAR] = offset + start
                token[END_CHAR] = offset + end
                annotate_mwt_words(token, words, offset + start)

            text_pieces.append(sent_text)
            offset += len(sent_text)
            previous_after = spaces_after[-1]

        # a document which explicitly annotates whitespace after its final
        # token, such as SpacesAfter=\n, keeps that whitespace.  otherwise the
        # text ends where the last token ends
        if previous_after and previous_after != " ":
            text_pieces.append(previous_after)

        return "".join(text_pieces)

    @staticmethod
    def conll2doc(input_file=None, input_str=None, ignore_gapping=True, zip_file=None, keep_line_numbers=False, reconstruct_text=False):
        """Read a CoNLL-U file or string into a Document

        reconstruct_text: rebuild the text of the document from the tokens so
          that start_char & end_char can be set.  Off by default: it costs
          time and memory, and a document with offsets writes those offsets
          back out in the MISC column, which a document read from a treebank
          would not otherwise have
        """
        doc_dict, doc_comments, doc_empty = CoNLL.conll2dict(input_file, input_str, ignore_gapping, zip_file=zip_file, keep_line_numbers=keep_line_numbers)
        if not reconstruct_text:
            return Document(doc_dict, text=None, comments=doc_comments, empty_sentences=doc_empty)

        # the offsets have to be added to the dicts before the Document is
        # built, since that is what the Tokens and Words are built from
        text = CoNLL.add_char_offsets(doc_dict, doc_comments)
        # the text is attached after building the document rather than passed
        # to the constructor.  passing it would run mark_whitespace, which
        # rewrites the MISC column of every token with a space annotation
        doc = Document(doc_dict, text=None, comments=doc_comments, empty_sentences=doc_empty)
        doc.text = text
        for sentence in doc.sentences:
            if sentence.text is None and len(sentence.tokens) > 0:
                sentence.text = text[sentence.tokens[0].start_char:sentence.tokens[-1].end_char]
        doc.build_ents()
        return doc

    @staticmethod
    def conll2multi_docs(input_file=None, input_str=None, ignore_gapping=True, zip_file=None):
        doc_dict, doc_comments, doc_empty = CoNLL.conll2dict(input_file, input_str, ignore_gapping, zip_file=zip_file)

        docs = []
        current_doc = []
        current_comments = []
        current_empty = []
        current_doc_id = None
        for doc, comments, empty in zip(doc_dict, doc_comments, doc_empty):
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
                    current_doc = [doc]
                    current_comments = [comments]
                    current_empty = [empty]
                    break
            else: # no comments defined a new doc_id, so just add it to the current document
                current_doc.append(doc)
                current_comments.append(comments)
                current_empty.append(empty)
        if len(current_doc) > 0:
            new_doc = Document(current_doc, text=None, comments=current_comments, empty_sentences=current_empty)
            if current_doc_id != None:
                for i in new_doc.sentences:
                    i.doc_id = current_doc_id.strip()
            docs.append(new_doc)
            current_doc_id = doc_id

        return docs

    @staticmethod
    def dict2conll(doc_dict, filename):
        """
        Convert the dictionary format input data to the CoNLL-U format output data and write to a file.
        """
        doc = Document(doc_dict)
        CoNLL.write_doc2conll(doc, filename)


    @staticmethod
    def write_doc2conll(doc, filename, mode='w', encoding='utf-8'):
        """
        Writes the doc as a conll file to the given file.

        If passed a string, that filename will be opened.  Otherwise, filename.write() will be called.

        Note that the output needs an extra \n\n at the end to be a legal output file
        """
        if hasattr(filename, "write"):
            filename.write("{:C}\n\n".format(doc))
        else:
            with open(filename, mode, encoding=encoding) as outfile:
                outfile.write("{:C}\n\n".format(doc))
