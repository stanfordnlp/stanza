"""
Utility functions for the loading and conversion of CoNLL-format files.
"""
import os
import io
from zipfile import ZipFile

from stanza.models.common.doc import Document
from stanza.models.common.doc import ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, NER, START_CHAR, END_CHAR
from stanza.models.common.doc import FIELD_TO_IDX, FIELD_NUM

class CoNLLError(ValueError):
    pass

class CoNLL:

    @staticmethod
    def load_conll(f, ignore_gapping=True):
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
                    raise CoNLLError("Could not process sentence %d token %d: %s" % (sent_idx, token_idx, str(e))) from e
                if '.' in token_dict[ID]:
                    token_dict[ID] = tuple(int(x) for x in token_dict[ID].split(".", maxsplit=1))
                    sent_empty.append(token_dict)
                else:
                    token_dict[ID] = tuple(int(x) for x in token_dict[ID].split("-", maxsplit=1))
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
        for field in FIELD_TO_IDX:
            value = token_conll[FIELD_TO_IDX[field]]
            if value != '_':
                if field == HEAD:
                    token_dict[field] = int(value)
                else:
                    token_dict[field] = value
            # special case if text is '_'
            if token_conll[FIELD_TO_IDX[TEXT]] == '_':
                token_dict[TEXT] = token_conll[FIELD_TO_IDX[TEXT]]
                token_dict[LEMMA] = token_conll[FIELD_TO_IDX[LEMMA]]
        return token_dict

    @staticmethod
    def conll2dict(input_file=None, input_str=None, ignore_gapping=True, zip_file=None):
        """ Load the CoNLL-U format data from file or string into lists of dictionaries.
        """
        assert any([input_file, input_str]) and not all([input_file, input_str]), 'either use input file or input string'
        if zip_file: assert input_file, 'must provide input_file if zip_file is set'

        if input_str:
            infile = io.StringIO(input_str)
            doc_conll, doc_comments = CoNLL.load_conll(infile, ignore_gapping)
        elif zip_file:
            with ZipFile(zip_file) as zin:
                with zin.open(input_file) as fin:
                    doc_conll, doc_comments = CoNLL.load_conll(io.TextIOWrapper(fin, encoding="utf-8"), ignore_gapping)
        else:
            with open(input_file, encoding='utf-8') as fin:
                doc_conll, doc_comments = CoNLL.load_conll(fin, ignore_gapping)

        doc_dict, doc_empty = CoNLL.convert_conll(doc_conll)
        return doc_dict, doc_comments, doc_empty

    @staticmethod
    def conll2doc(input_file=None, input_str=None, ignore_gapping=True, zip_file=None):
        doc_dict, doc_comments, doc_empty = CoNLL.conll2dict(input_file, input_str, ignore_gapping, zip_file=zip_file)
        return Document(doc_dict, text=None, comments=doc_comments, empty_sentences=doc_empty)

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
