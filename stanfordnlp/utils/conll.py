"""
Utility functions for the loading and conversion of CoNLL-format files.
"""
import os
import io

FIELD_NUM = 10

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
FIELD_TO_IDX = {ID: 0, TEXT: 1, LEMMA: 2, UPOS: 3, XPOS: 4, FEATS: 5, HEAD: 6, DEPREL: 7, DEPS: 8, MISC: 9}

class CoNLL:

    @staticmethod
    def load_conll(f, ignore_gapping=True):
        """ Load the file or string into the CoNLL-U format data.
        Input: file or string reader, where the data is in CoNLL-U format.
        Output: a list of list of list for each token in each sentence in the data, where the innermost list represents 
        all fields of a token.
        """
        # f is open() or io.StringIO()
        doc, sent = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(sent) > 0:
                    doc.append(sent)
                    sent = []
            else:
                if line.startswith('#'): # skip comment line
                    continue
                array = line.split('\t')
                if ignore_gapping and '.' in array[0]:
                    continue
                assert len(array) == FIELD_NUM, \
                        f"Cannot parse CoNLL line: expecting {FIELD_NUM} fields, {len(array)} found."
                sent += [array]
        if len(sent) > 0:
            doc.append(sent)
        return doc

    @staticmethod
    def convert_conll(doc_conll):
        """ Convert the CoNLL-U format input data to a dictionary format output data.
        Input: list of token fields loaded from the CoNLL-U format data, where the outmost list represents a list of sentences, and the inside list represents all fields of a token.
        Output: a list of list of dictionaries for each token in each sentence in the document.
        """
        doc_dict = []
        for sent_conll in doc_conll:
            sent_dict = []
            for token_conll in sent_conll:
                token_dict = CoNLL.convert_conll_token(token_conll)
                sent_dict.append(token_dict)
            doc_dict.append(sent_dict)
        return doc_dict
    
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
    def conll2dict(input_file=None, input_str=None, ignore_gapping=True):
        """ Load the CoNLL-U format data from file or string into lists of dictionaries.
        """
        assert any([input_file, input_str]) and not all([input_file, input_str]), 'either input input file or input string'
        if input_str:
            infile = io.StringIO(input_str)
        else:
            infile = open(input_file)
        doc_conll = CoNLL.load_conll(infile, ignore_gapping)
        doc_dict = CoNLL.convert_conll(doc_conll)
        return doc_dict

    @staticmethod
    def convert_dict(doc_dict):
        """ Convert the dictionary format input data to the CoNLL-U format output data. This is the reverse function of 
        `convert_conll`.
        Input: dictionary format data, which is a list of list of dictionaries for each token in each sentence in the data.
        Output: CoNLL-U format data, which is a list of list of list for each token in each sentence in the data.
        """
        doc_conll = []
        for sent_dict in doc_dict:
            sent_conll = []
            for token_dict in sent_dict:
                token_conll = CoNLL.convert_token_dict(token_dict)
                sent_conll.append(token_conll)
            doc_conll.append(sent_conll)
        return doc_conll
            
    @staticmethod
    def convert_token_dict(token_dict):
        """ Convert the dictionary format input token to the CoNLL-U format output token. This is the reverse function of 
        `convert_conll_token`.
        Input: dictionary format token, which is a dictionaries for the token.
        Output: CoNLL-U format token, which is a list for the token.
        """
        token_conll = ['_' for i in range(FIELD_NUM)]
        for key in token_dict:
            if key in FIELD_TO_IDX:
                token_conll[FIELD_TO_IDX[key]] = str(token_dict[key])
        # when a word (not mwt token) without head is found, we insert dummy head as required by the UD eval script
        if '-' not in token_dict[ID] and HEAD not in token_dict:
            token_conll[FIELD_TO_IDX[HEAD]] = str(int(token_dict[ID]) - 1) # evaluation script requires head: int
        return token_conll

    @staticmethod
    def conll_as_string(doc):
        """ Dump the loaded CoNLL-U format list data to string. """
        return_string = ""
        for sent in doc:
            for ln in sent:
                return_string += ("\t".join(ln)+"\n")
            return_string += "\n"
        return return_string
    
    @staticmethod
    def dict2conll(doc_dict, filename):
        """ Convert the dictionary format input data to the CoNLL-U format output data and write to a file.
        """
        doc_conll = CoNLL.convert_dict(doc_dict)
        conll_string = CoNLL.conll_as_string(doc_conll)
        with open(filename, 'w') as outfile:
            outfile.write(conll_string)
        return
