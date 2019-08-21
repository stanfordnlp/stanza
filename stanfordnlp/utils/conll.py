"""
A wrapper/loader for the official conll-u format files.
"""
import os
import io

FIELD_NUM = 10

FIELD_TO_IDX = {'id': 0, 'text': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}

class CoNLL:

    @staticmethod
    def load_conll(f, ignore_gapping=True):
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
                assert len(array) == FIELD_NUM
                sent += [array]
        if len(sent) > 0:
            doc.append(sent)
        return doc

    @staticmethod
    def convert_conll(doc_conll):
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
        token_dict = {}
        for field in FIELD_TO_IDX:
            value = token_conll[FIELD_TO_IDX[field]]
            if value != '_':
                if field == 'head':
                    token_dict[field] = int(value)
                else:
                    token_dict[field] = value
            # <COMMENT>: A very special case, like `_	_	SYM	NFP	_	1	punct	1:punct	_`
            if token_conll[FIELD_TO_IDX['text']] == '_':
                token_dict['text'] = token_conll[FIELD_TO_IDX['text']]
                token_dict['lemma'] = token_conll[FIELD_TO_IDX['lemma']]
        return token_dict
        
    @staticmethod
    def conll2dict(input_file=None, input_str=None, ignore_gapping=True):
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
        token_conll = ['_' for i in range(FIELD_NUM)]
        for key in token_dict:
            token_conll[FIELD_TO_IDX[key]] = str(token_dict[key])
        return token_conll

    @staticmethod
    def conll_as_string(doc):
        """ Return current conll contents as string
        """
        return_string = ""
        for sent in doc:
            for ln in sent:
                return_string += ("\t".join(ln)+"\n")
            return_string += "\n"
        return return_string
    
    @staticmethod
    def dict2conll(doc_dict, filename):
        doc_conll = CoNLL.convert_dict(doc_dict)
        conll_string = CoNLL.conll_as_string(doc_conll)
        with open(filename, 'w') as outfile:
            outfile.write(conll_string)
        return
        
    # def get_mwt_expansions(self):
    #     word_idx = FIELD_TO_IDX['word']
    #     expansions = []
    #     src = ''
    #     dst = []
    #     for sent in self.sents:
    #         mwt_begin = 0
    #         mwt_end = -1
    #         for ln in sent:
    #             if '.' in ln[0]:
    #                 # skip ellipsis
    #                 continue

    #             if '-' in ln[0]:
    #                 mwt_begin, mwt_end = [int(x) for x in ln[0].split('-')]
    #                 src = ln[word_idx]
    #                 continue

    #             if mwt_begin <= int(ln[0]) < mwt_end:
    #                 dst += [ln[word_idx]]
    #             elif int(ln[0]) == mwt_end:
    #                 dst += [ln[word_idx]]
    #                 expansions += [[src, ' '.join(dst)]]
    #                 src = ''
    #                 dst = []

    #     return expansions

    # def get_mwt_expansion_cands(self):
    #     word_idx = FIELD_TO_IDX['word']
    #     cands = []
    #     for sent in self.sents:
    #         for ln in sent:
    #             if "MWT=Yes" in ln[-1]:
    #                 cands += [ln[word_idx]]

    #     return cands

    # def write_conll_with_mwt_expansions(self, expansions, output_file):
    #     """ Expands MWTs predicted by the tokenizer and write to file. This method replaces the head column with a right branching tree. """
    #     idx = 0
    #     count = 0

    #     for sent in self.sents:
    #         for ln in sent:
    #             idx += 1
    #             if "MWT=Yes" not in ln[-1]:
    #                 print("{}\t{}".format(idx, "\t".join(ln[1:6] + [str(idx-1)] + ln[7:])), file=output_file)
    #             else:
    #                 # print MWT expansion
    #                 expanded = [x for x in expansions[count].split(' ') if len(x) > 0]
    #                 count += 1
    #                 endidx = idx + len(expanded) - 1

    #                 ln[-1] = '_' if ln[-1] == 'MWT=Yes' else '|'.join([x for x in ln[-1].split('|') if x != 'MWT=Yes'])
    #                 print("{}-{}\t{}".format(idx, endidx, "\t".join(['_' if i == 5 else x for i, x in enumerate(ln[1:])])), file=output_file)
    #                 for e_i, e_word in enumerate(expanded):
    #                     print("{}\t{}\t{}".format(idx + e_i, e_word, "\t".join(['_'] * 4 + [str(idx + e_i - 1)] + ['_'] * 3)), file=output_file)
    #                 idx = endidx

    #         print("", file=output_file)
    #         idx = 0

    #     assert count == len(expansions), "{} {} {}".format(count, len(expansions), expansions)
    #     return
