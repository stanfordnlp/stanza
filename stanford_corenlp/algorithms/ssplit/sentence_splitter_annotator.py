"""
Module containing annotators for performing tokenization
"""

from stanford_corenlp.annotator import Annotator
from stanford_corenlp.data_structures import Sentence
from stanford_corenlp.annotations import TOKENS, SENTENCES

class SentenceSplitterAnnotator(Annotator):
    """Class which splits a tokenized document into sentences"""

    def __init__(self, settings={}):
        pass

    def annotate(self, doc):
        # store list of sentences for doc
        doc_sentences = []
        # iterate through all tokens, building sentences
        curr_sentence_tokens = []
        sentence_num = 0
        for token in doc.tokens + ["DOC_END"]:
            # if this token starts a line, finish previous sentence, start a new one
            if token == "DOC_END" or token.starts_line:
                if len(curr_sentence_tokens) != 0:
                    new_sentence = Sentence(doc, sentence_num, curr_sentence_tokens)
                    doc_sentences.append(new_sentence)
                    curr_sentence_tokens = [token]
                    sentence_num += 1
            else:
                curr_sentence_tokens.append(token)
        doc.sentences = doc_sentences

    def requires(self):
        # this annotator requires tokens
        return set([TOKENS])

    def requirements_satisfied(self):
        # this annotator produces sentences
        return set([SENTENCES])
