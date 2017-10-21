"""
Module containing annotators for performing tokenization.
Tokenization is performed with the lexer PLY.

"""

from ply import lex
from stanfordnlp.annotator import Annotator
from stanfordnlp.data_structures import Token
from stanfordnlp.annotations import TOKENS

# set up the lexer
tokens = (
    "TOKEN",
    "WHITESPACE"
)

t_TOKEN = (r"[^ \n]+")
t_WHITESPACE = (r"[ \n]+")

lex.lex()


class TokenizerAnnotator(Annotator):
    """Class which tokenizes text"""

    def __init__(self, settings={}):
        self._token_delimiter = " "

    @property
    def token_delimiter(self):
        """String which denotes separation between tokens"""
        return self._token_delimiter

    @token_delimiter.setter
    def token_delimiter(self, value):
        """Set string which denotes separation between tokens"""
        self._token_delimiter = value

    def annotate(self, doc):
        """Tokenize the document"""
        # submit text to lexer
        lex.input(doc.text)
        # iterate through tokens
        doc_tokens = []
        num_tokens_seen = 0
        prev_token = None
        for found_token in iter(lex.token, None):
            if found_token.type == "WHITESPACE":
                pass
            else:
                # build new token if not whitespace
                new_token = Token((found_token.lexpos, found_token.lexpos+len(found_token.value)),
                                  doc, num_tokens_seen, found_token.value)
                # check if preceding character was a "\n", mark this token as starting a line
                if prev_token and prev_token.value[-1] == "\n":
                    new_token.starts_line = True
                # add to list of tokens
                doc_tokens.append(new_token)
            prev_token = found_token
        doc.tokens = doc_tokens

    def requires(self):
        # this annotator has no requirements, so return empty set
        return set([])

    def requirements_satisfied(self):
        # this annotator produces tokens
        return set([TOKENS])
