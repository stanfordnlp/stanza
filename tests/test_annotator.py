"""
A test annotator (tokens).
"""

from pytest import fixture

import corenlp
from .happyfuntokenizer import Tokenizer

class HappyFunTokenizer(Tokenizer, corenlp.Annotator):
    def __init__(self, preserve_case=False):
        Tokenizer.__init__(self, preserve_case)
        corenlp.Annotator.__init__(self)

    @property
    def name(self):
        """
        Name of the annotator (used by CoreNLP)
        """
        return "happyfun"

    @property
    def requires(self):
        """
        Requires has to specify all the annotations required before we
        are called.
        """
        return []

    @property
    def provides(self):
        """
        The set of annotations guaranteed to be provided when we are done.
        NOTE: that these annotations are either fully qualified Java
        class names or refer to nested classes of
        edu.stanford.nlp.ling.CoreAnnotations (as is the case below).
        """
        return ["TextAnnotation",
                "TokensAnnotation",
                "TokenBeginAnnotation",
                "TokenEndAnnotation"]

    def annotate(self, ann):
        """
        @ann: is a protobuf annotation object.
        Actually populate @ann with tokens.
        """
        for i, word in enumerate(self.tokenize(ann.text)):
            token = ann.sentencelessToken.add()
            # These are the bare minimum required for the TokenAnnotation
            token.word = word
            token.tokenOffsetBegin = i
            token.tokenOffsetEnd = i+1

def test_tokenizer():
    annotator = HappyFunTokenizer()
    annotator.start()

    client = corenlp.CoreNLPClient(properties=annotator.properties, annotators="happyfun pos".split())
    ann = client.annotate("RT @ #happyfuncoding: this is a typical Twitter tweet :-)")

    tokens = [t.word for t in ann.sentencelessToken]
    assert tokens == "RT @ #happyfuncoding : this is a typical Twitter tweet :-)".split()

    annotator.join()
