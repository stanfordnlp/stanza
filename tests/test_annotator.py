"""
A test annotator (tokens).
"""
import requests
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
                "TokenEndAnnotation",
                "CharacterOffsetBeginAnnotation",
                "CharacterOffsetEndAnnotation",
               ]

    def annotate(self, ann):
        """
        @ann: is a protobuf annotation object.
        Actually populate @ann with tokens.
        """
        buf, beg_idx, end_idx = ann.text.lower(), 0, 0
        for i, word in enumerate(self.tokenize(ann.text)):
            token = ann.sentencelessToken.add()
            # These are the bare minimum required for the TokenAnnotation
            token.word = word
            token.tokenBeginIndex = i
            token.tokenEndIndex = i+1

            # Seek into the txt until you can find this word.
            try:
                # Try to update beginning index
                beg_idx = buf.index(word, beg_idx)
            except ValueError:
                # Give up -- this will be something random
                end_idx = beg_idx + len(word)

            token.beginChar = beg_idx
            token.endChar = end_idx

            beg_idx, end_idx = end_idx, end_idx

def test_annotator_annotate():
    cases = [("RT @ #happyfuncoding: this is a typical Twitter tweet :-)",
              "rt @ #happyfuncoding : this is a typical twitter tweet :-)".split()),
             ("HTML entities &amp; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(",
              "html entities and other web oddities can be an ácute".split() + ["<em class='grumpy'>", "pain", "</em>", ">:("]),
             ("It's perhaps noteworthy that phone numbers like +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace.",
              "it's perhaps noteworthy that phone numbers like".split() + ["+1 (800) 123-4567", ",", "(800) 123-4567", ",", "and", "123-4567"] + "are treated as words despite their whitespace .".split())
            ]

    annotator = HappyFunTokenizer()

    for text, tokens in cases:
        ann = corenlp.Document()
        ann.text = text
        annotator.annotate(ann)
        tokens_ = [t.word for t in ann.sentencelessToken]
        assert tokens_ == tokens

def test_annotator_alive():
    annotator = HappyFunTokenizer()
    annotator.start()

    # Ping the annotator.
    r = requests.get("http://localhost:8432/ping")
    assert r.ok
    assert r.content.decode("utf-8") == "pong"
    r = requests.get("http://localhost:8432/ping/")
    assert r.ok
    assert r.content.decode("utf-8") == "pong"

    annotator.terminate()
    annotator.join()

def test_tokenizer():
    cases = [("RT @ #happyfuncoding: this is a typical Twitter tweet :-)",
              "rt @ #happyfuncoding : this is a typical twitter tweet :-)".split()),
             ("HTML entities &amp; other Web oddities can be an &aacute;cute <em class='grumpy'>pain</em> >:(",
              "html entities and other web oddities can be an ácute".split() + ["<em class='grumpy'>", "pain", "</em>", ">:("]),
             ("It's perhaps noteworthy that phone numbers like +1 (800) 123-4567, (800) 123-4567, and 123-4567 are treated as words despite their whitespace.",
              "it's perhaps noteworthy that phone numbers like".split() + ["+1 (800) 123-4567", ",", "(800) 123-4567", ",", "and", "123-4567"] + "are treated as words despite their whitespace .".split())
            ]

    annotator = HappyFunTokenizer()
    annotator.start()

    with corenlp.CoreNLPClient(properties=annotator.properties, annotators="happyfun ssplit pos".split()) as client:
        for text, tokens in cases:
            ann = client.annotate(text)
            tokens_ = [t.word for t in ann.sentence[0].token]
            assert tokens == tokens_

    annotator.terminate()
    annotator.join()
