"""
Tests that call a running CoreNLPClient.
"""
import pytest
import stanfordnlp.server as corenlp

# set the marker for this module
pytestmark = pytest.mark.travis

TEXT = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP.\n"


def test_connect():
    with corenlp.CoreNLPClient() as client:
        client.ensure_alive()
        assert client.is_active
        assert client.is_alive()


def test_annotate():
    with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
        ann = client.annotate(TEXT)
        assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


def test_update():
    with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
        ann = client.annotate(TEXT)
        ann = client.update(ann)
        assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


def test_tokensregex():
    with corenlp.CoreNLPClient(annotators='tokenize ssplit ner depparse'.split(), timeout=60000) as client:
        # Example pattern from: https://nlp.stanford.edu/software/tokensregex.shtml
        pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
        matches = client.tokensregex(TEXT, pattern)
        assert len(matches["sentences"]) == 1
        assert matches["sentences"][0]["length"] == 1
        assert matches == {
            "sentences": [{
                "0": {
                    "text": "Chris wrote a simple sentence",
                    "begin": 0,
                    "end": 5,
                    "1": {
                        "text": "Chris",
                        "begin": 0,
                        "end": 1
                        }},
                "length": 1
                },]}


def test_semgrex():
    with corenlp.CoreNLPClient(annotators='tokenize ssplit pos lemma ner depparse'.split(), timeout=60000) as client:
        pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
        matches = client.semgrex(TEXT, pattern, to_words=True)
        assert matches == [
                {
                    "text": "wrote",
                    "begin": 1,
                    "end": 2,
                    "$subject": {
                        "text": "Chris",
                        "begin": 0,
                        "end": 1
                        },
                    "$object": {
                        "text": "sentence",
                        "begin": 4,
                        "end": 5
                        },
                    "sentence": 0,}]
