"""
Tests that call a running CoreNLPClient.
"""
import pytest
import stanfordnlp.server as corenlp


# set the marker for this module
pytestmark = pytest.mark.travis

TEXT = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP.\n"

MAX_REQUEST_ATTEMPTS = 5


@pytest.fixture(scope="module")
def corenlp_client():
    """ Client to run tests on """
    client = corenlp.CoreNLPClient(annotators='tokenize,ssplit,pos,lemma,ner,depparse',
                                   server_id='stanfordnlp_main_test_server')
    yield client
    client.stop()


def test_connect(corenlp_client):
    corenlp_client.ensure_alive()
    assert corenlp_client.is_active
    assert corenlp_client.is_alive()


def test_context_manager():
    with corenlp.CoreNLPClient(annotators="tokenize,ssplit") as context_client:
        ann = context_client.annotate(TEXT)
        assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


def test_annotate(corenlp_client):
    ann = corenlp_client.annotate(TEXT)
    assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


def test_update(corenlp_client):
    ann = corenlp_client.annotate(TEXT)
    ann = corenlp_client.update(ann)
    assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


def test_tokensregex(corenlp_client):
    pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
    matches = corenlp_client.tokensregex(TEXT, pattern)
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


def test_semgrex(corenlp_client):
    pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
    matches = corenlp_client.semgrex(TEXT, pattern, to_words=True)
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

