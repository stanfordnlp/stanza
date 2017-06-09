"""
Tests that call a running CoreNLPClient.
"""
import corenlp

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
