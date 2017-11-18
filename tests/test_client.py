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

def test_tokensregex():
    with corenlp.CoreNLPClient(annotators='tokenize ssplit ner depparse'.split()) as client:
        # Example pattern from: https://nlp.stanford.edu/software/tokensregex.shtml
        text = 'Hello. Bob Ross was a famous painter. Goodbye.'
        pattern = '([ner: PERSON]+) /was|is/ /an?/ []{0,3} /painter|artist/'
        matches = client.tokensregex(text, pattern)
        assert matches == {
            "sentences": [{
                "length": 0
                },{
                    "0": {
                        "text": "Ross was a famous painter",
                        "begin": 1,
                        "end": 6,
                        "1": {
                            "text": "Ross",
                            "begin": 1,
                            "end": 2
                            }},
                    "length": 1
                },{
                    "length": 0
                }]}

def test_semgrex():
    with corenlp.CoreNLPClient(annotators='tokenize ssplit depparse'.split()) as client:
        text = 'I ran.'
        pattern = '{} < {}'
        matches = client.semgrex(text, pattern, to_words=True)
        assert matches == [{
            "text": ".",
            "begin": 2,
            "end": 3,
            "sentence": 0
        },{
            "text": "I",
            "begin": 0,
            "end": 1,
            "sentence": 0
            }]
