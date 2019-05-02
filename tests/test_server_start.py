"""
Tests for starting a server in Python code
"""

import stanfordnlp.server as corenlp
import time

EN_DOC = "Joe Smith lives in California."

EN_DOC_DEFAULT_GOLD = """
Sentence #1 (6 tokens):
Joe Smith lives in California.

Tokens:
[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP Lemma=Joe NamedEntityTag=PERSON]
[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP Lemma=Smith NamedEntityTag=PERSON]
[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ Lemma=live NamedEntityTag=O]
[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN Lemma=in NamedEntityTag=O]
[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP Lemma=California NamedEntityTag=STATE_OR_PROVINCE]
[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=. Lemma=. NamedEntityTag=O]

Dependency Parse (enhanced plus plus dependencies):
root(ROOT-0, lives-3)
compound(Smith-2, Joe-1)
nsubj(lives-3, Smith-2)
case(California-5, in-4)
nmod:in(lives-3, California-5)
punct(lives-3, .-6)

Extracted the following NER entity mentions:
Joe Smith	PERSON
California	STATE_OR_PROVINCE
""".strip()


def annotate_and_time(client, text, properties={}):
    """ Submit an annotation request and return how long it took """
    start = time.time()
    ann = client.annotate(text, properties=properties, output_format="text")
    end = time.time()
    return {'annotation': ann, 'start_time': start, 'end_time': end}


def test_preload(server_id='test_server_start_preload'):
    """ Test that the default annotators load fully immediately upon server start """
    with corenlp.CoreNLPClient(server_id='test_server_start_preload') as client:
        # wait for annotators to load
        time.sleep(140)
        results = annotate_and_time(client, EN_DOC)
        assert results['annotation'].strip() == EN_DOC_DEFAULT_GOLD
        assert results['end_time'] - results['start_time'] < 1.5
