"""
Tests for starting a server in Python code
"""

import stanfordnlp.server as corenlp
import time

from tests import *

EN_DOC = "Joe Smith lives in California."

EN_PRELOAD_GOLD = """
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

EN_PROPS_FILE_GOLD = """
Sentence #1 (6 tokens):
Joe Smith lives in California.

Tokens:
[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP]
[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP]
[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ]
[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN]
[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP]
[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=.]
""".strip()

GERMAN_DOC = "Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland."

GERMAN_FULL_PROPS_GOLD = """
Sentence #1 (10 tokens):
Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.

Tokens:
[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=NE Lemma=angela NamedEntityTag=PERSON]
[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=NE Lemma=merkel NamedEntityTag=PERSON]
[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=VAFIN Lemma=ist NamedEntityTag=O]
[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=APPR Lemma=seit NamedEntityTag=O]
[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=CARD Lemma=2005 NamedEntityTag=O]
[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NN Lemma=bundeskanzlerin NamedEntityTag=O]
[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=ART Lemma=der NamedEntityTag=O]
[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=NN Lemma=bundesrepublik NamedEntityTag=LOCATION]
[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=NE Lemma=deutschland NamedEntityTag=LOCATION]
[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=$. Lemma=. NamedEntityTag=O]

Constituency parse: 
(ROOT
  (S
    (MPN (NE Angela) (NE Merkel))
    (VAFIN ist)
    (PP (APPR seit) (CARD 2005) (NN Bundeskanzlerin)
      (NP (ART der) (NN Bundesrepublik) (NE Deutschland)))
    ($. .)))


Extracted the following NER entity mentions:
Angela Merkel	PERSON
Bundesrepublik Deutschland	LOCATION
""".strip()


GERMAN_SMALL_PROPS = {'annotators': 'tokenize,ssplit,pos', 'tokenize.language': 'de',
                      'pos.model': 'edu/stanford/nlp/models/pos-tagger/german/german-hgc.tagger'}

GERMAN_SMALL_PROPS_GOLD = """
Sentence #1 (10 tokens):
Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.

Tokens:
[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=NE]
[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=NE]
[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=VAFIN]
[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=APPR]
[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=CARD]
[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NN]
[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=ART]
[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=NN]
[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=NE]
[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=$.]
""".strip()


def annotate_and_time(client, text, properties={}):
    """ Submit an annotation request and return how long it took """
    start = time.time()
    ann = client.annotate(text, properties=properties, output_format="text")
    end = time.time()
    return {'annotation': ann, 'start_time': start, 'end_time': end}


def test_preload():
    """ Test that the default annotators load fully immediately upon server start """
    with corenlp.CoreNLPClient(server_id='test_server_start_preload') as client:
        # wait for annotators to load
        time.sleep(140)
        results = annotate_and_time(client, EN_DOC)
        assert results['annotation'].strip() == EN_PRELOAD_GOLD
        assert results['end_time'] - results['start_time'] < 1.5


def test_props_file():
    """ Test starting the server with a props file """
    with corenlp.CoreNLPClient(properties=SERVER_TEST_PROPS, server_id='test_server_start_props_file') as client:
        ann = client.annotate(EN_DOC, output_format="text")
        assert ann.strip() == EN_PROPS_FILE_GOLD


def test_lang_start():
    """ Test starting the server with a Stanford CoreNLP language name """
    with corenlp.CoreNLPClient(properties='german', server_id='test_server_start_lang_name') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        assert ann.strip() == GERMAN_FULL_PROPS_GOLD


def test_python_dict():
    """ Test starting the server with a Python dictionary as default properties """
    with corenlp.CoreNLPClient(properties=GERMAN_SMALL_PROPS, server_id='test_server_start_python_dict') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        assert ann.strip() == GERMAN_SMALL_PROPS_GOLD


