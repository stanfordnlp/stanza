"""
Tests for starting a server in Python code
"""

import pytest
import stanza.server as corenlp
from stanza.server.client import AnnotationException
import time

from stanza.tests import *

pytestmark = pytest.mark.client

EN_DOC = "Joe Smith lives in California."

# results on EN_DOC with standard StanfordCoreNLP defaults
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
obl:in(lives-3, California-5)
punct(lives-3, .-6)

Extracted the following NER entity mentions:
Joe Smith       PERSON              PERSON:0.9972202681743931
California      STATE_OR_PROVINCE   LOCATION:0.9990868267559281

Extracted the following KBP triples:
1.0 Joe Smith per:statesorprovinces_of_residence California
"""

# results with an example properties file
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
"""

GERMAN_DOC = "Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland."

# results with standard German properties
GERMAN_FULL_PROPS_GOLD = """
Sentence #1 (10 tokens):
Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.

Tokens:
[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN Lemma=angela NamedEntityTag=PERSON]
[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN Lemma=merkel NamedEntityTag=PERSON]
[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=AUX Lemma=ist NamedEntityTag=O]
[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=ADP Lemma=seit NamedEntityTag=O]
[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=NUM Lemma=2005 NamedEntityTag=O]
[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NOUN Lemma=bundeskanzlerin NamedEntityTag=O]
[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=DET Lemma=der NamedEntityTag=O]
[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=PROPN Lemma=bundesrepublik NamedEntityTag=LOCATION]
[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=PROPN Lemma=deutschland NamedEntityTag=LOCATION]
[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=PUNCT Lemma=. NamedEntityTag=O]

Dependency Parse (enhanced plus plus dependencies):
root(ROOT-0, Bundeskanzlerin-6)
nsubj(Bundeskanzlerin-6, Angela-1)
flat(Angela-1, Merkel-2)
cop(Bundeskanzlerin-6, ist-3)
case(2005-5, seit-4)
nmod:seit(Bundeskanzlerin-6, 2005-5)
det(Bundesrepublik-8, der-7)
nmod(Bundeskanzlerin-6, Bundesrepublik-8)
appos(Bundesrepublik-8, Deutschland-9)
punct(Bundeskanzlerin-6, .-10)

Extracted the following NER entity mentions:
Angela Merkel              PERSON   PERSON:0.9999981583351504
Bundesrepublik Deutschland LOCATION LOCATION:0.9682902289749544
"""


GERMAN_SMALL_PROPS = {'annotators': 'tokenize,ssplit,pos', 'tokenize.language': 'de',
                      'pos.model': 'edu/stanford/nlp/models/pos-tagger/german-ud.tagger'}

# results with custom Python dictionary set properties
GERMAN_SMALL_PROPS_GOLD = """
Sentence #1 (10 tokens):
Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.

Tokens:
[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN]
[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN]
[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17 PartOfSpeech=AUX]
[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22 PartOfSpeech=ADP]
[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27 PartOfSpeech=NUM]
[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43 PartOfSpeech=NOUN]
[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47 PartOfSpeech=DET]
[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62 PartOfSpeech=PROPN]
[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74 PartOfSpeech=PROPN]
[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75 PartOfSpeech=PUNCT]
"""

# results with custom Python dictionary set properties and annotators=tokenize,ssplit
GERMAN_SMALL_PROPS_W_ANNOTATORS_GOLD = """
Sentence #1 (10 tokens):
Angela Merkel ist seit 2005 Bundeskanzlerin der Bundesrepublik Deutschland.

Tokens:
[Text=Angela CharacterOffsetBegin=0 CharacterOffsetEnd=6]
[Text=Merkel CharacterOffsetBegin=7 CharacterOffsetEnd=13]
[Text=ist CharacterOffsetBegin=14 CharacterOffsetEnd=17]
[Text=seit CharacterOffsetBegin=18 CharacterOffsetEnd=22]
[Text=2005 CharacterOffsetBegin=23 CharacterOffsetEnd=27]
[Text=Bundeskanzlerin CharacterOffsetBegin=28 CharacterOffsetEnd=43]
[Text=der CharacterOffsetBegin=44 CharacterOffsetEnd=47]
[Text=Bundesrepublik CharacterOffsetBegin=48 CharacterOffsetEnd=62]
[Text=Deutschland CharacterOffsetBegin=63 CharacterOffsetEnd=74]
[Text=. CharacterOffsetBegin=74 CharacterOffsetEnd=75]
"""

# properties for username/password example
USERNAME_PASS_PROPS = {'annotators': 'tokenize,ssplit,pos'}

USERNAME_PASS_GOLD = """
Sentence #1 (6 tokens):
Joe Smith lives in California.

Tokens:
[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=NNP]
[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=NNP]
[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=VBZ]
[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=IN]
[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=NNP]
[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=.]
"""


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
        compare_ignoring_whitespace(results['annotation'], EN_PRELOAD_GOLD)
        assert results['end_time'] - results['start_time'] < 3


def test_props_file():
    """ Test starting the server with a props file """
    with corenlp.CoreNLPClient(properties=SERVER_TEST_PROPS, server_id='test_server_start_props_file') as client:
        ann = client.annotate(EN_DOC, output_format="text")
        assert ann.strip() == EN_PROPS_FILE_GOLD.strip()


def test_lang_start():
    """ Test starting the server with a Stanford CoreNLP language name """
    with corenlp.CoreNLPClient(properties='german', server_id='test_server_start_lang_name') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        compare_ignoring_whitespace(ann, GERMAN_FULL_PROPS_GOLD)


def test_python_dict():
    """ Test starting the server with a Python dictionary as default properties """
    with corenlp.CoreNLPClient(properties=GERMAN_SMALL_PROPS, server_id='test_server_start_python_dict') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        assert ann.strip() == GERMAN_SMALL_PROPS_GOLD.strip()


def test_python_dict_w_annotators():
    """ Test starting the server with a Python dictionary as default properties, override annotators """
    with corenlp.CoreNLPClient(properties=GERMAN_SMALL_PROPS, annotators="tokenize,ssplit",
                               server_id='test_server_start_python_dict_w_annotators') as client:
        ann = client.annotate(GERMAN_DOC, output_format='text')
        assert ann.strip() == GERMAN_SMALL_PROPS_W_ANNOTATORS_GOLD.strip()


def test_username_password():
    """ Test starting a server with a username and password """
    with corenlp.CoreNLPClient(properties=USERNAME_PASS_PROPS, username='user-1234', password='1234',
                               server_id="test_server_username_pass") as client:
        # check with correct password
        ann = client.annotate(EN_DOC, output_format='text', username='user-1234', password='1234')
        assert ann.strip() == USERNAME_PASS_GOLD.strip()
        # check with incorrect password, should throw AnnotationException
        try:
            ann = client.annotate(EN_DOC, output_format='text', username='user-1234', password='12345')
            assert False
        except AnnotationException as ae:
            pass
        except Exception as e:
            assert False


