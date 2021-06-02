"""
Tests for setting request properties of servers
"""

import json
import pytest
import stanza.server as corenlp

from stanza.protobuf import Document
from stanza.tests import TEST_WORKING_DIR, compare_ignoring_whitespace

pytestmark = pytest.mark.client

EN_DOC = "Joe Smith lives in California."

# results with an example properties file
EN_DOC_GOLD = """
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

GERMAN_DOC_GOLD = """
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

FRENCH_CUSTOM_PROPS = {'annotators': 'tokenize,ssplit,mwt,pos,parse',
                       'tokenize.language': 'fr',
                       'pos.model': 'edu/stanford/nlp/models/pos-tagger/french-ud.tagger',
                       'parse.model': 'edu/stanford/nlp/models/srparser/frenchSR.ser.gz',
                       'mwt.mappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt.tsv',
                       'mwt.pos.model': 'edu/stanford/nlp/models/mwt/french/french-mwt.tagger',
                       'mwt.statisticalMappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt-statistical.tsv',
                       'mwt.preserveCasing': 'false',
                       'outputFormat': 'text'}

FRENCH_EXTRA_PROPS = {'annotators': 'tokenize,ssplit,mwt,pos,depparse',
                      'tokenize.language': 'fr',
                      'pos.model': 'edu/stanford/nlp/models/pos-tagger/french-ud.tagger',
                      'mwt.mappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt.tsv',
                      'mwt.pos.model': 'edu/stanford/nlp/models/mwt/french/french-mwt.tagger',
                      'mwt.statisticalMappingFile': 'edu/stanford/nlp/models/mwt/french/french-mwt-statistical.tsv',
                      'mwt.preserveCasing': 'false',
                      'depparse.model': 'edu/stanford/nlp/models/parser/nndep/UD_French.gz'}

FRENCH_DOC = "Cette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt."

FRENCH_CUSTOM_GOLD = """
Sentence #1 (16 tokens):
Cette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt.

Tokens:
[Text=Cette CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=DET]
[Text=enquête CharacterOffsetBegin=6 CharacterOffsetEnd=13 PartOfSpeech=NOUN]
[Text=préliminaire CharacterOffsetBegin=14 CharacterOffsetEnd=26 PartOfSpeech=ADJ]
[Text=fait CharacterOffsetBegin=27 CharacterOffsetEnd=31 PartOfSpeech=VERB]
[Text=suite CharacterOffsetBegin=32 CharacterOffsetEnd=37 PartOfSpeech=NOUN]
[Text=à CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=ADP]
[Text=les CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=DET]
[Text=révélations CharacterOffsetBegin=42 CharacterOffsetEnd=53 PartOfSpeech=NOUN]
[Text=de CharacterOffsetBegin=54 CharacterOffsetEnd=56 PartOfSpeech=ADP]
[Text=l’ CharacterOffsetBegin=57 CharacterOffsetEnd=59 PartOfSpeech=NOUN]
[Text=hebdomadaire CharacterOffsetBegin=59 CharacterOffsetEnd=71 PartOfSpeech=ADJ]
[Text=quelques CharacterOffsetBegin=72 CharacterOffsetEnd=80 PartOfSpeech=DET]
[Text=jours CharacterOffsetBegin=81 CharacterOffsetEnd=86 PartOfSpeech=NOUN]
[Text=plus CharacterOffsetBegin=87 CharacterOffsetEnd=91 PartOfSpeech=ADV]
[Text=tôt CharacterOffsetBegin=92 CharacterOffsetEnd=95 PartOfSpeech=ADV]
[Text=. CharacterOffsetBegin=95 CharacterOffsetEnd=96 PartOfSpeech=PUNCT]

Constituency parse: 
(ROOT
  (SENT
    (NP (DET Cette)
      (MWN (NOUN enquête) (ADJ préliminaire)))
    (VN
      (MWV (VERB fait) (NOUN suite)))
    (PP (ADP à)
      (NP (DET les) (NOUN révélations)
        (PP (ADP de)
          (NP (NOUN l’)
            (AP (ADJ hebdomadaire))))))
    (NP (DET quelques) (NOUN jours))
    (AdP (ADV plus) (ADV tôt))
    (PUNCT .)))
"""

FRENCH_EXTRA_GOLD = """
Sentence #1 (16 tokens):
Cette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt.

Tokens:
[Text=Cette CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=DET]
[Text=enquête CharacterOffsetBegin=6 CharacterOffsetEnd=13 PartOfSpeech=NOUN]
[Text=préliminaire CharacterOffsetBegin=14 CharacterOffsetEnd=26 PartOfSpeech=ADJ]
[Text=fait CharacterOffsetBegin=27 CharacterOffsetEnd=31 PartOfSpeech=VERB]
[Text=suite CharacterOffsetBegin=32 CharacterOffsetEnd=37 PartOfSpeech=NOUN]
[Text=à CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=ADP]
[Text=les CharacterOffsetBegin=38 CharacterOffsetEnd=41 PartOfSpeech=DET]
[Text=révélations CharacterOffsetBegin=42 CharacterOffsetEnd=53 PartOfSpeech=NOUN]
[Text=de CharacterOffsetBegin=54 CharacterOffsetEnd=56 PartOfSpeech=ADP]
[Text=l’ CharacterOffsetBegin=57 CharacterOffsetEnd=59 PartOfSpeech=NOUN]
[Text=hebdomadaire CharacterOffsetBegin=59 CharacterOffsetEnd=71 PartOfSpeech=ADJ]
[Text=quelques CharacterOffsetBegin=72 CharacterOffsetEnd=80 PartOfSpeech=DET]
[Text=jours CharacterOffsetBegin=81 CharacterOffsetEnd=86 PartOfSpeech=NOUN]
[Text=plus CharacterOffsetBegin=87 CharacterOffsetEnd=91 PartOfSpeech=ADV]
[Text=tôt CharacterOffsetBegin=92 CharacterOffsetEnd=95 PartOfSpeech=ADV]
[Text=. CharacterOffsetBegin=95 CharacterOffsetEnd=96 PartOfSpeech=PUNCT]

Dependency Parse (enhanced plus plus dependencies):
root(ROOT-0, fait-4)
det(enquête-2, Cette-1)
nsubj(fait-4, enquête-2)
amod(enquête-2, préliminaire-3)
obj(fait-4, suite-5)
case(révélations-8, à-6)
det(révélations-8, les-7)
obl:à(fait-4, révélations-8)
case(l’-10, de-9)
nmod:de(révélations-8, l’-10)
amod(révélations-8, hebdomadaire-11)
det(jours-13, quelques-12)
obl(fait-4, jours-13)
advmod(tôt-15, plus-14)
advmod(jours-13, tôt-15)
punct(fait-4, .-16)
"""

FRENCH_JSON_GOLD = json.loads(open(f'{TEST_WORKING_DIR}/out/example_french.json').read())

ES_DOC = 'Andrés Manuel López Obrador es el presidente de México.'

ES_PROPS = {'annotators': 'tokenize,ssplit,mwt,pos,depparse', 'tokenize.language': 'es',
            'pos.model': 'edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger',
            'mwt.mappingFile': 'edu/stanford/nlp/models/mwt/spanish/spanish-mwt.tsv',
            'depparse.model': 'edu/stanford/nlp/models/parser/nndep/UD_Spanish.gz'}

ES_PROPS_GOLD = """
Sentence #1 (10 tokens):
Andrés Manuel López Obrador es el presidente de México.

Tokens:
[Text=Andrés CharacterOffsetBegin=0 CharacterOffsetEnd=6 PartOfSpeech=PROPN]
[Text=Manuel CharacterOffsetBegin=7 CharacterOffsetEnd=13 PartOfSpeech=PROPN]
[Text=López CharacterOffsetBegin=14 CharacterOffsetEnd=19 PartOfSpeech=PROPN]
[Text=Obrador CharacterOffsetBegin=20 CharacterOffsetEnd=27 PartOfSpeech=PROPN]
[Text=es CharacterOffsetBegin=28 CharacterOffsetEnd=30 PartOfSpeech=AUX]
[Text=el CharacterOffsetBegin=31 CharacterOffsetEnd=33 PartOfSpeech=DET]
[Text=presidente CharacterOffsetBegin=34 CharacterOffsetEnd=44 PartOfSpeech=NOUN]
[Text=de CharacterOffsetBegin=45 CharacterOffsetEnd=47 PartOfSpeech=ADP]
[Text=México CharacterOffsetBegin=48 CharacterOffsetEnd=54 PartOfSpeech=PROPN]
[Text=. CharacterOffsetBegin=54 CharacterOffsetEnd=55 PartOfSpeech=PUNCT]

Dependency Parse (enhanced plus plus dependencies):
root(ROOT-0, presidente-7)
nsubj(presidente-7, Andrés-1)
flat(Andrés-1, Manuel-2)
flat(Andrés-1, López-3)
flat(Andrés-1, Obrador-4)
cop(presidente-7, es-5)
det(presidente-7, el-6)
case(México-9, de-8)
nmod:de(presidente-7, México-9)
punct(presidente-7, .-10)
"""


@pytest.fixture(scope="module")
def corenlp_client():
    """ Client to run tests on """
    client = corenlp.CoreNLPClient(annotators='tokenize,ssplit,pos', server_id='stanza_request_tests_server')
    yield client
    client.stop()


def test_basic(corenlp_client):
    """ Basic test of making a request, test default output format is a Document """
    ann = corenlp_client.annotate(EN_DOC, output_format="text")
    assert ann.strip() == EN_DOC_GOLD.strip()
    ann = corenlp_client.annotate(EN_DOC)
    assert isinstance(ann, Document)


def test_python_dict(corenlp_client):
    """ Test using a Python dictionary to specify all request properties """
    ann = corenlp_client.annotate(ES_DOC, properties=ES_PROPS, output_format="text")
    assert ann.strip() == ES_PROPS_GOLD.strip()
    ann = corenlp_client.annotate(FRENCH_DOC, properties=FRENCH_CUSTOM_PROPS)
    assert ann.strip() == FRENCH_CUSTOM_GOLD.strip()


def test_lang_setting(corenlp_client):
    """ Test using a Stanford CoreNLP supported languages as a properties key """
    ann = corenlp_client.annotate(GERMAN_DOC, properties="german", output_format="text")
    compare_ignoring_whitespace(ann, GERMAN_DOC_GOLD)


def test_annotators_and_output_format(corenlp_client):
    """ Test setting the annotators and output_format """
    ann = corenlp_client.annotate(FRENCH_DOC, properties=FRENCH_EXTRA_PROPS,
                                  annotators="tokenize,ssplit,mwt,pos", output_format="json")
    assert FRENCH_JSON_GOLD == ann
