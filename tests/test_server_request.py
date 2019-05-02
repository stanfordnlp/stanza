"""
Tests for setting request properties of servers
"""

import pytest
import stanfordnlp.server as corenlp

from stanfordnlp.protobuf import Document

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
"""

FRENCH_CUSTOM_PROPS = {'annotators': 'tokenize,ssplit,pos,parse', 'tokenize.language': 'fr',
                       'pos.model': 'edu/stanford/nlp/models/pos-tagger/french/french.tagger',
                       'parse.model': 'edu/stanford/nlp/models/lexparser/frenchFactored.ser.gz',
                       'outputFormat': 'text'}

FRENCH_DOC = "Cette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt."

FRENCH_CUSTOM_GOLD = """
Sentence #1 (16 tokens):
Cette enquête préliminaire fait suite aux révélations de l’hebdomadaire quelques jours plus tôt.

Tokens:
[Text=Cette CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=DET]
[Text=enquête CharacterOffsetBegin=6 CharacterOffsetEnd=13 PartOfSpeech=NC]
[Text=préliminaire CharacterOffsetBegin=14 CharacterOffsetEnd=26 PartOfSpeech=ADJ]
[Text=fait CharacterOffsetBegin=27 CharacterOffsetEnd=31 PartOfSpeech=V]
[Text=suite CharacterOffsetBegin=32 CharacterOffsetEnd=37 PartOfSpeech=N]
[Text=à CharacterOffsetBegin=38 CharacterOffsetEnd=39 PartOfSpeech=P]
[Text=les CharacterOffsetBegin=39 CharacterOffsetEnd=41 PartOfSpeech=DET]
[Text=révélations CharacterOffsetBegin=42 CharacterOffsetEnd=53 PartOfSpeech=NC]
[Text=de CharacterOffsetBegin=54 CharacterOffsetEnd=56 PartOfSpeech=P]
[Text=l' CharacterOffsetBegin=57 CharacterOffsetEnd=59 PartOfSpeech=DET]
[Text=hebdomadaire CharacterOffsetBegin=59 CharacterOffsetEnd=71 PartOfSpeech=NC]
[Text=quelques CharacterOffsetBegin=72 CharacterOffsetEnd=80 PartOfSpeech=DET]
[Text=jours CharacterOffsetBegin=81 CharacterOffsetEnd=86 PartOfSpeech=NC]
[Text=plus CharacterOffsetBegin=87 CharacterOffsetEnd=91 PartOfSpeech=ADV]
[Text=tôt CharacterOffsetBegin=92 CharacterOffsetEnd=95 PartOfSpeech=ADV]
[Text=. CharacterOffsetBegin=95 CharacterOffsetEnd=96 PartOfSpeech=PUNC]

Constituency parse: 
(ROOT
  (SENT
    (NP (DET Cette) (NC enquête)
      (AP (ADJ préliminaire)))
    (VN
      (MWV (V fait) (N suite)))
    (PP (P à)
      (NP (DET les) (NC révélations)
        (PP (P de)
          (NP (DET l') (NC hebdomadaire)
            (AdP
              (NP (DET quelques) (NC jours))
              (ADV plus) (ADV tôt))))))
    (PUNC .)))
"""


@pytest.fixture(scope="module")
def corenlp_client():
    """ Client to run tests on """
    client = corenlp.CoreNLPClient(annotators='tokenize,ssplit,pos', server_id='stanfordnlp_request_tests_server')
    client.register_properties_key('fr-custom', FRENCH_CUSTOM_PROPS)
    yield client
    client.stop()


def test_basic(corenlp_client):
    """ Basic test of making a request, test default output format is a Document """
    ann = corenlp_client.annotate(EN_DOC, output_format="text")
    assert ann.strip() == EN_DOC_GOLD.strip()
    ann = corenlp_client.annotate(EN_DOC)
    assert isinstance(ann, Document)


def test_properties_key(corenlp_client):
    """ Test using the properties_key which was registered with the properties cache """
    ann = corenlp_client.annotate(FRENCH_DOC, properties_key='fr-custom')
    assert ann.strip() == FRENCH_CUSTOM_GOLD.strip()


def test_lang_setting(corenlp_client):
    """ Test using a Stanford CoreNLP supported languages as a properties key """
    ann = corenlp_client.annotate(GERMAN_DOC, properties_key="german", output_format="text")
    assert ann.strip() == GERMAN_DOC_GOLD.strip()
