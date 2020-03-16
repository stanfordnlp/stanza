"""
Misc tests for the server
"""

import pytest
import stanza.server as corenlp

pytestmark = pytest.mark.client

EN_DOC = "Joe Smith lives in California."

EN_DOC_GOLD = """
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
nmod(lives-3, California-5)
punct(lives-3, .-6)

Extracted the following NER entity mentions:
Joe Smith	PERSON
California	STATE_OR_PROVINCE
"""


def test_english_request():
    """ Test case of starting server with Spanish defaults, and then requesting default English properties """
    with corenlp.CoreNLPClient(properties='spanish', server_id='test_english_request') as client:
        ann = client.annotate(EN_DOC, properties_key='english', output_format='text')
        assert ann.strip() == EN_DOC_GOLD.strip()


