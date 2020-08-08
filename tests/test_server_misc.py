"""
Misc tests for the server
"""

import pytest
import re
import stanza.server as corenlp
from tests import compare_ignoring_whitespace

pytestmark = pytest.mark.client

EN_DOC = "Joe Smith lives in California."

EN_DOC_GOLD = """
Sentence #1 (6 tokens):
Joe Smith lives in California.

Tokens:
[Text=Joe CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=PROPN Lemma=joe NamedEntityTag=PERSON]
[Text=Smith CharacterOffsetBegin=4 CharacterOffsetEnd=9 PartOfSpeech=PROPN Lemma=smith NamedEntityTag=PERSON]
[Text=lives CharacterOffsetBegin=10 CharacterOffsetEnd=15 PartOfSpeech=ADJ Lemma=lives NamedEntityTag=O]
[Text=in CharacterOffsetBegin=16 CharacterOffsetEnd=18 PartOfSpeech=ADP Lemma=in NamedEntityTag=O]
[Text=California CharacterOffsetBegin=19 CharacterOffsetEnd=29 PartOfSpeech=PROPN Lemma=california NamedEntityTag=STATE_OR_PROVINCE]
[Text=. CharacterOffsetBegin=29 CharacterOffsetEnd=30 PartOfSpeech=PUNCT Lemma=. NamedEntityTag=O]

Dependency Parse (enhanced plus plus dependencies):
root(ROOT-0, Joe-1)
flat(Joe-1, Smith-2)
amod(Joe-1, lives-3)
case(California-5, in-4)
nmod(lives-3, California-5)
punct(Joe-1, .-6)

Extracted the following NER entity mentions:
Joe Smith       PERSON  PERSON:0.9989563558707318
California      STATE_OR_PROVINCE       LOCATION:0.9911262738614187
"""


EN_DOC_POS_ONLY_GOLD = """
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

def test_english_request():
    """ Test case of starting server with Spanish defaults, and then requesting default English properties """
    with corenlp.CoreNLPClient(properties='spanish', server_id='test_english_request') as client:
        ann = client.annotate(EN_DOC, properties='english', output_format='text')
        compare_ignoring_whitespace(ann, EN_DOC_GOLD)


def test_default_annotators():
    """
    Test case of creating a client with start_server=False and a set of annotators
    The annotators should be used instead of the server's default annotators
    """
    with corenlp.CoreNLPClient(server_id='test_default_annotators',
                               output_format='text',
                               annotators=['tokenize','ssplit','pos','lemma','ner','depparse']) as client:
        with corenlp.CoreNLPClient(start_server=False,
                                   output_format='text',
                                   annotators=['tokenize','ssplit','pos']) as client2:
            ann = client2.annotate(EN_DOC)
            print(ann)

expected_codepoints = ((0, 1), (2, 4), (5, 8), (9, 15), (16, 20))
expected_characters = ((0, 1), (2, 4), (5, 10), (11, 17), (18, 22))
codepoint_doc = "I am 𝒚̂𝒊 random text"

def test_codepoints():
    """ Test case of asking for codepoints from the English tokenizer """
    with corenlp.CoreNLPClient(annotators=['tokenize','ssplit'], # 'depparse','coref'],
                               properties={'tokenize.codepoint': 'true'}) as client:
        ann = client.annotate(codepoint_doc)
        for i, (codepoints, characters) in enumerate(zip(expected_codepoints, expected_characters)):
            token = ann.sentence[0].token[i]
            assert token.codepointOffsetBegin == codepoints[0]
            assert token.codepointOffsetEnd == codepoints[1]
            assert token.beginChar == characters[0]
            assert token.endChar == characters[1]
