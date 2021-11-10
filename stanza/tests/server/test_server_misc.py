"""
Misc tests for the server
"""

import pytest
import re
import stanza.server as corenlp
from stanza.tests import compare_ignoring_whitespace

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
obl:in(lives-3, California-5)
punct(lives-3, .-6)

Extracted the following NER entity mentions:
Joe Smith       PERSON  PERSON:0.9972202681743931
California      STATE_OR_PROVINCE       LOCATION:0.9990868267559281

Extracted the following KBP triples:
1.0     Joe Smith       per:statesorprovinces_of_residence      California
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
    with corenlp.CoreNLPClient(properties='spanish', server_id='test_spanish_english_request') as client:
        ann = client.annotate(EN_DOC, properties='english', output_format='text')
        compare_ignoring_whitespace(ann, EN_DOC_GOLD)

    # Rerun the test with a server created in English mode to verify
    # that the expected output is what the defaults actually give us
    with corenlp.CoreNLPClient(properties='english', server_id='test_english_request') as client:
        ann = client.annotate(EN_DOC, output_format='text')
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

def test_codepoint_text():
    """ Test case of extracting the correct sentence text using codepoints """

    text = 'Unban mox opal 🐱.  This is a second sentence.'

    with corenlp.CoreNLPClient(annotators=["tokenize","ssplit"],
                               properties={'tokenize.codepoint': 'true'}) as client:
        ann = client.annotate(text)

        text_start = ann.sentence[0].token[0].codepointOffsetBegin
        text_end = ann.sentence[0].token[-1].codepointOffsetEnd
        sentence_text = text[text_start:text_end]
        assert sentence_text == 'Unban mox opal 🐱.'

        text_start = ann.sentence[1].token[0].codepointOffsetBegin
        text_end = ann.sentence[1].token[-1].codepointOffsetEnd
        sentence_text = text[text_start:text_end]
        assert sentence_text == 'This is a second sentence.'
