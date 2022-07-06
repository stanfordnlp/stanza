"""
Basic testing of part of speech tagging
"""

import pytest
import stanza

from stanza.tests import *

pytestmark = pytest.mark.pipeline

EN_DOC = "Joe Smith was born in California."

EN_DOC_GOLD = """
<Token id=1;words=[<Word id=1;text=Joe;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token id=2;words=[<Word id=2;text=Smith;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token id=3;words=[<Word id=3;text=was;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin>]>
<Token id=4;words=[<Word id=4;text=born;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass>]>
<Token id=5;words=[<Word id=5;text=in;upos=ADP;xpos=IN>]>
<Token id=6;words=[<Word id=6;text=California;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token id=7;words=[<Word id=7;text=.;upos=PUNCT;xpos=.>]>
""".strip()

@pytest.fixture(scope="module")
def pos_pipeline():
    return stanza.Pipeline(**{'processors': 'tokenize,pos', 'dir': TEST_MODELS_DIR, 'lang': 'en'})

def test_part_of_speech(pos_pipeline):
    doc = pos_pipeline(EN_DOC)
    assert EN_DOC_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])

def test_get_known_xpos(pos_pipeline):
    tags = pos_pipeline.processors['pos'].get_known_xpos()
    # make sure we have xpos...
    assert 'DT' in tags
    # ... and not upos
    assert 'DET' not in tags

def test_get_known_upos(pos_pipeline):
    tags = pos_pipeline.processors['pos'].get_known_upos()
    # make sure we have upos...
    assert 'DET' in tags
    # ... and not xpos
    assert 'DT' not in tags


def test_get_known_feats(pos_pipeline):
    feats = pos_pipeline.processors['pos'].get_known_feats()
    # I appreciate how self-referential the Abbr feat is
    assert 'Abbr' in feats
    assert 'Yes' in feats['Abbr']
