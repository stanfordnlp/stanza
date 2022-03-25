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


def test_part_of_speech():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos', 'dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
