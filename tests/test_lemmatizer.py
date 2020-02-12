"""
Basic testing of lemmatization
"""

import pytest
import stanfordnlp

from tests import *

pytestmark = pytest.mark.pipeline

EN_DOC = "Joe Smith was born in California."

EN_DOC_IDENTITY_GOLD = """
<Token id=1;words=[<Word id=1;text=Joe;lemma=Joe>]>
<Token id=2;words=[<Word id=2;text=Smith;lemma=Smith>]>
<Token id=3;words=[<Word id=3;text=was;lemma=was>]>
<Token id=4;words=[<Word id=4;text=born;lemma=born>]>
<Token id=5;words=[<Word id=5;text=in;lemma=in>]>
<Token id=6;words=[<Word id=6;text=California;lemma=California>]>
<Token id=7;words=[<Word id=7;text=.;lemma=.>]>
""".strip()

EN_DOC_LEMMATIZER_MODEL_GOLD = """
<Token id=1;words=[<Word id=1;text=Joe;lemma=Joe;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token id=2;words=[<Word id=2;text=Smith;lemma=Smith;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token id=3;words=[<Word id=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin>]>
<Token id=4;words=[<Word id=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass>]>
<Token id=5;words=[<Word id=5;text=in;lemma=in;upos=ADP;xpos=IN>]>
<Token id=6;words=[<Word id=6;text=California;lemma=California;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token id=7;words=[<Word id=7;text=.;lemma=.;upos=PUNCT;xpos=.>]>
""".strip()


def test_identity_lemmatizer():
    nlp = stanfordnlp.Pipeline(**{'processors': 'tokenize,lemma', 'models_dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'lemma_use_identity': True})
    doc = nlp(EN_DOC)
    assert EN_DOC_IDENTITY_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])


def test_full_lemmatizer():
    nlp = stanfordnlp.Pipeline(**{'processors': 'tokenize,pos,lemma', 'models_dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp(EN_DOC)
    assert EN_DOC_LEMMATIZER_MODEL_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])



