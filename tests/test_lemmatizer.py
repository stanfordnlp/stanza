"""
Basic testing of lemmatization
"""

import stanfordnlp

from tests import *

EN_DOC = "Joe Smith was born in California."

EN_DOC_IDENTITY_GOLD = """
<Token index=1;words=[<Word index=1;text=Joe;lemma=Joe>]>
<Token index=2;words=[<Word index=2;text=Smith;lemma=Smith>]>
<Token index=3;words=[<Word index=3;text=was;lemma=was>]>
<Token index=4;words=[<Word index=4;text=born;lemma=born>]>
<Token index=5;words=[<Word index=5;text=in;lemma=in>]>
<Token index=6;words=[<Word index=6;text=California;lemma=California>]>
<Token index=7;words=[<Word index=7;text=.;lemma=.>]>
""".strip()

EN_DOC_LEMMATIZER_MODEL_GOLD = """
<Token index=1;words=[<Word index=1;text=Joe;lemma=Joe;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token index=2;words=[<Word index=2;text=Smith;lemma=Smith;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token index=3;words=[<Word index=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin>]>
<Token index=4;words=[<Word index=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass>]>
<Token index=5;words=[<Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_>]>
<Token index=6;words=[<Word index=6;text=California;lemma=California;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token index=7;words=[<Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_>]>
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



