"""
Basic testing of part of speech tagging
"""

import stanfordnlp

from tests import *

EN_DOC = "Joe Smith was born in California."


EN_DOC_GOLD = """
<Token index=1;words=[<Word index=1;text=Joe;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token index=2;words=[<Word index=2;text=Smith;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token index=3;words=[<Word index=3;text=was;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin>]>
<Token index=4;words=[<Word index=4;text=born;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass>]>
<Token index=5;words=[<Word index=5;text=in;upos=ADP;xpos=IN;feats=_>]>
<Token index=6;words=[<Word index=6;text=California;upos=PROPN;xpos=NNP;feats=Number=Sing>]>
<Token index=7;words=[<Word index=7;text=.;upos=PUNCT;xpos=.;feats=_>]>
""".strip()


def test_part_of_speech():
    nlp = stanfordnlp.Pipeline(**{'processors': 'tokenize,pos', 'models_dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
