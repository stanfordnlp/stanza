"""
Basic testing of tokenization
"""

import stanfordnlp

from tests import *


EN_DOC = "Joe Smith lives in California. Joe's favorite food is pizza. He enjoys going to the beach."

EN_DOC_GOLD_TOKENS = """
<Token id=1;words=[<Word id=1;text=Joe>]>
<Token id=2;words=[<Word id=2;text=Smith>]>
<Token id=3;words=[<Word id=3;text=lives>]>
<Token id=4;words=[<Word id=4;text=in>]>
<Token id=5;words=[<Word id=5;text=California>]>
<Token id=6;words=[<Word id=6;text=.>]>

<Token id=1;words=[<Word id=1;text=Joe>]>
<Token id=2;words=[<Word id=2;text='s>]>
<Token id=3;words=[<Word id=3;text=favorite>]>
<Token id=4;words=[<Word id=4;text=food>]>
<Token id=5;words=[<Word id=5;text=is>]>
<Token id=6;words=[<Word id=6;text=pizza>]>
<Token id=7;words=[<Word id=7;text=.>]>

<Token id=1;words=[<Word id=1;text=He>]>
<Token id=2;words=[<Word id=2;text=enjoys>]>
<Token id=3;words=[<Word id=3;text=going>]>
<Token id=4;words=[<Word id=4;text=to>]>
<Token id=5;words=[<Word id=5;text=the>]>
<Token id=6;words=[<Word id=6;text=beach>]>
<Token id=7;words=[<Word id=7;text=.>]>
""".strip()


EN_DOC_PRETOKENIZED = \
    "Joe Smith lives in California .\nJoe's favorite  food is  pizza .\n\nHe enjoys going to the beach.\n"

EN_DOC_PRETOKENIZED_GOLD_TOKENS = """
<Token id=1;words=[<Word id=1;text=Joe>]>
<Token id=2;words=[<Word id=2;text=Smith>]>
<Token id=3;words=[<Word id=3;text=lives>]>
<Token id=4;words=[<Word id=4;text=in>]>
<Token id=5;words=[<Word id=5;text=California>]>
<Token id=6;words=[<Word id=6;text=.>]>

<Token id=1;words=[<Word id=1;text=Joe's>]>
<Token id=2;words=[<Word id=2;text=favorite>]>
<Token id=3;words=[<Word id=3;text=food>]>
<Token id=4;words=[<Word id=4;text=is>]>
<Token id=5;words=[<Word id=5;text=pizza>]>
<Token id=6;words=[<Word id=6;text=.>]>

<Token id=1;words=[<Word id=1;text=He>]>
<Token id=2;words=[<Word id=2;text=enjoys>]>
<Token id=3;words=[<Word id=3;text=going>]>
<Token id=4;words=[<Word id=4;text=to>]>
<Token id=5;words=[<Word id=5;text=the>]>
<Token id=6;words=[<Word id=6;text=beach.>]>
""".strip()


EN_DOC_PRETOKENIZED_LIST = [['Joe', 'Smith', 'lives', 'in', 'California', '.'], ['He', 'loves', 'pizza', '.']]

EN_DOC_PRETOKENIZED_LIST_GOLD_TOKENS = """
<Token id=1;words=[<Word id=1;text=Joe>]>
<Token id=2;words=[<Word id=2;text=Smith>]>
<Token id=3;words=[<Word id=3;text=lives>]>
<Token id=4;words=[<Word id=4;text=in>]>
<Token id=5;words=[<Word id=5;text=California>]>
<Token id=6;words=[<Word id=6;text=.>]>

<Token id=1;words=[<Word id=1;text=He>]>
<Token id=2;words=[<Word id=2;text=loves>]>
<Token id=3;words=[<Word id=3;text=pizza>]>
<Token id=4;words=[<Word id=4;text=.>]>
""".strip()

EN_DOC_NO_SSPLIT = ["This is a sentence. This is another.", "This is a third."]

EN_DOC_NO_SSPLIT_SENTENCES = [['This', 'is', 'a', 'sentence', '.', 'This', 'is', 'another', '.'], ['This', 'is', 'a', 'third', '.']]


def test_tokenize():
    nlp = stanfordnlp.Pipeline(processors='tokenize', models_dir=TEST_MODELS_DIR, lang='en')
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])


def test_pretokenized():
    nlp = stanfordnlp.Pipeline(**{'processors': 'tokenize', 'models_dir': '.', 'lang': 'en',
                                  'tokenize_pretokenized': True})
    doc = nlp(EN_DOC_PRETOKENIZED)
    assert EN_DOC_PRETOKENIZED_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    doc = nlp(EN_DOC_PRETOKENIZED_LIST)
    assert EN_DOC_PRETOKENIZED_LIST_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])

def test_no_ssplit():
    nlp = stanfordnlp.Pipeline(**{'processors': 'tokenize', 'models_dir': '.', 'lang': 'en',
                                  'tokenize_no_ssplit': True})

    doc = nlp(EN_DOC_NO_SSPLIT)
    assert EN_DOC_NO_SSPLIT_SENTENCES == [[w.text for w in s.words] for s in doc.sentences]