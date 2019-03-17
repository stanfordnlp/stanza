"""
Basic testing of tokenization
"""

import stanfordnlp

from tests import *


def setup_module(module):
    """Set up resources for all tests in this module"""
    safe_rm(EN_MODELS_DIR)
    stanfordnlp.download('en', resource_dir=TEST_WORKING_DIR, force=True)


def teardown_module(module):
    """Clean up resources after tests complete"""
    safe_rm(EN_MODELS_DIR)


EN_DOC = "Joe Smith lives in California. Joe's favorite food is pizza. He enjoys going to the beach."

EN_DOC_GOLD_TOKENS = """
<Token index=1;words=[<Word index=1;text=Joe>]>
<Token index=2;words=[<Word index=2;text=Smith>]>
<Token index=3;words=[<Word index=3;text=lives>]>
<Token index=4;words=[<Word index=4;text=in>]>
<Token index=5;words=[<Word index=5;text=California>]>
<Token index=6;words=[<Word index=6;text=.>]>

<Token index=1;words=[<Word index=1;text=Joe>]>
<Token index=2;words=[<Word index=2;text='s>]>
<Token index=3;words=[<Word index=3;text=favorite>]>
<Token index=4;words=[<Word index=4;text=food>]>
<Token index=5;words=[<Word index=5;text=is>]>
<Token index=6;words=[<Word index=6;text=pizza>]>
<Token index=7;words=[<Word index=7;text=.>]>

<Token index=1;words=[<Word index=1;text=He>]>
<Token index=2;words=[<Word index=2;text=enjoys>]>
<Token index=3;words=[<Word index=3;text=going>]>
<Token index=4;words=[<Word index=4;text=to>]>
<Token index=5;words=[<Word index=5;text=the>]>
<Token index=6;words=[<Word index=6;text=beach>]>
<Token index=7;words=[<Word index=7;text=.>]>
""".strip()


EN_DOC_PRETOKENIZED = \
    "Joe Smith lives in California .\nJoe's favorite  food is  pizza .\n\nHe enjoys going to the beach.\n"

EN_DOC_PRETOKENIZED_GOLD_TOKENS = """
<Token index=1;words=[<Word index=1;text=Joe>]>
<Token index=2;words=[<Word index=2;text=Smith>]>
<Token index=3;words=[<Word index=3;text=lives>]>
<Token index=4;words=[<Word index=4;text=in>]>
<Token index=5;words=[<Word index=5;text=California>]>
<Token index=6;words=[<Word index=6;text=.>]>

<Token index=1;words=[<Word index=1;text=Joe's>]>
<Token index=2;words=[<Word index=2;text=favorite>]>
<Token index=3;words=[<Word index=3;text=food>]>
<Token index=4;words=[<Word index=4;text=is>]>
<Token index=5;words=[<Word index=5;text=pizza>]>
<Token index=6;words=[<Word index=6;text=.>]>

<Token index=1;words=[<Word index=1;text=He>]>
<Token index=2;words=[<Word index=2;text=enjoys>]>
<Token index=3;words=[<Word index=3;text=going>]>
<Token index=4;words=[<Word index=4;text=to>]>
<Token index=5;words=[<Word index=5;text=the>]>
<Token index=6;words=[<Word index=6;text=beach.>]>
""".strip()


def test_tokenize():
    nlp = stanfordnlp.Pipeline(processors='tokenize', models_dir=TEST_WORKING_DIR, lang='en')
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])


def test_pretokenized():
    nlp = stanfordnlp.Pipeline(**{'processors': 'tokenize', 'models_dir': '.', 'lang': 'en',
                                  'tokenize_pretokenized': True})
    doc = nlp(EN_DOC_PRETOKENIZED)
    assert EN_DOC_PRETOKENIZED_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
