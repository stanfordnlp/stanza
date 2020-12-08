"""
Basic testing of tokenization
"""

import pytest
import stanza

from tests import *

pytestmark = pytest.mark.pipeline

EN_DOC = "Joe Smith lives in California. Joe's favorite food is pizza. He enjoys going to the beach."
EN_DOC_WITH_EXTRA_WHITESPACE = "Joe   Smith \n lives in\n California.   Joe's    favorite food \tis pizza. \t\t\tHe enjoys \t\tgoing to the beach."
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

JA_DOC = "北京は中国の首都です。 北京の人口は2152万人です。\n" # add some random whitespaces that need to be skipped
JA_DOC_GOLD_TOKENS = """
<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=は>]>
<Token id=3;words=[<Word id=3;text=中国>]>
<Token id=4;words=[<Word id=4;text=の>]>
<Token id=5;words=[<Word id=5;text=首都>]>
<Token id=6;words=[<Word id=6;text=です>]>
<Token id=7;words=[<Word id=7;text=。>]>

<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=の>]>
<Token id=3;words=[<Word id=3;text=人口>]>
<Token id=4;words=[<Word id=4;text=は>]>
<Token id=5;words=[<Word id=5;text=2152万>]>
<Token id=6;words=[<Word id=6;text=人>]>
<Token id=7;words=[<Word id=7;text=です>]>
<Token id=8;words=[<Word id=8;text=。>]>
""".strip()

JA_DOC_GOLD_NOSSPLIT_TOKENS = """
<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=は>]>
<Token id=3;words=[<Word id=3;text=中国>]>
<Token id=4;words=[<Word id=4;text=の>]>
<Token id=5;words=[<Word id=5;text=首都>]>
<Token id=6;words=[<Word id=6;text=です>]>
<Token id=7;words=[<Word id=7;text=。>]>
<Token id=8;words=[<Word id=8;text=北京>]>
<Token id=9;words=[<Word id=9;text=の>]>
<Token id=10;words=[<Word id=10;text=人口>]>
<Token id=11;words=[<Word id=11;text=は>]>
<Token id=12;words=[<Word id=12;text=2152万>]>
<Token id=13;words=[<Word id=13;text=人>]>
<Token id=14;words=[<Word id=14;text=です>]>
<Token id=15;words=[<Word id=15;text=。>]>
""".strip()

ZH_DOC = "北京是中国的首都。 北京有2100万人口，是一个直辖市。\n"
ZH_DOC_GOLD_TOKENS = """
<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=是>]>
<Token id=3;words=[<Word id=3;text=中国>]>
<Token id=4;words=[<Word id=4;text=的>]>
<Token id=5;words=[<Word id=5;text=首都>]>
<Token id=6;words=[<Word id=6;text=。>]>

<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=有>]>
<Token id=3;words=[<Word id=3;text=2100>]>
<Token id=4;words=[<Word id=4;text=万>]>
<Token id=5;words=[<Word id=5;text=人口>]>
<Token id=6;words=[<Word id=6;text=，>]>
<Token id=7;words=[<Word id=7;text=是>]>
<Token id=8;words=[<Word id=8;text=一个>]>
<Token id=9;words=[<Word id=9;text=直辖市>]>
<Token id=10;words=[<Word id=10;text=。>]>
""".strip()

ZH_DOC_GOLD_NOSSPLIT_TOKENS = """
<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=是>]>
<Token id=3;words=[<Word id=3;text=中国>]>
<Token id=4;words=[<Word id=4;text=的>]>
<Token id=5;words=[<Word id=5;text=首都>]>
<Token id=6;words=[<Word id=6;text=。>]>
<Token id=7;words=[<Word id=7;text=北京>]>
<Token id=8;words=[<Word id=8;text=有>]>
<Token id=9;words=[<Word id=9;text=2100>]>
<Token id=10;words=[<Word id=10;text=万>]>
<Token id=11;words=[<Word id=11;text=人口>]>
<Token id=12;words=[<Word id=12;text=，>]>
<Token id=13;words=[<Word id=13;text=是>]>
<Token id=14;words=[<Word id=14;text=一个>]>
<Token id=15;words=[<Word id=15;text=直辖市>]>
<Token id=16;words=[<Word id=16;text=。>]>
""".strip()

def test_tokenize():
    nlp = stanza.Pipeline(processors='tokenize', dir=TEST_MODELS_DIR, lang='en')
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_tokenize_ssplit_robustness():
    nlp = stanza.Pipeline(processors='tokenize', dir=TEST_MODELS_DIR, lang='en')
    doc = nlp(EN_DOC_WITH_EXTRA_WHITESPACE)
    assert EN_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_pretokenized():
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'tokenize_pretokenized': True})
    doc = nlp(EN_DOC_PRETOKENIZED)
    assert EN_DOC_PRETOKENIZED_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])
    doc = nlp(EN_DOC_PRETOKENIZED_LIST)
    assert EN_DOC_PRETOKENIZED_LIST_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_no_ssplit():
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'tokenize_no_ssplit': True})

    doc = nlp(EN_DOC_NO_SSPLIT)
    assert EN_DOC_NO_SSPLIT_SENTENCES == [[w.text for w in s.words] for s in doc.sentences]
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_spacy():
    nlp = stanza.Pipeline(processors='tokenize', dir=TEST_MODELS_DIR, lang='en', tokenize_with_spacy=True)
    doc = nlp(EN_DOC)

    # make sure the loaded tokenizer is actually spacy
    assert "SpacyTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert EN_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_sudachipy():
    nlp = stanza.Pipeline(lang='ja', dir=TEST_MODELS_DIR, processors={'tokenize': 'sudachipy'}, package=None)
    doc = nlp(JA_DOC)

    assert "SudachiPyTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert JA_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_sudachipy_no_ssplit():
    nlp = stanza.Pipeline(lang='ja', dir=TEST_MODELS_DIR, processors={'tokenize': 'sudachipy'}, tokenize_no_ssplit=True, package=None)
    doc = nlp(JA_DOC)

    assert "SudachiPyTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert JA_DOC_GOLD_NOSSPLIT_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_jieba():
    nlp = stanza.Pipeline(lang='zh', dir=TEST_MODELS_DIR, processors={'tokenize': 'jieba'}, package=None)
    doc = nlp(ZH_DOC)

    assert "JiebaTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert ZH_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_jieba_no_ssplit():
    nlp = stanza.Pipeline(lang='zh', dir=TEST_MODELS_DIR, processors={'tokenize': 'jieba'}, tokenize_no_ssplit=True, package=None)
    doc = nlp(ZH_DOC)

    assert "JiebaTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert ZH_DOC_GOLD_NOSSPLIT_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])
