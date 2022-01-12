"""
Basic testing of tokenization
"""

import pytest
import stanza

from stanza.tests import *

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

EN_DOC_GOLD_NOSSPLIT_TOKENS = """
<Token id=1;words=[<Word id=1;text=Joe>]>
<Token id=2;words=[<Word id=2;text=Smith>]>
<Token id=3;words=[<Word id=3;text=lives>]>
<Token id=4;words=[<Word id=4;text=in>]>
<Token id=5;words=[<Word id=5;text=California>]>
<Token id=6;words=[<Word id=6;text=.>]>
<Token id=7;words=[<Word id=7;text=Joe>]>
<Token id=8;words=[<Word id=8;text='s>]>
<Token id=9;words=[<Word id=9;text=favorite>]>
<Token id=10;words=[<Word id=10;text=food>]>
<Token id=11;words=[<Word id=11;text=is>]>
<Token id=12;words=[<Word id=12;text=pizza>]>
<Token id=13;words=[<Word id=13;text=.>]>
<Token id=14;words=[<Word id=14;text=He>]>
<Token id=15;words=[<Word id=15;text=enjoys>]>
<Token id=16;words=[<Word id=16;text=going>]>
<Token id=17;words=[<Word id=17;text=to>]>
<Token id=18;words=[<Word id=18;text=the>]>
<Token id=19;words=[<Word id=19;text=beach>]>
<Token id=20;words=[<Word id=20;text=.>]>
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
ZH_DOC1 = "北\n京是中\n国的首\n都。 北京有2100万人口，是一个直辖市。\n"
ZH_DOC2 = "北\n京是中\n国的首\n都。\n\n 北京有2100万人口，是一个直辖市。\n"
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

ZH_DOC1_GOLD_TOKENS="""
<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=是>]>
<Token id=3;words=[<Word id=3;text=中国>]>
<Token id=4;words=[<Word id=4;text=的>]>
<Token id=5;words=[<Word id=5;text=首都>]>
<Token id=6;words=[<Word id=6;text=。>]>

<Token id=1;words=[<Word id=1;text=北京>]>
<Token id=2;words=[<Word id=2;text=有>]>
<Token id=3;words=[<Word id=3;text=2100万>]>
<Token id=4;words=[<Word id=4;text=人口>]>
<Token id=5;words=[<Word id=5;text=，>]>
<Token id=6;words=[<Word id=6;text=是>]>
<Token id=7;words=[<Word id=7;text=一>]>
<Token id=8;words=[<Word id=8;text=个>]>
<Token id=9;words=[<Word id=9;text=直辖>]>
<Token id=10;words=[<Word id=10;text=市>]>
<Token id=11;words=[<Word id=11;text=。>]>
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

ZH_PARENS_DOC = "我们一起学(猫叫)"

TH_DOC = "ข้าราชการได้รับการหมุนเวียนเป็นระยะ และเขาได้รับมอบหมายให้ประจำในระดับภูมิภาค"
TH_DOC_GOLD_TOKENS = """
<Token id=1;words=[<Word id=1;text=ข้าราชการ>]>
<Token id=2;words=[<Word id=2;text=ได้รับ>]>
<Token id=3;words=[<Word id=3;text=การ>]>
<Token id=4;words=[<Word id=4;text=หมุนเวียน>]>
<Token id=5;words=[<Word id=5;text=เป็นระยะ>]>

<Token id=1;words=[<Word id=1;text=และ>]>
<Token id=2;words=[<Word id=2;text=เขา>]>
<Token id=3;words=[<Word id=3;text=ได้>]>
<Token id=4;words=[<Word id=4;text=รับมอบหมาย>]>
<Token id=5;words=[<Word id=5;text=ให้>]>
<Token id=6;words=[<Word id=6;text=ประจำ>]>
<Token id=7;words=[<Word id=7;text=ใน>]>
<Token id=8;words=[<Word id=8;text=ระดับ>]>
<Token id=9;words=[<Word id=9;text=ภูมิภาค>]>
""".strip()

TH_DOC_GOLD_NOSSPLIT_TOKENS = """
<Token id=1;words=[<Word id=1;text=ข้าราชการ>]>
<Token id=2;words=[<Word id=2;text=ได้รับ>]>
<Token id=3;words=[<Word id=3;text=การ>]>
<Token id=4;words=[<Word id=4;text=หมุนเวียน>]>
<Token id=5;words=[<Word id=5;text=เป็นระยะ>]>
<Token id=6;words=[<Word id=6;text=และ>]>
<Token id=7;words=[<Word id=7;text=เขา>]>
<Token id=8;words=[<Word id=8;text=ได้>]>
<Token id=9;words=[<Word id=9;text=รับมอบหมาย>]>
<Token id=10;words=[<Word id=10;text=ให้>]>
<Token id=11;words=[<Word id=11;text=ประจำ>]>
<Token id=12;words=[<Word id=12;text=ใน>]>
<Token id=13;words=[<Word id=13;text=ระดับ>]>
<Token id=14;words=[<Word id=14;text=ภูมิภาค>]>
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

def test_pretokenized_multidoc():
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'tokenize_pretokenized': True})
    doc = nlp(EN_DOC_PRETOKENIZED)
    assert EN_DOC_PRETOKENIZED_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])
    doc = nlp([stanza.Document([], text=EN_DOC_PRETOKENIZED_LIST)])[0]
    assert EN_DOC_PRETOKENIZED_LIST_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_no_ssplit():
    nlp = stanza.Pipeline(**{'processors': 'tokenize', 'dir': TEST_MODELS_DIR, 'lang': 'en',
                                  'tokenize_no_ssplit': True})

    doc = nlp(EN_DOC_NO_SSPLIT)
    assert EN_DOC_NO_SSPLIT_SENTENCES == [[w.text for w in s.words] for s in doc.sentences]
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_zh_tokenizer_skip_newline():
    nlp = stanza.Pipeline(lang='zh', processors='tokenize', dir=TEST_MODELS_DIR)
    doc = nlp(ZH_DOC1)

    assert ZH_DOC1_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char].replace('\n', '') == token.text for sent in doc.sentences for token in sent.tokens])

def test_zh_tokenizer_skip_newline_offsets():
    nlp = stanza.Pipeline(lang='zh', processors='tokenize', dir=TEST_MODELS_DIR)
    doc = nlp(ZH_DOC2)

    assert ZH_DOC1_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char].replace('\n', '') == token.text for sent in doc.sentences for token in sent.tokens])

def test_zh_tokenizer_parens():
    """
    The original fix for newlines in Chinese text broke () in Chinese text
    """
    nlp = stanza.Pipeline(lang='zh', processors="tokenize", dir=TEST_MODELS_DIR)
    doc = nlp(ZH_PARENS_DOC)

    # ... the results are kind of bad for this expression, so no testing of the results yet
    #assert ZH_PARENS_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])

def test_spacy():
    nlp = stanza.Pipeline(processors='tokenize', dir=TEST_MODELS_DIR, lang='en', tokenize_with_spacy=True)
    doc = nlp(EN_DOC)

    # make sure the loaded tokenizer is actually spacy
    assert "SpacyTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert EN_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_spacy_no_ssplit():
    nlp = stanza.Pipeline(processors='tokenize', dir=TEST_MODELS_DIR, lang='en', tokenize_with_spacy=True, tokenize_no_ssplit=True)
    doc = nlp(EN_DOC)

    # make sure the loaded tokenizer is actually spacy
    assert "SpacyTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert EN_DOC_GOLD_NOSSPLIT_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
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

def test_pythainlp():
    nlp = stanza.Pipeline(lang='th', dir=TEST_MODELS_DIR, processors={'tokenize': 'pythainlp'}, package=None)
    doc = nlp(TH_DOC)
    assert "PyThaiNLPTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert TH_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

def test_pythainlp_no_ssplit():
    nlp = stanza.Pipeline(lang='th', dir=TEST_MODELS_DIR, processors={'tokenize': 'pythainlp'}, tokenize_no_ssplit=True, package=None)
    doc = nlp(TH_DOC)
    assert "PyThaiNLPTokenizer" == nlp.processors['tokenize']._variant.__class__.__name__
    assert TH_DOC_GOLD_NOSSPLIT_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    assert all([doc.text[token._start_char: token._end_char] == token.text for sent in doc.sentences for token in sent.tokens])

