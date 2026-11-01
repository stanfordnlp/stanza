"""
Tests for converting the SPECIALIST Lexicon inflection table to a lemma dictionary
"""

import argparse
from collections import Counter
import os

import pytest

from stanza.utils.conll import CoNLL
from stanza.utils.datasets import prepare_lemma_treebank
from stanza.utils.datasets.lemma import convert_en_specialist
from stanza.utils.training import run_lemma

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# Rows from the SPECIALIST Lexicon release, plus a few rows built for the
# cases the real data needs handled: an aux row, a negated contraction,
# and a (form, upos) pair with two lemmas
SAMPLE = """
household supply|128|1|E0032097|household supply|household supply
household supplies|128|8|E0032097|household supply|household supply
household supply|128|512|E0032097|household supply|household supply
house wife|128|1|E0032099|house wife|housewife
house-wife|128|1|E0032099|house-wife|housewife
housewife|128|1|E0032099|housewife|housewife
house wives|128|8|E0032099|house wife|housewife
house-wives|128|8|E0032099|house-wife|housewife
housewives|128|8|E0032099|housewife|housewife
house wife|128|512|E0032099|house wife|housewife
house-wife|128|512|E0032099|house-wife|housewife
housewife|128|512|E0032099|housewife|housewife
hover|1024|1|E0032104|hover|hover
hovered|1024|32|E0032104|hover|hover
hovers|1024|128|E0032104|hover|hover
hovered|1024|64|E0032104|hover|hover
hovering|1024|16|E0032104|hover|hover
hover|1024|262144|E0032104|hover|hover
hover|1024|1024|E0032104|hover|hover
how|2|1|E0032106|how|how
how|2|256|E0032106|how|how
dive|128|1|E9000001|dive|dive
dives|128|8|E9000001|dive|dive
dives|1024|128|E9000002|dive|dive
redder|1|2|E9000003|red|red
can|64|1|E9000004|can|can
didn't|1024|4096|E9000005|do|do
axes|128|8|E9000006|axe|axe
axes|128|8|E9000007|axis|axis
(1)O(2)|128|1|E9000008|(1)O(2)|(1)O(2)
%LN|128|1|E9000009|%LN|%LN
IL-2|128|1|E9000010|IL-2|IL-2
β-blockers|128|8|E9000011|β-blocker|β-blocker
Mondays|128|8|E9000012|Monday|Monday
American|1|1|E9000013|American|American
""".lstrip()

EXPECTED = {
    ('house-wife',  'NOUN'): 'house-wife',
    ('house-wives', 'NOUN'): 'house-wife',
    ('housewife',   'NOUN'): 'housewife',
    ('housewives',  'NOUN'): 'housewife',
    ('hover',       'VERB'): 'hover',
    ('hovered',     'VERB'): 'hover',
    ('hovering',    'VERB'): 'hover',
    ('hovers',      'VERB'): 'hover',
    ('how',         'ADV'):  'how',
    ('dive',        'NOUN'): 'dive',
    ('dives',       'NOUN'): 'dive',
    ('dives',       'VERB'): 'dive',
    ('redder',      'ADJ'):  'red',
    ('IL-2',        'PROPN'): 'IL-2',
    ('Mondays',     'PROPN'): 'Monday',
    ('American',    'ADJ'):  'American',
    ('β-blockers',  'NOUN'): 'β-blocker',
}

@pytest.fixture
def sample_file(tmp_path):
    filename = tmp_path / "inflVars.data"
    filename.write_text(SAMPLE, encoding="utf-8")
    return str(filename)

def test_build_lemma_dict(sample_file):
    lemma_dict, ambiguous = convert_en_specialist.build_lemma_dict(convert_en_specialist.read_inflection_table(sample_file))
    assert lemma_dict == EXPECTED
    assert ambiguous == {('axes', 'NOUN'): ['axe', 'axis']}

def test_crlf(tmp_path):
    filename = tmp_path / "inflVars.data"
    filename.write_bytes(SAMPLE.replace("\n", "\r\n").encode("utf-8"))
    lemma_dict, _ = convert_en_specialist.build_lemma_dict(convert_en_specialist.read_inflection_table(str(filename)))
    assert lemma_dict == EXPECTED

def test_bad_line(tmp_path):
    filename = tmp_path / "inflVars.data"
    filename.write_text("housewives|128|8\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(convert_en_specialist.read_inflection_table(str(filename)))

def test_convert_file(sample_file, tmp_path):
    output_file = str(tmp_path / "specialist_lemmas.conllu")
    convert_en_specialist.convert_file(sample_file, output_file, num_numbers=0)
    doc = CoNLL.conll2doc(input_file=output_file)
    assert len(doc.sentences) == len(EXPECTED)
    words = {(word.text, word.upos): word.lemma for sentence in doc.sentences for word in sentence.words}
    assert words == EXPECTED
    sent_ids = [sentence.sent_id for sentence in doc.sentences]
    assert len(set(sent_ids)) == len(sent_ids)
    assert "en_specialist-NOUN-housewives" in sent_ids
    assert "en_specialist-VERB-dives" in sent_ids
    assert "en_specialist-NOUN-dives" in sent_ids

def test_skipped_counts(sample_file):
    skipped = Counter()
    list(convert_en_specialist.read_inflection_table(sample_file, skipped))
    assert skipped == Counter({'category': 1, 'multiword': 6, 'negation': 1, 'symbols': 2})

def test_is_word():
    for text in ("housewives", "IL-2", "β-blocker", "o'clock", "Ph.D.", "mother-in-law"):
        assert convert_en_specialist.is_word(text), text
    for text in ("(1)O(2)", "%LN", "AT&T", "Na+", "123", "-", "and/or"):
        assert not convert_en_specialist.is_word(text), text

# hovered has a deliberately unusual lemma, so the tests can tell
# whether the training data or the SPECIALIST entry was used
TRAIN = """
# text = The dives hovered.
1	The	the	DET	DT	_	2	det	_	_
2	dives	dive	NOUN	NNS	_	3	nsubj	_	_
3	hovered	hovered	VERB	VBD	_	0	root	_	_
4	.	.	PUNCT	.	_	3	punct	_	_
""".lstrip()

def test_train_dict(sample_file):
    """
    Training lemmas win ties against the SPECIALIST entries, and new words are covered
    """
    lemma_dict, _ = convert_en_specialist.build_lemma_dict(convert_en_specialist.read_inflection_table(sample_file))
    train_triples = [(word.text, word.upos, word.lemma) for sentence in CoNLL.conll2doc(input_str=TRAIN).sentences for word in sentence.words]
    specialist_triples = [(form, upos, lemma) for (form, upos), lemma in lemma_dict.items()]
    trainer = convert_en_specialist.build_dict_trainer(train_triples + specialist_triples)

    pairs = [('hovered', 'VERB'), ('housewives', 'NOUN'), ('housewives', 'PROPN'), ('dives', 'VERB'), ('axes', 'NOUN')]
    assert trainer.skip_seq2seq(pairs) == [True, True, True, True, False]
    assert trainer.predict_dict(pairs) == ['hovered', 'housewife', 'housewife', 'dive', 'axes']

DEV = """
# text = Housewives hover.
1	housewives	housewife	NOUN	NNS	_	2	nsubj	_	_
2	hover	hover	VERB	VBP	_	0	root	_	_
3	.	.	PUNCT	.	_	2	punct	_	_
""".lstrip()

def test_report(sample_file, tmp_path):
    train_file = tmp_path / "train.conllu"
    train_file.write_text(TRAIN, encoding="utf-8")
    dev_file = tmp_path / "dev.conllu"
    dev_file.write_text(DEV, encoding="utf-8")
    output_file = tmp_path / "specialist_lemmas.conllu"
    report_file = tmp_path / "report.md"

    convert_en_specialist.main([sample_file, str(output_file),
                            "--train_files", str(train_file),
                            "--dev_files", str(dev_file),
                            "--report", str(report_file)])
    report = report_file.read_text(encoding="utf-8")
    assert "| 1 | hovered | VERB | hovered | hover |" in report
    assert "Newly covered by the dictionary: 2, of which 2 match the gold lemma" in report

    lemma_dict, _ = convert_en_specialist.build_lemma_dict(convert_en_specialist.read_inflection_table(sample_file))
    train_triples = convert_en_specialist.read_conllu_triples([str(train_file)])
    dev_triples = convert_en_specialist.read_conllu_triples([str(dev_file)])
    summary, changed = convert_en_specialist.dev_comparison(lemma_dict, train_triples, dev_triples)
    assert summary['words'] == 3
    assert summary['newly_covered'] == 2
    assert sorted(changed) == [('housewives', 'NOUN', 'housewife', None, 'housewife'),
                               ('hover', 'VERB', 'hover', None, 'hover')]

def build_paths(tmp_path):
    return {"STANZA_EXTERN_DIR": str(tmp_path / "extern"),
            "LEMMA_DATA_DIR":    str(tmp_path / "data" / "lemma")}

def write_infl_vars(tmp_path):
    specialist_dir = tmp_path / "extern" / "english" / "specialist"
    specialist_dir.mkdir(parents=True)
    (specialist_dir / "inflVars.data").write_text(SAMPLE, encoding="utf-8")

def test_prepare_missing_lexicon(tmp_path):
    """
    Preparing en_combined fails if inflVars.data is missing, and sl_combined fails without Sloleks
    """
    paths = build_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare_lemma_treebank.prepare_external_dictionary(paths, "en_combined")
    with pytest.raises(FileNotFoundError):
        prepare_lemma_treebank.prepare_external_dictionary(paths, "sl_combined")

def test_prepare_external_dictionary(tmp_path):
    """
    The dictionary is written next to the train/dev/test files, and removed if it is turned off
    """
    paths = build_paths(tmp_path)
    write_infl_vars(tmp_path)

    dict_file = prepare_lemma_treebank.prepare_external_dictionary(paths, "en_combined")
    assert dict_file == str(tmp_path / "data" / "lemma" / "en_combined.dict.conllu")
    words = [word for sentence in CoNLL.conll2doc(input_file=dict_file).sentences for word in sentence.words]
    assert len([word for word in words if word.upos != 'NUM']) == len(EXPECTED)
    assert len([word for word in words if word.upos == 'NUM']) == convert_en_specialist.DEFAULT_NUMBERS

    assert prepare_lemma_treebank.prepare_external_dictionary(paths, "en_combined", use_external_dictionary=False) is None
    assert not os.path.exists(dict_file)

    # datasets without an external lexicon don't get a dictionary file
    assert prepare_lemma_treebank.prepare_external_dictionary(paths, "en_ewt") is None
    assert not os.path.exists(tmp_path / "data" / "lemma" / "en_ewt.dict.conllu")

def secondary_args(secondary_file=None, use_secondary_file=True):
    return argparse.Namespace(secondary_file=secondary_file, use_secondary_file=use_secondary_file)

def test_find_secondary_file(tmp_path):
    """
    run_lemma uses the prepared dictionary if it exists, unless overridden or turned off
    """
    paths = build_paths(tmp_path)
    assert run_lemma.find_secondary_file(paths, "en_combined", secondary_args()) is None

    write_infl_vars(tmp_path)
    dict_file = prepare_lemma_treebank.prepare_external_dictionary(paths, "en_combined")
    assert run_lemma.find_secondary_file(paths, "en_combined", secondary_args()) == dict_file
    assert run_lemma.find_secondary_file(paths, "en_combined", secondary_args("other.conllu")) == "other.conllu"
    assert run_lemma.find_secondary_file(paths, "en_combined", secondary_args(use_secondary_file=False)) is None
    assert run_lemma.find_secondary_file(paths, "en_combined", secondary_args("other.conllu", use_secondary_file=False)) is None
    assert run_lemma.find_secondary_file(paths, "en_ewt", secondary_args()) is None

def test_number_entries():
    entries = convert_en_specialist.number_entries()
    assert len(entries) == convert_en_specialist.DEFAULT_NUMBERS
    assert entries == convert_en_specialist.number_entries()
    lemmas = dict(entries)
    assert lemmas["11,111"] == "11111"
    assert lemmas["40,000,000"] == "40000000"
    for form, lemma in entries:
        assert "," in form
        assert lemma == form.replace(",", "")

def test_convert_file_numbers(sample_file, tmp_path):
    output_file = str(tmp_path / "specialist_lemmas.conllu")
    lemma_dict, _ = convert_en_specialist.convert_file(sample_file, output_file, num_numbers=200)
    doc = CoNLL.conll2doc(input_file=output_file)
    numbers = [(sentence.sent_id, sentence.words[0].text, sentence.words[0].lemma) for sentence in doc.sentences if sentence.words[0].upos == 'NUM']
    assert len(numbers) == 200
    assert ("en_numbers-NUM-1,000", "1,000", "1000") in numbers

    # the SPECIALIST entries come first, then the numbers, each sorted by form
    upos = [sentence.words[0].upos for sentence in doc.sentences]
    assert upos[-200:] == ['NUM'] * 200
    assert 'NUM' not in upos[:-200]
    forms = [sentence.words[0].text for sentence in doc.sentences]
    assert forms[:-200] == sorted(forms[:-200])
    assert forms[-200:] == sorted(forms[-200:])
