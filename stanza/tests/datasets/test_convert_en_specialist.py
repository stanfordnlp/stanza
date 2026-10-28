"""
Tests for converting the SPECIALIST Lexicon inflection table to a lemma dictionary
"""

import os
import time

import pytest

from stanza.utils.conll import CoNLL
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
    convert_en_specialist.convert_file(sample_file, output_file)
    doc = CoNLL.conll2doc(input_file=output_file)
    assert len(doc.sentences) == len(EXPECTED)
    words = {(word.text, word.upos): word.lemma for sentence in doc.sentences for word in sentence.words}
    assert words == EXPECTED

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

def test_run_lemma_rebuild(sample_file, tmp_path):
    """
    The run_lemma hook raises if inflVars.data is missing, and rebuilds the dictionary when inflVars.data is newer
    """
    paths = {"STANZA_EXTERN_DIR": str(tmp_path / "extern")}
    with pytest.raises(FileNotFoundError):
        run_lemma.maybe_rebuild_external_dictionary(paths, "en_combined")

    specialist_dir = tmp_path / "extern" / "english" / "specialist"
    specialist_dir.mkdir(parents=True)
    infl_vars = specialist_dir / "inflVars.data"
    infl_vars.write_text(SAMPLE, encoding="utf-8")

    lexicon = run_lemma.maybe_rebuild_external_dictionary(paths, "en_combined")
    assert lexicon == str(specialist_dir / "specialist_lemmas.conllu")
    assert os.path.exists(lexicon)
    assert len(CoNLL.conll2doc(input_file=lexicon).sentences) == len(EXPECTED)

    # an up to date lexicon is reused as is
    with open(lexicon, "a", encoding="utf-8") as fout:
        fout.write("# text = marker\n1\tmarker\tmarker\tNOUN\t_\t_\t0\troot\t_\t_\n\n")
    assert run_lemma.maybe_rebuild_external_dictionary(paths, "en_combined") == lexicon
    assert len(CoNLL.conll2doc(input_file=lexicon).sentences) == len(EXPECTED) + 1

    # a newer inflVars.data causes a rebuild
    future = time.time() + 100
    os.utime(infl_vars, (future, future))
    assert run_lemma.maybe_rebuild_external_dictionary(paths, "en_combined") == lexicon
    assert len(CoNLL.conll2doc(input_file=lexicon).sentences) == len(EXPECTED)

    assert run_lemma.maybe_rebuild_external_dictionary(paths, "en_ewt") is None
