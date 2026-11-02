"""
Tests for the exact dictionary of capitalized and uppercase NOUNs, and how it is prepared and used
"""

import os

import pytest

from stanza.models.lemma.trainer import Trainer
from stanza.utils.conll import CoNLL
from stanza.utils.datasets import prepare_lemma_treebank
from stanza.utils.datasets.lemma import noun_case_variants
from stanza.utils.training import run_lemma

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

TRAIN_TRIPLES = [
    ("beer",    "NOUN",  "beer"),
    ("beers",   "NOUN",  "beer"),
    ("BEER",    "PROPN", "Beer"),
    ("traders", "NOUN",  "trader"),
    ("Traders", "PROPN", "Traders"),
    # already a NOUN in this case, so it does not get a variant
    ("Model",   "NOUN",  "model"),
    ("model",   "NOUN",  "model"),
    # the most frequent lemma wins
    ("data",    "NOUN",  "data"),
    ("data",    "NOUN",  "data"),
    ("data",    "NOUN",  "datum"),
    # single letters keep their case
    ("k",       "NOUN",  "k"),
    # only NOUNs
    ("walked",  "VERB",  "walk"),
    # blank lemmas are skipped
    ("thing",   "NOUN",  "_"),
]

EXPECTED = [
    ("BEER",    "NOUN", "beer"),
    ("BEERS",   "NOUN", "beer"),
    ("Beer",    "NOUN", "beer"),
    ("Beers",   "NOUN", "beer"),
    ("DATA",    "NOUN", "data"),
    ("Data",    "NOUN", "data"),
    ("MODEL",   "NOUN", "model"),
    ("TRADERS", "NOUN", "trader"),
    ("Traders", "NOUN", "trader"),
]

def build_conllu(triples):
    lines = []
    for form, upos, lemma in triples:
        lines.append(f"# text = {form}")
        lines.append(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_")
        lines.append("")
    return "\n".join(lines) + "\n"

def test_noun_case_variants():
    assert noun_case_variants.noun_case_variants(TRAIN_TRIPLES) == EXPECTED

def test_build_exact_dictionary(tmp_path):
    train_file = tmp_path / "train.conllu"
    train_file.write_text(build_conllu(TRAIN_TRIPLES), encoding="utf-8")
    output_file = str(tmp_path / "exact.conllu")
    noun_case_variants.build_exact_dictionary(str(train_file), output_file)
    doc = CoNLL.conll2doc(input_file=output_file)
    assert [(word.text, word.upos, word.lemma) for sentence in doc.sentences for word in sentence.words] == EXPECTED
    assert doc.sentences[0].sent_id == "noun_case-NOUN-BEER"

def build_paths(tmp_path):
    return {"LEMMA_DATA_DIR": str(tmp_path / "data" / "lemma")}

def test_prepare_exact_dictionary(tmp_path):
    """
    The exact dictionary is built from the prepared training file, found by run_lemma, and removed if it is turned off
    """
    paths = build_paths(tmp_path)
    lemma_dir = tmp_path / "data" / "lemma"
    lemma_dir.mkdir(parents=True)
    (lemma_dir / "en_combined.train.in.conllu").write_text(build_conllu(TRAIN_TRIPLES), encoding="utf-8")
    assert run_lemma.find_exact_dict_file(paths, "en_combined") is None

    exact_file = prepare_lemma_treebank.prepare_exact_dictionary(paths, "en_combined")
    assert exact_file == str(lemma_dir / "en_combined.exact_dict.conllu")
    assert len(CoNLL.conll2doc(input_file=exact_file).sentences) == len(EXPECTED)
    assert run_lemma.find_exact_dict_file(paths, "en_combined") == exact_file

    assert prepare_lemma_treebank.prepare_exact_dictionary(paths, "en_combined", use_exact_dictionary=False) is None
    assert not os.path.exists(exact_file)
    assert run_lemma.find_exact_dict_file(paths, "en_combined") is None

    # datasets without an exact dictionary don't get one
    (lemma_dir / "en_ewt.train.in.conllu").write_text(build_conllu(TRAIN_TRIPLES), encoding="utf-8")
    assert prepare_lemma_treebank.prepare_exact_dictionary(paths, "en_ewt") is None
    assert not os.path.exists(lemma_dir / "en_ewt.exact_dict.conllu")

def test_add_exact_entries():
    """
    Exact entries only change their own (word, pos), not the same word with another pos
    """
    trainer = Trainer(args={'dict_only': True}, device='cpu')
    trainer.train_dict(TRAIN_TRIPLES)
    queries = [("Traders", "NOUN"), ("Traders", "PROPN"), ("Traders", "X"), ("Beer", "NOUN"), ("Beer", "PROPN")]
    assert trainer.predict_dict(queries) == ["Traders", "Traders", "Traders", "Beer", "Beer"]
    assert trainer.skip_seq2seq(queries) == [True, True, True, False, False]

    trainer.add_exact_entries(EXPECTED)
    assert trainer.predict_dict(queries) == ["trader", "Traders", "Traders", "beer", "Beer"]
    assert trainer.skip_seq2seq(queries) == [True, True, True, True, False]

def test_add_exact_entries_replaces():
    """
    An exact entry replaces the training data for that (word, pos), with the most frequent lemma winning
    """
    trainer = Trainer(args={'dict_only': True}, device='cpu')
    trainer.train_dict([("leaves", "NOUN", "leaf"), ("leaves", "VERB", "leave"), ("leaves", "VERB", "leave")])
    assert trainer.predict_dict([("leaves", "NOUN"), ("leaves", "VERB")]) == ["leaf", "leave"]
    trainer.add_exact_entries([("leaves", "NOUN", "leave"), ("leaves", "NOUN", "leafs"), ("leaves", "NOUN", "leave")])
    assert trainer.predict_dict([("leaves", "NOUN"), ("leaves", "VERB")]) == ["leave", "leave"]
