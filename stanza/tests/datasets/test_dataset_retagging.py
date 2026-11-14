"""
Tests for which splits build_postprocess retags for each of the tag methods
"""

import argparse

import pytest

from stanza.utils.conll import CoNLL
from stanza.utils.datasets import dataset_retagging
from stanza.utils.datasets.dataset_retagging import Tags, add_retagging_args, build_postprocess

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

SHORT_NAME = "en_test"

SENTENCE = """
# text = Cats sleep
# sent_id = 1
1	Cats	cat	NOUN	NNS	Number=Plur	2	nsubj	_	_
2	sleep	sleep	VERB	VBP	_	0	root	_	_
""".lstrip()

def fake_retagger(treebank, paths, args):
    """
    Stands in for the tagger: writes every word with the tag X
    """
    def retag_dataset(tokenizer_dir, tokenizer_file, dest_dir, dest_file, short_name):
        doc = CoNLL.conll2doc(input_file=f"{tokenizer_dir}/{short_name}.{tokenizer_file}.conllu")
        for sentence in doc.sentences:
            for word in sentence.words:
                word.upos = "X"
        CoNLL.write_doc2conll(doc, f"{dest_dir}/{short_name}.{dest_file}.conllu")
    return retag_dataset

def parse_tag_method(flags):
    parser = argparse.ArgumentParser()
    add_retagging_args(parser)
    return parser.parse_args(flags)

def test_tag_method_args():
    assert parse_tag_method([]).tag_method is Tags.PREDICTED
    assert parse_tag_method(["--predicted"]).tag_method is Tags.PREDICTED
    assert parse_tag_method(["--gold"]).tag_method is Tags.GOLD
    assert parse_tag_method(["--gold_train"]).tag_method is Tags.GOLD_TRAIN

def run_postprocess(postprocess, tmp_path):
    """
    Run postprocess on each split the same way copy_conllu_treebank does

    Returns a map from split to the UPOS of the words written for it
    """
    tokenizer_dir = tmp_path / "tokenize"
    dest_dir = tmp_path / "dest"
    tokenizer_dir.mkdir()
    dest_dir.mkdir()
    upos = {}
    for split in ("train", "dev", "test"):
        (tokenizer_dir / f"{SHORT_NAME}.{split}.gold.conllu").write_text(SENTENCE, encoding="utf-8")
        postprocess(str(tokenizer_dir), f"{split}.gold", str(dest_dir), f"{split}.in", SHORT_NAME)
        doc = CoNLL.conll2doc(input_file=str(dest_dir / f"{SHORT_NAME}.{split}.in.conllu"))
        upos[split] = [word.upos for sentence in doc.sentences for word in sentence.words]
    return upos

def test_gold_builds_no_tagger(monkeypatch):
    def no_tagger(treebank, paths, args):
        raise AssertionError("--gold should not look for a tagger")
    monkeypatch.setattr(dataset_retagging, "build_retagger", no_tagger)
    assert build_postprocess("UD_English-Test", {}, parse_tag_method(["--gold"])) is None

def test_predicted_retags_all(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_retagging, "build_retagger", fake_retagger)
    postprocess = build_postprocess("UD_English-Test", {}, parse_tag_method(["--predicted"]))
    upos = run_postprocess(postprocess, tmp_path)
    assert upos == {"train": ["X", "X"], "dev": ["X", "X"], "test": ["X", "X"]}

def test_gold_train_retags_dev_test(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_retagging, "build_retagger", fake_retagger)
    postprocess = build_postprocess("UD_English-Test", {}, parse_tag_method(["--gold_train"]))
    upos = run_postprocess(postprocess, tmp_path)
    assert upos == {"train": ["NOUN", "VERB"], "dev": ["X", "X"], "test": ["X", "X"]}
