"""
Runs a few tests on the split_wikiner file
"""

import os
import tempfile

import pytest

from stanza.utils.datasets.ner import split_wikiner

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# two sentences from the Italian dataset, split into many pieces
# to test the splitting functionality
FBK_SAMPLE = """
Il	O
Papa	O
si	O
aggrava	O

Le	O
condizioni	O
di	O

Papa	O
Giovanni	PER
Paolo	PER
II	PER
si	O

sono	O
aggravate	O
in	O
il	O
corso	O

di	O
la	O
giornata	O
di	O
giovedì	O
.	O

Il	O
portavoce	O
Navarro	PER
Valls	PER

ha	O
dichiarato	O
che	O

il	O
Santo	O
Padre	O

in	O
la	O
giornata	O

di	O
oggi	O
è	O
stato	O

colpito	O
da	O
una	O
affezione	O

altamente	O
febbrile	O
provocata	O
da	O
una	O

infezione	O
documentata	O

di	O
le	O
vie	O
urinarie	O
.	O

A	O
il	O
momento	O

non	O
è	O
previsto	O
il	O
ricovero	O

a	O
il	O
Policlinico	LOC
Gemelli	LOC
,	O

come	O
ha	O
precisato	O
il	O

responsabile	O
di	O
il	O
dipartimento	O

di	O
emergenza	O
professor	O
Rodolfo	PER
Proietti	PER
.	O
"""


def test_read_sentences():
    with tempfile.TemporaryDirectory() as tempdir:
        raw_filename = os.path.join(tempdir, "raw.tsv")
        with open(raw_filename, "w") as fout:
            fout.write(FBK_SAMPLE)

        sentences = split_wikiner.read_sentences(raw_filename, "utf-8")
        assert len(sentences) == 20
        text = [["\t".join(word) for word in sent] for sent in sentences]
        text = ["\n".join(sent) for sent in text]
        text = "\n\n".join(text)
        assert FBK_SAMPLE.strip() == text

def test_write_sentences():
    with tempfile.TemporaryDirectory() as tempdir:
        raw_filename = os.path.join(tempdir, "raw.tsv")
        with open(raw_filename, "w") as fout:
            fout.write(FBK_SAMPLE)

        sentences = split_wikiner.read_sentences(raw_filename, "utf-8")
        copy_filename = os.path.join(tempdir, "copy.tsv")
        split_wikiner.write_sentences_to_file(sentences, copy_filename)

        sent2 = split_wikiner.read_sentences(raw_filename, "utf-8")
        assert sent2 == sentences

def run_split_wikiner(expected_train=14, expected_dev=3, expected_test=3, **kwargs):
    """
    Runs a test using various parameters to check the results of the splitting process
    """
    with tempfile.TemporaryDirectory() as indir:
        raw_filename = os.path.join(indir, "raw.tsv")
        with open(raw_filename, "w") as fout:
            fout.write(FBK_SAMPLE)

        with tempfile.TemporaryDirectory() as outdir:
            split_wikiner.split_wikiner(outdir, raw_filename, **kwargs)

            train_file = os.path.join(outdir, "it_fbk.train.bio")
            dev_file = os.path.join(outdir, "it_fbk.dev.bio")
            test_file = os.path.join(outdir, "it_fbk.test.bio")

            assert os.path.exists(train_file)
            assert os.path.exists(dev_file)
            if kwargs["test_section"]:
                assert os.path.exists(test_file)
            else:
                assert not os.path.exists(test_file)

            train_sent = split_wikiner.read_sentences(train_file, "utf-8")
            dev_sent = split_wikiner.read_sentences(dev_file, "utf-8")
            assert len(train_sent) == expected_train
            assert len(dev_sent) == expected_dev
            if kwargs["test_section"]:
                test_sent = split_wikiner.read_sentences(test_file, "utf-8")
                assert len(test_sent) == expected_test
            else:
                test_sent = []

            if kwargs["shuffle"]:
                orig_sents = sorted(split_wikiner.read_sentences(raw_filename, "utf-8"))
                split_sents = sorted(train_sent + dev_sent + test_sent)
            else:
                orig_sents = split_wikiner.read_sentences(raw_filename, "utf-8")
                split_sents = train_sent + dev_sent + test_sent
            assert orig_sents == split_sents

def test_no_shuffle_split():
    run_split_wikiner(prefix="it_fbk", shuffle=False, test_section=True)

def test_shuffle_split():
    run_split_wikiner(prefix="it_fbk", shuffle=True, test_section=True)

def test_resize():
    run_split_wikiner(expected_train=12, expected_dev=2, expected_test=6, train_fraction=0.6, dev_fraction=0.1, prefix="it_fbk", shuffle=True, test_section=True)

def test_no_test_split():
    run_split_wikiner(expected_train=17, train_fraction=0.85, prefix="it_fbk", shuffle=False, test_section=False)

