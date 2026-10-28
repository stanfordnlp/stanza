"""
Tests for training the lemmatizer's seq2seq model on a secondary file such as a lexicon
"""

import glob
import itertools
import os

import pytest

from stanza.models import lemmatizer
from stanza.models.lemma.data import DataLoader
from stanza.models.lemma.trainer import Trainer
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def secondary_positions(num_train, rate):
    """
    Return the number of training batches seen before each secondary batch
    """
    secondary = itertools.count(1000)
    positions = []
    num_train_seen = 0
    for batch, is_secondary in lemmatizer.interleave_batches(range(num_train), secondary, rate):
        if is_secondary:
            assert batch >= 1000
            positions.append(num_train_seen)
        else:
            assert batch == num_train_seen
            num_train_seen += 1
    assert num_train_seen == num_train
    return positions

def test_interleave_rate():
    assert secondary_positions(25, 0.1) == [10, 20]
    assert secondary_positions(9, 0.1) == []
    assert secondary_positions(4, 0.5) == [2, 4]
    assert secondary_positions(3, 2) == [1, 1, 2, 2, 3, 3]
    assert secondary_positions(5, 0) == []

def test_interleave_rounding():
    """
    90 * 0.7 is 62.99999999999999 in floating point, but should still produce 63 batches
    """
    assert len(secondary_positions(90, 0.7)) == 63

def test_interleave_no_secondary():
    batches = list(lemmatizer.interleave_batches(range(5), None, 0.1))
    assert batches == [(x, False) for x in range(5)]

def build_conllu(entries):
    lines = []
    for form, upos, lemma in entries:
        lines.append(f"# text = {form}")
        lines.append(f"1\t{form}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_")
        lines.append("")
    return "\n".join(lines) + "\n"

TRAIN_ENTRIES = [
    ("dogs",    "NOUN", "dog"),
    ("cats",    "NOUN", "cat"),
    ("walked",  "VERB", "walk"),
    ("talking", "VERB", "talk"),
    ("bigger",  "ADJ",  "big"),
    ("the",     "DET",  "the"),
    ("quickly", "ADV",  "quickly"),
]

DEV_ENTRIES = [
    ("birds",  "NOUN", "bird"),
    ("jumped", "VERB", "jump"),
]

# é and ç only appear in the secondary file
SECONDARY_ENTRIES = [
    ("cafés",     "NOUN", "café"),
    ("façades",   "NOUN", "façade"),
    ("housewives", "NOUN", "housewife"),
]

def test_reshuffle():
    doc = CoNLL.conll2doc(input_str=build_conllu(TRAIN_ENTRIES))
    args = lemmatizer.parse_args(["--shorthand", "en_test"])
    loader = DataLoader(doc, 3, args, evaluation=False)
    before = sorted(x[-1] for batch in loader.data for x in batch)
    assert [len(batch) for batch in loader.data] == [3, 3, 1]
    for _ in range(5):
        loader.reshuffle()
        assert [len(batch) for batch in loader.data] == [3, 3, 1]
        assert sorted(x[-1] for batch in loader.data for x in batch) == before

def test_cycle_batches():
    doc = CoNLL.conll2doc(input_str=build_conllu(SECONDARY_ENTRIES))
    args = lemmatizer.parse_args(["--shorthand", "en_test"])
    loader = DataLoader(doc, 2, args, evaluation=False)
    batches = list(itertools.islice(lemmatizer.cycle_batches(loader), 6))
    words = [word for batch in batches for word in batch[-1]]
    # each pass is 3 words in 2 batches, so 6 batches is 3 complete passes
    assert len(words) == 9
    for idx in range(3):
        assert sorted(words[idx*3:idx*3+3]) == sorted(x[0] for x in SECONDARY_ENTRIES)

def train_model(tmp_path, secondary_rate):
    train_file = tmp_path / "train.conllu"
    train_file.write_text(build_conllu(TRAIN_ENTRIES), encoding="utf-8")
    dev_file = tmp_path / "dev.conllu"
    dev_file.write_text(build_conllu(DEV_ENTRIES), encoding="utf-8")
    secondary_file = tmp_path / "secondary.conllu"
    secondary_file.write_text(build_conllu(SECONDARY_ENTRIES), encoding="utf-8")
    save_dir = tmp_path / ("models_%s" % secondary_rate)

    lemmatizer.main(["--train_file", str(train_file),
                     "--eval_file", str(dev_file),
                     "--secondary_file", str(secondary_file),
                     "--secondary_rate", str(secondary_rate),
                     "--shorthand", "en_test",
                     "--save_dir", str(save_dir),
                     "--num_epoch", "2",
                     "--batch_size", "2",
                     "--hidden_dim", "8",
                     "--cpu"])
    model_files = glob.glob(str(save_dir / "*.pt"))
    assert len(model_files) == 1
    return Trainer(model_file=model_files[0], device="cpu")

def test_train_with_secondary(tmp_path):
    """
    With a nonzero rate, the char vocab includes the secondary file, and its entries are in the dictionary
    """
    trainer = train_model(tmp_path, 1.0)
    char_vocab = trainer.vocab['char']
    assert 'é' in char_vocab
    assert 'ç' in char_vocab
    assert trainer.predict_dict([("housewives", "NOUN"), ("dogs", "NOUN")]) == ["housewife", "dog"]

def test_train_dict_only_secondary(tmp_path):
    """
    With a rate of 0, the secondary file only goes in the dictionary, so its characters aren't in the vocab
    """
    trainer = train_model(tmp_path, 0)
    char_vocab = trainer.vocab['char']
    assert 'é' not in char_vocab
    assert trainer.predict_dict([("housewives", "NOUN"), ("dogs", "NOUN")]) == ["housewife", "dog"]
