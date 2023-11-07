"""
A few tests of specific operations from the Dataset
"""

import os
import pytest

from stanza.models.common.doc import *
from stanza.models import tagger
from stanza.models.pos.data import Dataset, ShuffledDataset
from stanza.utils.conll import CoNLL

from stanza.tests.pos.test_tagger import TRAIN_DATA, TRAIN_DATA_NO_XPOS, TRAIN_DATA_NO_UPOS, TRAIN_DATA_NO_FEATS

def test_basic_reading():
    """
    Test that a dataset with no xpos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_xpos():
    """
    Test that a dataset with no xpos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_XPOS)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert not data.has_xpos
    assert data.has_feats

def test_no_upos():
    """
    Test that a dataset with no upos is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_UPOS)

    data = Dataset(train_doc, args, None)
    assert not data.has_upos
    assert data.has_xpos
    assert data.has_feats

def test_no_feats():
    """
    Test that a dataset with no feats is detected by the Dataset
    """
    # empty args for building the data object
    args = tagger.parse_args(args=[])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA_NO_FEATS)

    data = Dataset(train_doc, args, None)
    assert data.has_upos
    assert data.has_xpos
    assert not data.has_feats

def test_no_augment():
    """
    Test that with no punct removing augmentation, the doc always has punct at the end
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "0.0"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    for i in range(50):
        for batch in data:
            for text in batch.text:
                assert text[-1] in (".", "!")

def test_augment():
    """
    Test that with 100% punct removing augmentation, the doc never has punct at the end
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "1.0"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    for i in range(50):
        for batch in data:
            for text in batch.text:
                assert text[-1] not in (".", "!")

def test_sometimes_augment():
    """
    Test 50% punct removing augmentation

    With this frequency, we should get a reasonable number of docs
    with a punct at the end and a reasonable without.
    """
    args = tagger.parse_args(args=["--shorthand", "en_test", "--augment_nopunct", "0.5"])

    train_doc = CoNLL.conll2doc(input_str=TRAIN_DATA)
    data = Dataset(train_doc, args, None)
    data = data.to_loader(batch_size=2)

    count_with = 0
    count_without = 0
    for i in range(50):
        for batch in data:
            for text in batch.text:
                if text[-1] in (".", "!"):
                    count_with += 1
                else:
                    count_without += 1

    # this should never happen
    # literally less than 1 in 10^20th odds
    assert count_with > 5
    assert count_without > 5


NO_XPOS_TEMPLATE = """
# text = Noxpos {indexp}
# sent_id = {index}
1	Noxpos	noxpos	NOUN	_	Number=Sing	0	root	_	start_char=0|end_char=8|ner=O
2	{indexp}	{indexp}	NUM	_	NumForm=Digit|NumType=Card	1	dep	_	start_char=9|end_char=10|ner=S-CARDINAL
""".strip()

YES_XPOS_TEMPLATE = """
# text = Yesxpos {indexp}
# sent_id = {index}
1	Yesxpos	yesxpos	NOUN	NN	Number=Sing	0	root	_	start_char=0|end_char=8|ner=O
2	{indexp}	{indexp}	NUM	CD	NumForm=Digit|NumType=Card	1	dep	_	start_char=9|end_char=10|ner=S-CARDINAL
""".strip()

def test_shuffle(tmp_path):
    args = tagger.parse_args(args=["--batch_size", "10", "--shorthand", "en_test", "--augment_nopunct", "0.0"])

    # 100 looked nice but was actually a 1/1000000 chance of the test failing
    # so let's crank it up to 1000 and make it 1/10^58
    no_xpos = [NO_XPOS_TEMPLATE.format(index=idx, indexp=idx+1) for idx in range(1000)]
    no_doc = CoNLL.conll2doc(input_str="\n\n".join(no_xpos))
    no_data = Dataset(no_doc, args, None)

    yes_xpos = [YES_XPOS_TEMPLATE.format(index=idx, indexp=idx+101) for idx in range(1000)]
    yes_doc = CoNLL.conll2doc(input_str="\n\n".join(yes_xpos))
    yes_data = Dataset(yes_doc, args, None)

    shuffled = ShuffledDataset([no_data, yes_data], 10)

    assert sum(1 for _ in shuffled) == 200

    num_with = 0
    num_without = 0
    for batch in shuffled:
        if batch.xpos is not None:
            num_with += 1
        else:
            num_without += 1
        # at the halfway point of the iteration, there should be at
        # least one in each category
        # for example, if we had forgotten to shuffle, this assertion would fail
        if num_with + num_without == 100:
            assert num_with > 1
            assert num_without > 1

    assert num_with == 100
    assert num_without == 100
