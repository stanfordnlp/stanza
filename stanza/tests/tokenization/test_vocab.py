import pytest

from stanza.models.common.vocab import UNK, PAD
from stanza.models.tokenization.vocab import Vocab

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_build():
    """
    Test that building a vocab out of a text produces the expected units and ids in the vocab
    """
    text = ["this is a test"]
    vocab = Vocab(data=text, lang="en")
    expected = {'<PAD>', '<UNK>', 't', 's', ' ', 'i', 'h', 'a', 'e'}
    assert expected == set(vocab._id2unit)
    for unit in vocab._id2unit:
        assert vocab.id2unit(vocab.unit2id(unit)) == unit


def test_append():
    text = ["this is a test"]
    vocab = Vocab(data=text, lang="en")

    assert 'z' not in vocab
    vocab.append('z')
    expected = {'<PAD>', '<UNK>', 't', 's', ' ', 'i', 'h', 'a', 'e', 'z'}
    assert expected == set(vocab._id2unit)
    for unit in vocab._id2unit:
        assert vocab.id2unit(vocab.unit2id(unit)) == unit
