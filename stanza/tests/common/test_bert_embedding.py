import pytest
import torch

from stanza.models.common.bert_embedding import load_bert, extract_bert_embeddings

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

BERT_MODEL = "hf-internal-testing/tiny-bert"

@pytest.fixture(scope="module")
def tiny_bert():
    m, t = load_bert(BERT_MODEL)
    return m, t

def test_load_bert(tiny_bert):
    """
    Empty method that just tests loading the bert
    """
    m, t = tiny_bert

def test_run_bert(tiny_bert):
    m, t = tiny_bert
    device = next(m.parameters()).device
    extract_bert_embeddings(BERT_MODEL, t, m, [["This", "is", "a", "test"]], device, True)

def test_run_bert_empty_word(tiny_bert):
    m, t = tiny_bert
    device = next(m.parameters()).device
    foo = extract_bert_embeddings(BERT_MODEL, t, m, [["This", "is", "-", "a", "test"]], device, True)
    bar = extract_bert_embeddings(BERT_MODEL, t, m, [["This", "is", "", "a", "test"]], device, True)

    assert len(foo) == 1
    assert torch.allclose(foo[0], bar[0])
