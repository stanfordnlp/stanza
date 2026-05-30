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

class TestLoadBert:
    def test_loads_model_and_tokenizer(self):
        model, tokenizer = load_bert(BERT_MODEL)
        assert model is not None
        assert tokenizer is not None

    def test_none_model_name_returns_none(self):
        model, tokenizer = load_bert(None)
        assert model is None
        assert tokenizer is None

    # Note: some decoder-style models (LLaMA, GPT) set use_cache=False
    # automatically when gradient_checkpointing_enable() is called, because
    # their KV cache is incompatible with gradient checkpointing during
    # training.  Encoder-only models like BERT and ELECTRA don't use the KV
    # cache during the forward pass, so transformers leaves use_cache
    # unchanged for them.  There is nothing to test or enforce here on the
    # load_bert side -- the behaviour is entirely within transformers.
    def test_gradient_checkpointing_disabled_by_default(self):
        from stanza.models.common.bert_embedding import load_bert
        model, _ = load_bert(BERT_MODEL)
        assert not model.is_gradient_checkpointing

    def test_gradient_checkpointing_enabled_when_requested(self):
        from stanza.models.common.bert_embedding import load_bert
        model, _ = load_bert(BERT_MODEL, enable_gradient_checkpointing=True)
        assert model.is_gradient_checkpointing

    def test_gradient_checkpointing_false_explicit(self):
        from stanza.models.common.bert_embedding import load_bert
        model, _ = load_bert(BERT_MODEL, enable_gradient_checkpointing=False)
        assert not model.is_gradient_checkpointing
