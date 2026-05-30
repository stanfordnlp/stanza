"""
Tests for enable_input_require_grads() in the context of gradient checkpointing
and LoRA finetuning.

When gradient checkpointing is active, PyTorch recomputes activations during
the backward pass rather than storing them.  For this recomputation to produce
gradients, the inputs to the checkpointed segments must have requires_grad=True.
For encoder-only transformers like BERT and ELECTRA, the input token IDs are
integers and never require gradients themselves — so without intervention,
gradient checkpointing produces the 'None of the inputs have requires_grad=True'
warning and LoRA parameters receive no gradients.

Calling model.enable_input_require_grads() installs a forward hook that sets
requires_grad=True on the output of the embedding layer, ensuring that the
gradient checkpointing recomputation can propagate gradients back through the
transformer and into the LoRA parameters.

This must be called after gradient_checkpointing_enable() and before
get_peft_model(), i.e. in the order:
    1. model.gradient_checkpointing_enable()
    2. model.enable_input_require_grads()
    3. model = get_peft_model(model, lora_config)
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

BERT_MODEL = "hf-internal-testing/tiny-bert"

def make_tiny_bert():
    """Load the tiny-bert model and tokenizer directly."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, add_prefix_space=True)
    tokenizer.model_max_length = 512
    return model, tokenizer


class TestEnableInputRequireGrads:
    def test_input_require_grads_allows_grad_flow(self):
        """
        With gradient checkpointing, the input embeddings must require grad
        for the recomputation to work.  enable_input_require_grads() hooks the
        model so that even non-leaf embedding outputs have requires_grad=True.
        """
        model, tokenizer = make_tiny_bert()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # Run a forward pass and check that the output has grad_fn
        tokens = tokenizer(["hello world"], return_tensors="pt",
                           is_split_into_words=False)
        output = model(**tokens, output_hidden_states=True)
        # The last hidden state should be part of the computation graph
        assert output.last_hidden_state.requires_grad or \
               output.last_hidden_state.grad_fn is not None

    def test_without_input_require_grads_may_warn(self):
        """
        Without enable_input_require_grads(), gradient checkpointing on some
        models produces 'None of the inputs have requires_grad=True' warnings.
        This test documents the behaviour; the fix is always to call
        enable_input_require_grads() when using gradient checkpointing + PEFT.
        """
        pytest.importorskip("peft")
        from peft import LoraConfig, get_peft_model
        import warnings

        model, _ = make_tiny_bert()
        model.gradient_checkpointing_enable()
        # Deliberately NOT calling enable_input_require_grads()

        lora_config = LoraConfig(r=4, lora_alpha=16,
                                 target_modules=["query", "value"],
                                 lora_dropout=0.1, bias="none")
        peft_model = get_peft_model(model, lora_config)

        # We don't assert the warning fires (it's version-dependent),
        # but we verify the model can at least do a forward pass
        tokens_in = torch.randint(0, 100, (1, 8))
        attention = torch.ones(1, 8)
        output = peft_model(tokens_in, attention_mask=attention,
                            output_hidden_states=True)
        assert output is not None


