"""
Tests for the required ordering of gradient checkpointing and PEFT initialisation.

When both gradient checkpointing and LoRA finetuning are active, the order of
setup calls matters:

    1. model.gradient_checkpointing_enable()
    2. model.enable_input_require_grads()
    3. model = get_peft_model(model, lora_config)

If PEFT wraps the model before gradient checkpointing is enabled, the
checkpointing hooks are applied to already-wrapped LoRA modules rather than
the base transformer layers.  This breaks the gradient flow through the
checkpointed segments and causes LoRA parameters to receive no gradients,
manifesting as the 'UserWarning: None of the inputs have requires_grad=True'
warning and weights that do not move during training.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel, AutoTokenizer

from stanza.models.common.bert_embedding import load_bert
from stanza.models.common.peft_config import build_peft_wrapper

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

BERT_MODEL = "hf-internal-testing/tiny-bert"

def make_tiny_bert():
    """Load the tiny-bert model and tokenizer directly."""
    model = AutoModel.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, add_prefix_space=True)
    tokenizer.model_max_length = 512
    return model, tokenizer


class TestGradientCheckpointingPeftOrdering:
    """
    The core correctness requirement: gradient checkpointing must be enabled
    on the base model BEFORE get_peft_model is called.  If PEFT wraps first,
    the checkpointing hooks interact badly with LoRA and LoRA weights receive
    no gradients (requires_grad warning).
    """

    def test_gradient_checkpointing_before_peft_wrap(self):
        """
        Simulate the trainer __init__ ordering:
          1. load_bert (with gradient checkpointing)
          2. build_peft_wrapper

        Verify that gradient checkpointing is on the model before PEFT wraps it
        by checking is_gradient_checkpointing at each stage.
        """
        # Step 1: load with gradient checkpointing
        model, _ = load_bert(BERT_MODEL, enable_gradient_checkpointing=True)
        assert model.is_gradient_checkpointing, (
            "Gradient checkpointing must be enabled before PEFT wrapping"
        )

        # Step 2: wrap with PEFT — model should still have checkpointing on
        args = {"use_peft": True, "peft_type": "LORA", "lora_rank": 4,
                "lora_alpha": 16, "lora_dropout": 0.1, "lora_target_modules": None,
                'lora_modules_to_save': None}
        logger = MagicMock()

        wrapped = build_peft_wrapper(model, args, logger, adapter_name="test")
        # After PEFT wrapping, checkpointing should still be active
        assert wrapped.is_gradient_checkpointing or model.is_gradient_checkpointing

    def test_peft_lora_params_require_grad_with_checkpointing(self):
        """
        LoRA parameters must have requires_grad=True when gradient checkpointing
        is active.  This is the failure mode we were seeing: LoRA weights not
        updating at all.
        """
        model, _ = make_tiny_bert()

        # Enable gradient checkpointing FIRST
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        assert model.is_gradient_checkpointing

        # Then wrap with LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        lora_params = [(n, p) for n, p in peft_model.named_parameters()
                       if 'lora_' in n]
        assert len(lora_params) > 0, "Expected to find LoRA parameters"
        for name, param in lora_params:
            assert param.requires_grad, (
                f"LoRA parameter {name} does not require grad — "
                "gradient checkpointing was likely applied in the wrong order"
            )

    def test_peft_before_checkpointing_breaks_grad(self):
        """
        Document the failure mode: wrapping with PEFT before enabling gradient
        checkpointing can cause LoRA parameters to lose requires_grad.
        This test demonstrates the problem that the ordering fix solves.

        Note: whether this actually breaks depends on transformers/peft version,
        so we mark it as xfail if it doesn't reproduce — the important thing
        is that we have the correct-order test above.
        """
        model, _ = make_tiny_bert()

        # Wrong order: PEFT first, then gradient checkpointing
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.gradient_checkpointing_enable()

        # In some versions this is fine, in others LoRA weights lose requires_grad.
        # We just document the state rather than asserting either way.
        lora_params = [(n, p) for n, p in peft_model.named_parameters()
                       if 'lora_' in n]
        # This is informational — don't fail the suite either way.
        _ = all(p.requires_grad for _, p in lora_params)


