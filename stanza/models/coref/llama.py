"""Functions related to BERT or similar models"""

import logging
from typing import List, Tuple
from peft import prepare_model_for_kbit_training


import numpy as np                                 # type: ignore
from transformers import LlamaModel, AutoTokenizer  # type: ignore

from stanza.models.coref.config import Config
from stanza.models.coref.const import Doc

import torch.nn as nn

logger = logging.getLogger('stanza')

def load_llama(config: Config):
    logger.debug(f"Loading {config.llama_model}...")

    base_llama_name = config.llama_model.split("/")[-1]
    tokenizer_kwargs = config.tokenizer_kwargs.get(base_llama_name, {})
    if tokenizer_kwargs:
        logger.debug(f"Using tokenizer kwargs: {tokenizer_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(config.llama_model, **tokenizer_kwargs)
    model = prepare_model_for_kbit_training(LlamaModel.from_pretrained(config.llama_model, load_in_8bit=True))
    # .to(config.device)

    logger.debug("Llama successfully loaded.")

    return model, tokenizer



