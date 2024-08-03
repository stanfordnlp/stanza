""" Describes Config, a simple namespace for config values.

For description of all config values, refer to config.toml.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """ Contains values needed to set up the coreference model. """
    section: str

    # TODO: can either eliminate data_dir or use it for the train/dev/test data
    data_dir: str
    save_dir: str
    save_name: str

    train_data: str
    dev_data: str
    test_data: str

    device: str

    bert_model: str
    bert_window_size: int

    embedding_size: int
    sp_embedding_size: int
    a_scoring_batch_size: int
    hidden_size: int
    n_hidden_layers: int

    max_span_len: int

    rough_k: int

    lora: bool
    lora_alpha: int
    lora_rank: int
    lora_dropout: float

    full_pairwise: bool

    lora_target_modules: List[str]
    lora_modules_to_save: List[str]

    clusters_starts_are_singletons: bool
    bert_finetune: bool
    dropout_rate: float
    learning_rate: float
    bert_learning_rate: float
    # we find that setting this to a small but non-zero number
    # makes the model less likely to forget how to do anything
    bert_finetune_begin_epoch: float
    train_epochs: int
    bce_loss_weight: float

    tokenizer_kwargs: Dict[str, dict]
    conll_log_dir: str

    save_each_checkpoint: bool
    log_norms: bool
    singletons: bool
    
