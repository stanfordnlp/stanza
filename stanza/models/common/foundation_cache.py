"""
Keeps BERT, charlm, word embedings in a cache to save memory
"""

from __future__ import annotations

import logging
from os import PathLike
import threading
from typing import Dict, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union

from stanza.models.common import bert_embedding
from stanza.models.common.char_model import CharacterLanguageModel
from stanza.models.common.pretrain import Pretrain

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger('stanza')

ModelPath = Union[str, PathLike[str]]
BertComponents = Union[
    Tuple["PreTrainedModel", "PreTrainedTokenizerBase"],
    Tuple[None, None],
]
BertComponentsWithPeft = Union[
    Tuple["PreTrainedModel", "PreTrainedTokenizerBase", Optional[str]],
    Tuple[None, None, Optional[str]],
]


class BertRecord(NamedTuple):
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizerBase"
    peft_ids: Dict[str, int]


def load_pretrain(filename: Optional[ModelPath], foundation_cache: Optional[FoundationCache] = None) -> Optional[Pretrain]:
    if not filename:
        return None

    if foundation_cache is not None:
        return foundation_cache.load_pretrain(filename)

    logger.debug("Loading pretrain from %s", filename)
    return Pretrain(filename)


def load_charlm(
    charlm_file: Optional[ModelPath],
    foundation_cache: Optional[FoundationCache] = None,
    finetune: bool = False,
) -> Optional[CharacterLanguageModel]:
    if not charlm_file:
        return None

    if finetune:
        # can't use the cache in the case of a model which will be finetuned
        # and the numbers will be different for other users of the model
        return CharacterLanguageModel.load(charlm_file, finetune=True)

    if foundation_cache is not None:
        return foundation_cache.load_charlm(charlm_file)

    logger.debug("Loading charlm from %s", charlm_file)
    return CharacterLanguageModel.load(charlm_file, finetune=False)


def load_bert(
    model_name: Optional[str],
    foundation_cache: Optional[FoundationCache] = None,
    local_files_only: Optional[bool] = None,
    enable_gradient_checkpointing: bool = False,
) -> BertComponents:
    """
    Load a bert, possibly using a foundation cache, ignoring the cache if None
    """
    if foundation_cache is None:
        if local_files_only is None:
            local_files_only = False
        return bert_embedding.load_bert(model_name, local_files_only=local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)
    else:
        return foundation_cache.load_bert(model_name, local_files_only=local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)


def load_bert_with_peft(
    model_name: Optional[str],
    peft_name: Optional[str],
    foundation_cache: Optional[FoundationCache] = None,
    local_files_only: Optional[bool] = None,
    enable_gradient_checkpointing: bool = False,
) -> BertComponentsWithPeft:
    if foundation_cache is None:
        if local_files_only is None:
            local_files_only = False
        m, t = bert_embedding.load_bert(model_name, local_files_only=local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)
        return m, t, peft_name
    return foundation_cache.load_bert_with_peft(model_name, peft_name, local_files_only=local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)


class FoundationCache:
    bert: Dict[str, BertRecord]
    charlms: Dict[ModelPath, CharacterLanguageModel]
    pretrains: Dict[ModelPath, Pretrain]
    local_files_only: bool

    def __init__(self, other: Optional[FoundationCache] = None, local_files_only: bool = False) -> None:
        if other is None:
            self.bert = {}
            self.charlms = {}
            self.pretrains = {}
            # future proof the module by using a lock for the glorious day
            # when the GIL is finally gone
            self.lock = threading.Lock()
        else:
            self.bert = other.bert
            self.charlms = other.charlms
            self.pretrains = other.pretrains
            self.lock = other.lock
        self.local_files_only = local_files_only

    def load_bert(
        self,
        transformer_name: Optional[str],
        local_files_only: Optional[bool] = None,
        enable_gradient_checkpointing: bool = False,
    ) -> BertComponents:
        components = self.load_bert_with_peft(
            transformer_name,
            None,
            local_files_only=local_files_only,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
        if components[0] is None:
            return None, None
        return components[0], components[1]

    def load_bert_with_peft(
        self,
        transformer_name: Optional[str],
        peft_name: Optional[str],
        local_files_only: Optional[bool] = None,
        enable_gradient_checkpointing: bool = False,
    ) -> BertComponentsWithPeft:
        """
        Load a transformer only once

        Uses a lock for thread safety
        """
        if not transformer_name:
            return None, None, None
        with self.lock:
            if transformer_name not in self.bert:
                if local_files_only is None:
                    local_files_only = self.local_files_only
                model, tokenizer = bert_embedding.load_bert(transformer_name, local_files_only=local_files_only)
                assert model is not None
                assert tokenizer is not None
                self.bert[transformer_name] = BertRecord(model, tokenizer, {})

            else:
                logger.debug("Reusing bert %s", transformer_name)

            bert_record = self.bert[transformer_name]
            if enable_gradient_checkpointing:
                # an issue with reusing existing bert models
                # and enabling gradient checkpointing is that
                # existing peft wrappers won't properly train
                # hopefully that doesn't come up too often
                # one way we try to avoid that is by only doing the
                # enabling in the training routines, which generally
                # only do one at a time
                bert_record.model.gradient_checkpointing_enable()
                # not all versions will enable input grads
                # in which case some versions of peft might think the model
                # doesn't need grads and winds up not finetuning anything
                # enabling it ourselves here prevents that from happening
                bert_record.model.enable_input_require_grads()
            if not peft_name:
                return bert_record.model, bert_record.tokenizer, None
            if peft_name not in bert_record.peft_ids:
                bert_record.peft_ids[peft_name] = 0
            else:
                bert_record.peft_ids[peft_name] = bert_record.peft_ids[peft_name] + 1
            peft_name = "%s_%d" % (peft_name, bert_record.peft_ids[peft_name])
            return bert_record.model, bert_record.tokenizer, peft_name

    def load_charlm(self, filename: Optional[ModelPath]) -> Optional[CharacterLanguageModel]:
        if not filename:
            return None

        with self.lock:
            if filename not in self.charlms:
                logger.debug("Loading charlm from %s", filename)
                self.charlms[filename] = CharacterLanguageModel.load(filename, finetune=False)
            else:
                logger.debug("Reusing charlm from %s", filename)

            return self.charlms[filename]

    def load_pretrain(self, filename: Optional[ModelPath]) -> Optional[Pretrain]:
        """
        Load a pretrained word embedding only once

        Uses a lock for thread safety
        """
        if filename is None:
            return None
        with self.lock:
            if filename not in self.pretrains:
                logger.debug("Loading pretrain %s", filename)
                self.pretrains[filename] = Pretrain(filename)
            else:
                logger.debug("Reusing pretrain %s", filename)

            return self.pretrains[filename]


class NoTransformerFoundationCache(FoundationCache):
    """
    Uses the underlying FoundationCache, but hiding the transformer.

    Useful for when loading a downstream model such as POS which has a
    finetuned transformer, and we don't want the transformer reused
    since it will then have the finetuned weights for other models
    which don't want them
    """
    def load_bert(
        self,
        transformer_name: Optional[str],
        local_files_only: Optional[bool] = None,
        enable_gradient_checkpointing: bool = False,
    ) -> BertComponents:
        return load_bert(transformer_name, local_files_only=self.local_files_only if local_files_only is None else local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)

    def load_bert_with_peft(
        self,
        transformer_name: Optional[str],
        peft_name: Optional[str],
        local_files_only: Optional[bool] = None,
        enable_gradient_checkpointing: bool = False,
    ) -> BertComponentsWithPeft:
        return load_bert_with_peft(transformer_name, peft_name, local_files_only=self.local_files_only if local_files_only is None else local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)
