"""
Processor for performing part-of-speech tagging
"""

from __future__ import annotations

import os
from typing import (
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    TypedDict,
    Union,
    runtime_checkable,
)

import torch

from stanza.models.common import doc
from stanza.models.common.doc import Document
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.utils import unsort
from stanza.models.common.vocab import BaseVocab, VOCAB_PREFIX, CompositeVocab
from stanza.models.pos.data import Dataset
from stanza.models.pos.trainer import Trainer
from stanza.models.pos.vocab import MultiVocab
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
from stanza.utils.get_tqdm import get_tqdm

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline

tqdm = get_tqdm()

_ModelPath = Union[str, os.PathLike[str]]
_CompositeKey = Union[int, str]
_POSTag = tuple[str, str, str]
_SentencePOSTags = list[_POSTag]


class _POSOptionalModelConfig(TypedDict, total=False):
    pretrain_path: Optional[_ModelPath]
    forward_charlm_path: Optional[_ModelPath]
    backward_charlm_path: Optional[_ModelPath]
    tqdm: bool


class _POSModelConfig(_POSOptionalModelConfig):
    model_path: _ModelPath


@runtime_checkable
class _StringVocabulary(Protocol):
    @property
    def _unit2id(self) -> Mapping[str, int]:
        ...


@runtime_checkable
class _CompositeVocabulary(Protocol):
    @property
    def _unit2id(self) -> Mapping[_CompositeKey, Mapping[str, int]]:
        ...


@runtime_checkable
class _RawPOSBatch(Protocol):
    words: torch.Tensor
    words_mask: torch.Tensor
    wordchars: torch.Tensor
    wordchars_mask: torch.Tensor
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    ufeats: Optional[torch.Tensor]
    pretrained: torch.Tensor
    orig_idx: Sequence[int]
    word_orig_idx: Sequence[int]
    lens: Sequence[torch.Tensor]
    word_lens: Sequence[int]
    text: Sequence[Sequence[str]]
    idx: Sequence[int]


class _POSBatch(NamedTuple):
    words: torch.Tensor
    words_mask: torch.Tensor
    wordchars: torch.Tensor
    wordchars_mask: torch.Tensor
    upos: Optional[torch.Tensor]
    xpos: Optional[torch.Tensor]
    ufeats: Optional[torch.Tensor]
    pretrained: torch.Tensor
    orig_idx: Sequence[int]
    word_orig_idx: Sequence[int]
    lens: Sequence[torch.Tensor]
    word_lens: Sequence[int]
    text: Sequence[Sequence[str]]
    idx: Sequence[int]


@runtime_checkable
class _POSTrainer(Protocol):
    def predict(
            self,
            batch: _POSBatch,
            unsort: bool = True,
        ) -> Sequence[Sequence[Sequence[str]]]:
        ...


def _normalize_batch(raw_batch: _RawPOSBatch) -> _POSBatch:
    tensors = (
        raw_batch.words,
        raw_batch.words_mask,
        raw_batch.wordchars,
        raw_batch.wordchars_mask,
        raw_batch.pretrained,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("POS batches must contain tensors")
    optional_tensors = (
        raw_batch.upos,
        raw_batch.xpos,
        raw_batch.ufeats,
    )
    if not all(
            value is None or isinstance(value, torch.Tensor)
            for value in optional_tensors
        ):
        raise TypeError("POS tag targets must be tensors or None")
    if not all(isinstance(value, int) for value in raw_batch.orig_idx):
        raise TypeError("POS original sentence indices must be integers")
    if not all(isinstance(value, int) for value in raw_batch.word_orig_idx):
        raise TypeError("POS original word indices must be integers")
    if not all(isinstance(value, torch.Tensor) for value in raw_batch.lens):
        raise TypeError("POS sentence lengths must be tensors")
    if not all(isinstance(value, int) for value in raw_batch.word_lens):
        raise TypeError("POS word lengths must be integers")
    if not all(
            all(isinstance(word, str) for word in sentence)
            for sentence in raw_batch.text
        ):
        raise TypeError("POS batch text must contain strings")
    if not all(isinstance(value, int) for value in raw_batch.idx):
        raise TypeError("POS sentence indices must be integers")
    return _POSBatch(
        raw_batch.words,
        raw_batch.words_mask,
        raw_batch.wordchars,
        raw_batch.wordchars_mask,
        raw_batch.upos,
        raw_batch.xpos,
        raw_batch.ufeats,
        raw_batch.pretrained,
        raw_batch.orig_idx,
        raw_batch.word_orig_idx,
        raw_batch.lens,
        raw_batch.word_lens,
        raw_batch.text,
        raw_batch.idx,
    )


def _normalize_predictions(
        raw_predictions: Sequence[Sequence[Sequence[str]]],
    ) -> list[_SentencePOSTags]:
    """Validate predictions returned by the currently untyped POS model."""
    predictions: list[_SentencePOSTags] = []
    for raw_sentence in raw_predictions:
        sentence: _SentencePOSTags = []
        for raw_tag in raw_sentence:
            if isinstance(raw_tag, str) or len(raw_tag) != 3:
                raise TypeError(
                    "POS predictions must contain UPOS, XPOS, and features"
                )
            upos, xpos, feats = raw_tag
            if not all(isinstance(value, str) for value in (upos, xpos, feats)):
                raise TypeError("POS prediction values must be strings")
            sentence.append((upos, xpos, feats))
        predictions.append(sentence)
    return predictions


def _known_units(unit_to_id: Mapping[str, int]) -> list[str]:
    units: list[str] = []
    for unit in unit_to_id:
        if not isinstance(unit, str):
            raise TypeError("POS vocabulary entries must be strings")
        if unit not in VOCAB_PREFIX:
            units.append(unit)
    return units


@register_processor(name=POS)
class POSProcessor(UDProcessor):
    _pretrain: Optional[Pretrain]
    _trainer: Optional[Trainer]
    _vocab: Optional[MultiVocab]
    _tqdm: bool

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([POS])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(
            self,
            config: _POSModelConfig,
            pipeline: Pipeline,
            device: Union[str, torch.device],
        ) -> None:
        # get pretrained word vectors
        self._pretrain = (
            pipeline.foundation_cache.load_pretrain(config['pretrain_path'])
            if 'pretrain_path' in config
            else None
        )
        args = {'charlm_forward_file': config.get('forward_charlm_path', None),
                'charlm_backward_file': config.get('backward_charlm_path', None)}
        # set up trainer
        self._trainer = Trainer(
            pretrain=self._pretrain,
            model_file=config['model_path'],
            device=device,
            args=args,
            foundation_cache=pipeline.foundation_cache,
        )
        self._tqdm = bool(config.get('tqdm', False))

    def __str__(self) -> str:
        return "POSProcessor(%s)" % self.config['model_path']

    def _require_trainer(self) -> _POSTrainer:
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("The POS processor model is not loaded")
        return trainer

    def _require_vocab(self) -> MultiVocab:
        vocab = self._vocab
        if not isinstance(vocab, MultiVocab):
            raise RuntimeError("The POS processor vocabulary is not loaded")
        return vocab

    def _require_vocab_item(self, name: str) -> BaseVocab:
        vocab_item = self._require_vocab()[name]
        if not isinstance(vocab_item, BaseVocab):
            raise TypeError(f"The POS {name} vocabulary must be a BaseVocab")
        return vocab_item

    def get_known_xpos(
            self,
        ) -> Union[list[str], dict[_CompositeKey, set[str]]]:
        """
        Returns the xpos tags known by this model
        """
        xpos_vocab = self._require_vocab_item('xpos')
        if isinstance(xpos_vocab, CompositeVocab):
            if not isinstance(xpos_vocab, _CompositeVocabulary):
                raise TypeError(
                    "The composite XPOS vocabulary must map parts to units"
                )
            composite_units: list[tuple[_CompositeKey, list[str]]] = []
            for key, unit_to_id in xpos_vocab._unit2id.items():
                if not isinstance(key, (int, str)):
                    raise TypeError(
                        "Composite XPOS vocabulary keys must be strings or integers"
                    )
                if not isinstance(unit_to_id, Mapping):
                    raise TypeError(
                        "Composite XPOS vocabulary entries must be mappings"
                    )
                composite_units.append((key, _known_units(unit_to_id)))
            if len(composite_units) == 1:
                return composite_units[0][1]
            return {
                key: set(units)
                for key, units in composite_units
            }
        if not isinstance(xpos_vocab, _StringVocabulary):
            raise TypeError("The XPOS vocabulary must map tags to identifiers")
        return _known_units(xpos_vocab._unit2id)

    def is_composite_xpos(self) -> bool:
        """
        Returns if the xpos tags are part of a composite vocab
        """
        return isinstance(self._require_vocab_item('xpos'), CompositeVocab)

    def get_known_upos(self) -> list[str]:
        """
        Returns the upos tags known by this model
        """
        upos_vocab = self._require_vocab_item('upos')
        if not isinstance(upos_vocab, _StringVocabulary):
            raise TypeError("The UPOS vocabulary must map tags to identifiers")
        return _known_units(upos_vocab._unit2id)

    def get_known_feats(self) -> dict[str, set[str]]:
        """
        Returns the features known by this model
        """
        feats_vocab = self._require_vocab_item('feats')
        if not isinstance(feats_vocab, _CompositeVocabulary):
            raise TypeError(
                "The POS feature vocabulary must map features to values"
            )
        values: dict[str, set[str]] = {}
        for key, unit_to_id in feats_vocab._unit2id.items():
            if not isinstance(key, str):
                raise TypeError("POS feature vocabulary keys must be strings")
            if not isinstance(unit_to_id, Mapping):
                raise TypeError("POS feature vocabulary entries must be mappings")
            values[key] = set(_known_units(unit_to_id))
        return values

    def process(self, document: Document) -> Document:
        # currently, POS models are saved w/o the batch_maximum_tokens flag
        maximum_tokens = self.config.get('batch_maximum_tokens', 5000)
        if maximum_tokens is not None and not isinstance(maximum_tokens, int):
            raise TypeError("POS batch_maximum_tokens must be an integer or None")
        batch_size = self.config['batch_size']
        if not isinstance(batch_size, int):
            raise TypeError("POS batch_size must be an integer")

        dataset = Dataset(
            document, self.config, self._pretrain, vocab=self._require_vocab(),
            evaluation=True,
            sort_during_eval=True)
        batch = iter(dataset.to_length_limited_loader(
            batch_size=batch_size,
            maximum_tokens=maximum_tokens,
        ))
        preds: list[_SentencePOSTags] = []

        idx: list[int] = []
        trainer = self._require_trainer()
        with torch.no_grad():
            if self._tqdm:
                batch = tqdm(batch)
            for raw_batch in batch:
                if not isinstance(raw_batch, _RawPOSBatch):
                    raise TypeError(
                        "The POS data loader must provide a complete POS batch"
                    )
                pos_batch = _normalize_batch(raw_batch)
                idx.extend(pos_batch.idx)
                preds.extend(_normalize_predictions(trainer.predict(pos_batch)))

        preds = _normalize_predictions(unsort(preds, idx))
        document.set(
            [doc.UPOS, doc.XPOS, doc.FEATS],
            [tag for sentence in preds for tag in sentence],
        )
        return document
