"""
Processor for performing dependency parsing
"""

from __future__ import annotations

import operator
from typing import Mapping, Optional, Protocol, Sequence, SupportsIndex, Union, runtime_checkable

import torch

from stanza.models.common import doc
from stanza.models.common.doc import Document
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.utils import unsort
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.trainer import Trainer
from stanza.models.pos.vocab import MultiVocab, WordVocab
from stanza.pipeline._constants import DEPPARSE, LEMMA, POS, TOKENIZE
from stanza.pipeline.processor import UDProcessor, register_processor

# these imports trigger the "register_variant" decorations
from stanza.pipeline.external.corenlp_converter_depparse import ConverterDepparse

DEFAULT_SEPARATE_BATCH = 150

_PredictionValue = Union[SupportsIndex, str]
_DepparseConfigValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Sequence["_DepparseConfigValue"],
    dict[str, "_DepparseConfigValue"],
]
_RawDependencyPrediction = Sequence[_PredictionValue]
_DependencyPrediction = tuple[int, str]
_DependencyBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[Sequence[str]],
]


@runtime_checkable
class _DepparseTrainer(Protocol):
    @property
    def args(self) -> Mapping[str, _DepparseConfigValue]:
        ...

    @property
    def vocab(self) -> MultiVocab:
        ...

    def predict(
            self,
            batch: _DependencyBatch,
            unsort: bool = True,
            resolve_head_constraints: bool = False,
        ) -> Sequence[Sequence[_RawDependencyPrediction]]:
        ...


_DepparseTrainerState = Union[Trainer, _DepparseTrainer]


def _normalize_predictions(
        raw_predictions: Sequence[Sequence[_RawDependencyPrediction]],
    ) -> list[list[_DependencyPrediction]]:
    """Validate predictions coming from an as-yet untyped model boundary."""
    predictions: list[list[_DependencyPrediction]] = []
    for raw_sentence in raw_predictions:
        sentence: list[_DependencyPrediction] = []
        for raw_prediction in raw_sentence:
            if len(raw_prediction) != 2:
                raise TypeError(
                    "Dependency predictions must contain a head and relation"
                )
            raw_head, raw_relation = raw_prediction
            if isinstance(raw_head, (bool, str)):
                raise TypeError("Dependency prediction heads must be integers")
            try:
                head = operator.index(raw_head)
            except TypeError as e:
                raise TypeError(
                    "Dependency prediction heads must be integers"
                ) from e
            if not isinstance(raw_relation, str):
                raise TypeError(
                    "Dependency prediction relations must be strings"
                )
            sentence.append((head, raw_relation))
        predictions.append(sentence)
    return predictions


def _flatten_document_predictions(
        document: Document,
        predictions: Sequence[Sequence[_DependencyPrediction]],
    ) -> list[_DependencyPrediction]:
    """Validate prediction dimensions and head ranges before mutating a document."""
    if len(predictions) != len(document.sentences):
        raise ValueError(
            "Dependency predictions must contain one result per sentence"
        )

    flattened: list[_DependencyPrediction] = []
    for sentence_idx, (sentence, sentence_predictions) in enumerate(
            zip(document.sentences, predictions)
        ):
        word_count = len(sentence.words)
        if len(sentence_predictions) != word_count:
            raise ValueError(
                "Dependency predictions for sentence "
                f"{sentence_idx} contain {len(sentence_predictions)} results "
                f"for {word_count} words"
            )
        for head, relation in sentence_predictions:
            if head < 0 or head > word_count:
                raise ValueError(
                    "Dependency prediction head "
                    f"{head} is outside 0..{word_count} in sentence {sentence_idx}"
                )
            flattened.append((head, relation))
    return flattened


@register_processor(name=DEPPARSE)
class DepparseProcessor(UDProcessor):
    _pretagged: bool
    _pretrain: Optional[Pretrain]
    _trainer: Optional[_DepparseTrainerState]
    _vocab: Optional[MultiVocab]

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([DEPPARSE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS, LEMMA])

    def __init__(self, config, pipeline, device) -> None:
        self._pretagged = False
        super().__init__(config, pipeline, device)

    def _set_up_requires(self) -> None:
        self._pretagged = bool(self._config.get('pretagged'))
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(self, config, pipeline, device) -> None:
        configured_trainer = config.get('trainer')
        if configured_trainer is not None:
            if not isinstance(configured_trainer, _DepparseTrainer):
                raise TypeError(
                    "The depparse trainer configuration must provide predict()"
                )
            self._trainer = configured_trainer
            return

        pretrain = (
            pipeline.foundation_cache.load_pretrain(config['pretrain_path'])
            if 'pretrain_path' in config
            else None
        )
        if pretrain is not None and not isinstance(pretrain, Pretrain):
            raise TypeError("The depparse pretrain must be a Pretrain")
        self._pretrain = pretrain
        args = {'charlm_forward_file': config.get('forward_charlm_path', None),
                'charlm_backward_file': config.get('backward_charlm_path', None)}
        trainer = Trainer.load(
            filename=config['model_path'],
            args=args,
            pretrain=pretrain,
            device=device,
            foundation_cache=pipeline.foundation_cache,
        )
        self._trainer = trainer

    def _require_trainer(self) -> _DepparseTrainerState:
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("The depparse processor model is not loaded")
        return trainer

    def _require_vocab(self) -> MultiVocab:
        vocab = self._vocab
        if not isinstance(vocab, MultiVocab):
            raise RuntimeError("The depparse processor vocabulary is not loaded")
        return vocab

    def _model_state(
            self,
        ) -> tuple[_DepparseTrainerState, Optional[Pretrain], MultiVocab]:
        return self._require_trainer(), self._pretrain, self._require_vocab()

    def get_known_relations(self) -> list[str]:
        """
        Return a list of relations which this processor can produce
        """
        deprel_vocab = self._require_vocab()['deprel']
        if not isinstance(deprel_vocab, WordVocab):
            raise TypeError(
                "The depparse relation vocabulary must be a WordVocab"
            )
        relations: list[str] = []
        for relation in deprel_vocab._unit2id:
            if not isinstance(relation, str):
                raise TypeError(
                    "Dependency relation vocabulary entries must be strings"
                )
            if relation not in VOCAB_PREFIX:
                relations.append(relation)
        return relations

    def process(self, document: Document) -> Document:
        if hasattr(self, '_variant'):
            return self._variant.process(document)

        if any(word.upos is None and word.xpos is None for sentence in document.sentences for word in sentence.words):
            raise ValueError("POS not run before depparse!")
        trainer, pretrain, vocab = self._model_state()
        try:
            batch = DataLoader(
                document,
                self.config['batch_size'],
                self.config,
                pretrain,
                vocab=vocab,
                evaluation=True,
                sort_during_eval=self.config.get('sort_during_eval', True),
                min_length_to_batch_separately=self.config.get(
                    'min_length_to_batch_separately',
                    DEFAULT_SEPARATE_BATCH,
                ),
            )
            with torch.no_grad():
                preds: list[list[_DependencyPrediction]] = []
                resolve_head_constraints = self.config.get('resolve_head_constraints', True)
                for batch_item in batch:
                    raw_predictions = trainer.predict(
                        batch_item,
                        resolve_head_constraints=resolve_head_constraints,
                    )
                    preds.extend(_normalize_predictions(raw_predictions))
            if batch.data_orig_idx is not None:
                preds = _normalize_predictions(
                    unsort(preds, batch.data_orig_idx)
                )
            processed_document = batch.doc
            if not isinstance(processed_document, Document):
                raise TypeError(
                    "The depparse data loader did not contain a Document"
                )
            flattened_predictions = _flatten_document_predictions(
                processed_document,
                preds,
            )
            processed_document.set(
                (doc.HEAD, doc.DEPREL),
                flattened_predictions,
            )
            # build dependencies based on predictions
            for sentence in processed_document.sentences:
                sentence.build_dependencies()
            return processed_document
        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory. Tried to allocate"):
                new_message = str(e) + " ... You may be able to compensate for this by separating long sentences into their own batch with a parameter such as depparse_min_length_to_batch_separately=150 or by limiting the overall batch size with depparse_batch_size=400."
                raise RuntimeError(new_message) from e
            raise
