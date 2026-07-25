"""Processor that attaches a sentiment score to a sentence

The model used is a generally a model trained on the Stanford
Sentiment Treebank or some similar dataset.  When run, this processor
attaches a score in the form of a string to each sentence in the
document.

TODO: a possible way to generalize this would be to make it a
ClassifierProcessor and have "sentiment" be an option.
"""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Protocol, TypedDict, Union, runtime_checkable

import torch

from stanza.models.classifiers.config import CNNConfig, ConstituencyConfig
from stanza.models.classifiers.trainer import Trainer
from stanza.models.classifiers.utils import ModelType
from stanza.models.common.doc import Document
from stanza.models.constituency.parse_tree import Tree
from stanza.pipeline._constants import SENTIMENT, TOKENIZE
from stanza.pipeline.processor import UDProcessor, register_processor

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline


_ModelPath = Union[str, os.PathLike[str]]
_Device = Union[str, torch.device]
_SentimentSentence = Union[list[str], Tree]
_ExtractedSentimentSentence = Union[_SentimentSentence, str]
_SentimentModelConfig = Union[CNNConfig, ConstituencyConfig]


class _RequiredSentimentSetupConfig(TypedDict):
    model_path: Optional[_ModelPath]


class _SentimentSetupConfig(_RequiredSentimentSetupConfig, total=False):
    pretrain_path: Optional[_ModelPath]
    forward_charlm_path: Optional[_ModelPath]
    backward_charlm_path: Optional[_ModelPath]
    batch_size: int


@runtime_checkable
class _SentimentModel(Protocol):
    @property
    def config(self) -> _SentimentModelConfig:
        ...

    def extract_sentences(
            self,
            doc: Document,
        ) -> list[Optional[_ExtractedSentimentSentence]]:
        ...

    def label_sentences(
            self,
            sentences: list[_SentimentSentence],
            batch_size: Optional[int] = None,
        ) -> list[int]:
        ...


def _normalize_sentiment_sentences(
        extracted: list[Optional[_ExtractedSentimentSentence]],
    ) -> list[_SentimentSentence]:
    sentences: list[_SentimentSentence] = []
    for sentence in extracted:
        if isinstance(sentence, Tree):
            sentences.append(sentence)
        elif isinstance(sentence, list) and all(
                isinstance(word, str) for word in sentence
            ):
            sentences.append(sentence)
        else:
            raise TypeError(
                "Sentiment input must contain token text or constituency trees"
            )
    return sentences


def _require_sentiment_model(model: torch.nn.Module) -> _SentimentModel:
    if not isinstance(model, _SentimentModel):
        raise TypeError(
            "The loaded sentiment model does not provide the classifier interface"
        )
    return model


@register_processor(SENTIMENT)
class SentimentProcessor(UDProcessor):
    _trainer: Optional[Trainer]
    _model: _SentimentModel
    _model_type: ModelType
    _batch_size: int

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([SENTIMENT])
    # set of processor requirements for this processor
    # TODO: a constituency based model needs CONSTITUENCY as well
    # issue: by the time we load the model in Processor.__init__,
    # the requirements are already prepared
    REQUIRES_DEFAULT = set([TOKENIZE])

    # default batch size, measured in words per batch
    DEFAULT_BATCH_SIZE = 5000

    def _set_up_model(
            self,
            config: _SentimentSetupConfig,
            pipeline: Pipeline,
            device: _Device,
        ) -> None:
        # get pretrained word vectors
        pretrain_path = config.get('pretrain_path', None)
        forward_charlm_path = config.get('forward_charlm_path', None)
        backward_charlm_path = config.get('backward_charlm_path', None)
        # elmo does not have a convenient way to download intermediate
        # models the way stanza downloads charlms & pretrains or
        # transformers downloads bert etc
        # however, elmo in general is not as good as using a
        # transformer, so it is unlikely we will ever fix this
        args = SimpleNamespace(device = device,
                               charlm_forward_file = forward_charlm_path,
                               charlm_backward_file = backward_charlm_path,
                               wordvec_pretrain_file = pretrain_path,
                               elmo_model = None,
                               use_elmo = False,
                               save_dir = None)
        filename = config['model_path']
        if filename is None:
            raise FileNotFoundError("No model specified for the sentiment processor.  Perhaps it is not supported for the language.  {}".format(config))
        # set up model
        trainer = Trainer.load(filename=filename,
                               args=args,
                               foundation_cache=pipeline.foundation_cache)
        model = _require_sentiment_model(trainer.model)
        self._trainer = trainer
        self._model = model
        self._model_type = self._model.config.model_type
        # batch size counted as words
        self._batch_size = config.get('batch_size', SentimentProcessor.DEFAULT_BATCH_SIZE)

    def _set_up_final_config(self, config: _SentimentSetupConfig) -> None:
        loaded_args = dataclasses.asdict(self._model.config)
        loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        loaded_args.update(config)
        self._config = loaded_args


    def process(self, document: Document) -> Document:
        sentences = _normalize_sentiment_sentences(
            self._model.extract_sentences(document)
        )
        with torch.no_grad():
            labels = self._model.label_sentences(sentences, batch_size=self._batch_size)
        # TODO: allow a classifier processor for any attribute, not just sentiment
        document.set(SENTIMENT, labels, to_sentence=True)
        return document
