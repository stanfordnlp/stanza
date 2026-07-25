"""
Processor that attaches a constituency tree to a sentence
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from typing import (
    ClassVar,
    Literal,
    Optional,
    Protocol,
    SupportsIndex,
    SupportsInt,
    TYPE_CHECKING,
    TypedDict,
    Union,
    runtime_checkable,
)

import torch
from stanza.models.constituency.base_trainer import BaseTrainer
from stanza.models.constituency.lstm_model import (
    ConstituencyComposition,
    SentenceBoundary,
    StackHistory,
)
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.trainer import Trainer

from stanza.models.common.doc import Document
from stanza.models.common.utils import sort_with_indices, unsort
from stanza.utils.get_tqdm import get_tqdm
from stanza.pipeline._constants import CONSTITUENCY, POS, TOKENIZE
from stanza.pipeline.processor import UDProcessor, register_processor

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline

_ModelPath = Union[str, os.PathLike[str]]
_Device = Union[str, torch.device]
_ModelArgument = Union[
    None,
    bool,
    int,
    float,
    str,
    os.PathLike[str],
    torch.device,
    SupportsInt,
    SupportsIndex,
    Sequence[int],
    Sequence[str],
    ConstituencyComposition,
    SentenceBoundary,
    StackHistory,
    TransitionScheme,
]
_TaggedWord = tuple[str, str]
_TaggedSentence = list[_TaggedWord]
_PathConfigKey = Literal[
    "pretrain_path",
    "forward_charlm_path",
    "backward_charlm_path",
]


class _ConstituencyLoadArgs(TypedDict):
    wordvec_pretrain_file: Optional[_ModelPath]
    charlm_forward_file: Optional[_ModelPath]
    charlm_backward_file: Optional[_ModelPath]
    device: _Device


def _optional_model_path(
        config: Mapping[str, _ModelArgument],
        key: _PathConfigKey,
    ) -> Optional[_ModelPath]:
    value = config.get(key)
    if value is None or isinstance(value, (str, os.PathLike)):
        return value
    raise TypeError(f"The constituency {key} must be a path")


def _require_model_path(
        config: Mapping[str, _ModelArgument],
    ) -> _ModelPath:
    value = config["model_path"]
    if isinstance(value, (str, os.PathLike)):
        return value
    raise TypeError("The constituency model_path must be a path")


class _TaggedSentenceBatch(Protocol):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[_TaggedSentence]:
        ...


class _ProgressFactory(Protocol):
    def __call__(
            self,
            sentences: _TaggedSentenceBatch,
        ) -> _TaggedSentenceBatch:
        ...


@runtime_checkable
class _ConstituencyModel(Protocol):
    args: Mapping[str, _ModelArgument]

    def eval(self) -> _ConstituencyModel:
        ...

    def uses_xpos(self) -> bool:
        ...

    def parse_tagged_words(
            self,
            words: _TaggedSentenceBatch,
            batch_size: int,
        ) -> list[Tree]:
        ...


@runtime_checkable
class _ConstituencyLabels(Protocol):
    constituents: Sequence[str]


def _require_constituency_model(
        model: torch.nn.Module,
    ) -> _ConstituencyModel:
    if not isinstance(model, _ConstituencyModel):
        raise TypeError(
            "The loaded model does not provide the constituency parser "
            "interface"
        )
    return model


tqdm: _ProgressFactory = get_tqdm()

@register_processor(CONSTITUENCY)
class ConstituencyProcessor(UDProcessor):
    _pretagged: bool
    _trainer: Optional[BaseTrainer]
    _model: _ConstituencyModel
    _batch_size: int
    _tqdm: bool

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT: ClassVar[set[str]] = set([CONSTITUENCY])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT: ClassVar[set[str]] = set([TOKENIZE, POS])

    # default batch size, measured in sentences
    DEFAULT_BATCH_SIZE: ClassVar[int] = 50

    def _set_up_requires(self) -> None:
        self._pretagged = bool(self._config.get('pretagged'))
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(
            self,
            config: Mapping[str, _ModelArgument],
            pipeline: Pipeline,
            device: _Device,
        ) -> None:
        # set up model
        # pretrain and charlm paths are args from the config
        # bert (if used) will be chosen from the model save file
        args: _ConstituencyLoadArgs = {
            "wordvec_pretrain_file": _optional_model_path(
                config,
                "pretrain_path",
            ),
            "charlm_forward_file": _optional_model_path(
                config,
                "forward_charlm_path",
            ),
            "charlm_backward_file": _optional_model_path(
                config,
                "backward_charlm_path",
            ),
            "device": device,
        }
        trainer = Trainer.load(filename=_require_model_path(config),
                               args=args,
                               foundation_cache=pipeline.foundation_cache)
        self._trainer = trainer
        self._model = _require_constituency_model(trainer.model)
        self._model.eval()
        # batch size counted as sentences
        configured_batch_size = config.get(
            'batch_size',
            ConstituencyProcessor.DEFAULT_BATCH_SIZE,
        )
        if not isinstance(
                configured_batch_size,
                (str, bytes, bytearray, SupportsInt, SupportsIndex),
            ):
            raise TypeError(
                "The constituency batch_size must be convertible to int"
            )
        self._batch_size = int(configured_batch_size)
        self._tqdm = bool(config.get('tqdm', False))

    def _set_up_final_config(
            self,
            config: Mapping[str, _ModelArgument],
        ) -> None:
        loaded_args = self._model.args
        loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        loaded_args.update(config)
        self._config = loaded_args

    def process(self, document: Document) -> Document:
        sentences = document.sentences

        if self._model.uses_xpos():
            raw_words = [[(w.text, w.xpos) for w in s.words] for s in sentences]
            tag_name = "XPOS"
        else:
            raw_words = [[(w.text, w.upos) for w in s.words] for s in sentences]
            tag_name = "UPOS"

        words: list[_TaggedSentence] = []
        for sentence_idx, sentence in enumerate(raw_words):
            tagged_sentence: _TaggedSentence = []
            for word_idx, (word, tag) in enumerate(sentence):
                if tag is None:
                    raise ValueError(
                        f"Missing {tag_name} tag for word {word_idx} "
                        f"in sentence {sentence_idx}"
                    )
                tagged_sentence.append((word, tag))
            words.append(tagged_sentence)

        sort_result: tuple[
            Sequence[_TaggedSentence],
            Sequence[int],
        ] = sort_with_indices(words, key=len, reverse=True)
        sorted_words, original_indices = sort_result
        words_to_parse: _TaggedSentenceBatch = sorted_words
        if self._tqdm:
            words_to_parse = tqdm(words_to_parse)

        sorted_trees = self._model.parse_tagged_words(
            words_to_parse,
            self._batch_size,
        )
        trees: list[Tree] = unsort(sorted_trees, original_indices)
        document.set(CONSTITUENCY, trees, to_sentence=True)
        return document

    def get_constituents(self) -> set[str]:
        """
        Return a set of the constituents known by this model

        For a pipeline, this can be queried with
          pipeline.processors["constituency"].get_constituents()
        """
        model = self._model
        if not isinstance(model, _ConstituencyLabels):
            raise AttributeError(
                "The constituency model does not expose constituent labels"
            )
        return set(model.constituents)
