from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Optional, Protocol, TYPE_CHECKING, TypedDict, Union

import torch

from stanza.models.common.doc import Document
from stanza.pipeline.core import UnsupportedProcessorError
from stanza.pipeline.processor import UDProcessor, register_processor
from stanza.pipeline._constants import MORPHSEG, TOKENIZE

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline


class _MorphSegConfig(TypedDict, total=False):
    lang: str
    model_path: Optional[str]


_Device = Union[str, torch.device]


class _AlignmentPosition(Protocol):
    @property
    def symbol(self) -> str:
        ...


class _Prediction(Protocol):
    @property
    def prediction(self) -> list[str]:
        ...

    @property
    def alignment(self) -> Sequence[_AlignmentPosition]:
        ...


class _DeviceAwareModule(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    @device.setter
    def device(self, value: torch.device) -> None:
        ...

    def to(self, device: torch.device) -> _DeviceAwareModule:
        ...


class _TrainedModel(Protocol):
    @property
    def model(self) -> _DeviceAwareModule:
        ...


class _MorphSegSettings(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    @device.setter
    def device(self, value: torch.device) -> None:
        ...


class _SequenceLabeller(Protocol):
    @property
    def model(self) -> Optional[_TrainedModel]:
        ...

    @property
    def settings(self) -> _MorphSegSettings:
        ...

    def predict(
            self,
            sources: list[list[str]],
            features: Optional[list[list[str]]] = None,
            show_progress: bool = True,
        ) -> Sequence[_Prediction]:
        ...


class _MorphemeSegmenter(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    @device.setter
    def device(self, value: torch.device) -> None:
        ...

    @property
    def sequence_labeller(self) -> Optional[_SequenceLabeller]:
        ...

    def is_ready(self) -> bool:
        ...

    def normalize_for_morphology(self, text: str) -> str:
        ...


def _move_segmenter_to_device(
        segmenter: _MorphemeSegmenter,
        device: _Device,
    ) -> None:
    target_device = torch.device(device)
    sequence_labeller = segmenter.sequence_labeller
    if sequence_labeller is None or sequence_labeller.model is None:
        raise RuntimeError("Morphseg model is not ready")

    segmenter.device = target_device
    sequence_labeller.settings.device = target_device
    sequence_labeller.model.model.to(target_device)
    sequence_labeller.model.model.device = target_device


class _RulesToSentence(Protocol):
    def __call__(self, source: list[str], actions: list[str]) -> str:
        ...


def _decode_prediction(
        prediction: _Prediction,
        rules2sent: _RulesToSentence,
    ) -> list[str]:
    return rules2sent(
        source=[align_pos.symbol for align_pos in prediction.alignment],
        actions=prediction.prediction,
    ).split(' @@')


@register_processor(name=MORPHSEG)
class MorphSegProcessor(UDProcessor):
    PROVIDES_DEFAULT: ClassVar[set[str]] = {MORPHSEG}
    REQUIRES_DEFAULT: ClassVar[set[str]] = {TOKENIZE}

    _config: _MorphSegConfig
    _pipeline: Pipeline
    _segmenter: _MorphemeSegmenter

    def _set_up_model(
            self,
            config: _MorphSegConfig,
            pipeline: Pipeline,
            device: _Device,
        ) -> None:
        try:
            from morphseg import MorphemeSegmenter
        except ImportError:
            raise ImportError(
                "morphseg is required for morpheme segmentation. "
                "Install it with: pip install morphseg"
            )

        lang = config.get('lang', 'en')
        model_path = config.get('model_path', None)

        if model_path:
            self._segmenter = MorphemeSegmenter(
                lang=lang,
                load_pretrained=True,
                model_filepath=model_path,
                is_local=True
            )
        else:
            self._segmenter = MorphemeSegmenter(
                lang=lang,
                load_pretrained=True
            )
        if not self._segmenter.is_ready():
            raise UnsupportedProcessorError("morphseg", lang)
        _move_segmenter_to_device(self._segmenter, device)

    def process(self, document: Document) -> Document:
        # Collect all words from all sentences
        all_words: list[str] = []
        word_mapping: list[tuple[int, int]] = []

        for sent_idx, sent in enumerate(document.sentences):
            if not sent.words:
                continue
            for word_idx, word in enumerate(sent.words):
                all_words.append(word.text)
                word_mapping.append((sent_idx, word_idx))

        if not all_words:
            return document

        # Prepare input for morphseg (it expects normalized, lowercased character lists)
        word_char_lists = [
            list(self._segmenter.normalize_for_morphology(word))
            for word in all_words
        ]

        # Batch predict using the internal sequence_labeller
        sequence_labeller = self._segmenter.sequence_labeller
        if sequence_labeller is None:
            raise RuntimeError("Morphseg model is not ready")
        predictions = sequence_labeller.predict(sources=word_char_lists)
        if len(predictions) != len(word_mapping):
            raise ValueError(
                "Morphseg returned a different number of predictions and words"
            )

        # Extract segmentations from predictions
        from morphseg.training.oracle import rules2sent as untyped_rules2sent
        rules2sent: _RulesToSentence = untyped_rules2sent
        segmentations: list[list[str]] = [
            _decode_prediction(pred, rules2sent)
            for pred in predictions
        ]

        # Assign segmentations back to words
        for (sent_idx, word_idx), seg in zip(word_mapping, segmentations):
            document.sentences[sent_idx].words[word_idx].morphemes = seg

        return document
