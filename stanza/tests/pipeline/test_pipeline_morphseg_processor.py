import sys
from types import ModuleType
from typing import Optional

import pytest
import torch
from pytest import MonkeyPatch
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from stanza.pipeline.morphseg_processor import MorphSegProcessor
from stanza.pipeline.processor import ProcessorRequirementsException


class _AlignmentPosition:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _Prediction:
    def __init__(self, source: list[str]) -> None:
        self.prediction = ["COPY"] * len(source)
        self.alignment = [_AlignmentPosition(symbol) for symbol in source]


class _SequenceLabeller:
    class _Settings:
        def __init__(self) -> None:
            self.device = torch.device("mps")

    class _Module:
        def __init__(self) -> None:
            self.device = torch.device("mps")
            self.moves: list[torch.device] = []

        def to(self, device: torch.device) -> "_SequenceLabeller._Module":
            self.moves.append(device)
            return self

    class _Model:
        def __init__(self) -> None:
            self.model = _SequenceLabeller._Module()

    def __init__(self) -> None:
        self.model = self._Model()
        self.settings = self._Settings()

    def predict(
            self,
            sources: list[list[str]],
            features: Optional[list[list[str]]] = None,
            show_progress: bool = True,
        ) -> list[_Prediction]:
        return [_Prediction(source) for source in sources]


class _Segmenter:
    def __init__(self) -> None:
        self.device = torch.device("mps")
        self.sequence_labeller = _SequenceLabeller()

    def is_ready(self) -> bool:
        return True

    def normalize_for_morphology(self, text: str) -> str:
        return text.lower()


class _PipelineWithoutTokenizer:
    loaded_processors: list[MorphSegProcessor] = []
    load_list = [("morphseg", "default")]


def _rules2sent(source: list[str], actions: list[str]) -> str:
    assert len(source) == len(actions)
    if len(source) < 2:
        return "".join(source)
    return "{} @@{}".format("".join(source[:-1]), source[-1])


def test_morphseg_processes_all_words_without_loading_a_model(
        monkeypatch: MonkeyPatch,
    ) -> None:
    oracle_module = ModuleType("morphseg.training.oracle")
    setattr(oracle_module, "rules2sent", _rules2sent)
    monkeypatch.setitem(sys.modules, "morphseg.training.oracle", oracle_module)

    processor = MorphSegProcessor.__new__(MorphSegProcessor)
    processor._segmenter = _Segmenter()
    document = Document([
        [{"text": "Dogs"}, {"text": "run"}],
        [{"text": "I"}],
    ])

    assert processor.process(document) is document
    assert [word.morphemes for sentence in document.sentences for word in sentence.words] == [
        ["dog", "s"],
        ["ru", "n"],
        ["i"],
    ]


def test_morphseg_leaves_an_empty_document_unchanged() -> None:
    processor = MorphSegProcessor.__new__(MorphSegProcessor)
    processor._segmenter = _Segmenter()
    document = Document([])

    assert processor.process(document) is document


def test_morphseg_checks_tokenizer_requirement_before_loading_model() -> None:
    with pytest.raises(ProcessorRequirementsException):
        MorphSegProcessor(
            {"lang": "en"},
            _PipelineWithoutTokenizer(),
            "cpu",
        )


def test_morphseg_uses_filtered_custom_model_path(
        monkeypatch: MonkeyPatch,
    ) -> None:
    calls: list[tuple[str, bool, Optional[str], bool]] = []

    class _ConfiguredSegmenter(_Segmenter):
        def __init__(
                self,
                lang: str,
                load_pretrained: bool = True,
                model_filepath: Optional[str] = None,
                is_local: bool = True,
            ) -> None:
            super().__init__()
            calls.append((
                lang,
                load_pretrained,
                model_filepath,
                is_local,
            ))
            configured_segmenters.append(self)

    configured_segmenters: list[_ConfiguredSegmenter] = []

    morphseg_module = ModuleType("morphseg")
    setattr(morphseg_module, "MorphemeSegmenter", _ConfiguredSegmenter)
    monkeypatch.setitem(sys.modules, "morphseg", morphseg_module)

    processor = MorphSegProcessor.__new__(MorphSegProcessor)
    pipeline = Pipeline.__new__(Pipeline)
    processor._set_up_model(
        {"lang": "en", "model_path": "/tmp/custom-morphseg.safetensors"},
        pipeline,
        torch.device("cpu"),
    )

    assert calls == [
        ("en", True, "/tmp/custom-morphseg.safetensors", True),
    ]
    segmenter = configured_segmenters[0]
    assert segmenter.device == torch.device("cpu")
    assert segmenter.sequence_labeller.settings.device == torch.device("cpu")
    model = segmenter.sequence_labeller.model.model
    assert model.moves == [torch.device("cpu")]
    assert model.device == torch.device("cpu")
