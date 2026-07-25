"""
Processor for performing multi-word-token expansion
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from typing import NamedTuple, Optional, Protocol, TYPE_CHECKING, TypedDict, Union

import torch

from stanza.models.common.doc import Document
from stanza.models.common.vocab import DeltaVocab
from stanza.models.mwt.data import DataLoader
from stanza.models.mwt.trainer import Trainer
from stanza.models.mwt.vocab import Vocab
from stanza.pipeline._constants import *
from stanza.pipeline.processor import ProcessorDevice, UDProcessor, register_processor

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline


_ModelPath = Union[str, os.PathLike[str]]
_MWTVocab = Union[Vocab, DeltaVocab]


class _MWTSetupConfig(TypedDict):
    model_path: _ModelPath


class _MWTOptionalRuntimeConfig(TypedDict, total=False):
    ensemble_dict: bool


class _MWTModelConfig(_MWTOptionalRuntimeConfig):
    batch_size: int
    dict_only: bool


class _MWTRuntimeConfig(_MWTModelConfig):
    model_path: _ModelPath


class _MWTBatch(NamedTuple):
    src: torch.Tensor
    src_mask: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    orig_text: Sequence[str]
    orig_idx: Sequence[int]


class _MWTBatchLoader(Protocol):
    @property
    def doc(self) -> Document:
        ...

    @property
    def vocab(self) -> _MWTVocab:
        ...

    def __len__(self) -> int:
        ...

    def to_loader(self) -> Iterable[_MWTBatch]:
        ...


class _MWTTrainer(Protocol):
    @property
    def args(self) -> Optional[_MWTModelConfig]:
        ...

    @property
    def vocab(self) -> Optional[Vocab]:
        ...

    def predict_dict(self, words: Sequence[str]) -> Sequence[str]:
        ...

    def predict(
            self,
            batch: _MWTBatch,
            unsort: bool = True,
            never_decode_unk: bool = False,
            vocab: Optional[_MWTVocab] = None,
        ) -> Sequence[str]:
        ...

    def ensemble(
            self,
            cands: Sequence[str],
            other_preds: Sequence[str],
        ) -> Sequence[str]:
        ...

@register_processor(MWT)
class MWTProcessor(UDProcessor):
    _config: _MWTRuntimeConfig
    _trainer: Optional[_MWTTrainer]

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([MWT])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(
            self,
            config: _MWTSetupConfig,
            pipeline: Optional[Pipeline],
            device: Optional[ProcessorDevice],
        ) -> None:
        self._trainer = Trainer(model_file=config['model_path'], device=device)

    def _runtime_config(self) -> _MWTRuntimeConfig:
        return self._config

    def _require_trainer(self) -> _MWTTrainer:
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("The MWT processor model is not loaded")
        return trainer

    def _require_vocab(self) -> Vocab:
        vocab = self._vocab
        if not isinstance(vocab, Vocab):
            raise RuntimeError("The MWT processor vocabulary is not loaded")
        return vocab

    def build_batch(self, document: Document) -> DataLoader:
        config = self._runtime_config()
        return DataLoader(
            document,
            config['batch_size'],
            config,
            vocab=self._require_vocab(),
            evaluation=True,
            expand_unk_vocab=True,
        )

    @staticmethod
    def _batch_loader_view(batch: DataLoader) -> _MWTBatchLoader:
        return batch

    def process(self, document: Document) -> Document:
        batch = self._batch_loader_view(self.build_batch(document))
        trainer = self._require_trainer()
        config = self._runtime_config()

        # process the rest
        raw_expansions = batch.doc.get_mwt_expansions(evaluation=True)
        expansions: list[str] = []
        for expansion in raw_expansions:
            if not isinstance(expansion, str):
                raise TypeError(
                    "Evaluation MWT expansions must be strings"
                )
            expansions.append(expansion)
        if len(batch) > 0:
            # decide trainer type and run eval
            if config['dict_only']:
                preds = list(trainer.predict_dict(expansions))
            else:
                with torch.no_grad():
                    preds: list[str] = []
                    for model_batch in batch.to_loader():
                        preds.extend(
                            trainer.predict(
                                model_batch,
                                never_decode_unk=True,
                                vocab=batch.vocab,
                            )
                        )

                if config.get('ensemble_dict', False):
                    preds = list(trainer.ensemble(expansions, preds))
        else:
            # skip eval if dev data does not exist
            preds = []

        batch.doc.set_mwt_expansions(preds, process_manual_expanded=False)
        return batch.doc

    def bulk_process(self, docs: list[Document]) -> list[Document]:
        """
        MWT processor counts some statistics on the individual docs, so we need to separately redo those stats
        """
        docs = super().bulk_process(docs)
        for doc in docs:
            doc._count_words()
        return docs
