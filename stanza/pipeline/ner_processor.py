"""
Processor for performing named entity tagging.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional, Protocol, Sequence, TYPE_CHECKING, Union, runtime_checkable

import torch

from stanza.models.common import doc as doc_module
from stanza.models.common.doc import Document, Span
from stanza.models.common.exceptions import ForwardCharlmNotFoundError, BackwardCharlmNotFoundError
from stanza.models.ner.data import DataLoader
from stanza.models.ner.model import NERTagger
from stanza.models.ner.trainer import Trainer
from stanza.models.ner.vocab import MultiVocab
from stanza.models.ner.utils import merge_tags
from stanza.pipeline._constants import NER, TOKENIZE
from stanza.pipeline.processor import UDProcessor, register_processor

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline

logger = logging.getLogger('stanza')


_NERConfigValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Sequence["_NERConfigValue"],
    dict[str, "_NERConfigValue"],
]
_NERConfig = dict[str, _NERConfigValue]
_Device = Union[str, torch.device]
_NERBatch = tuple[
    Sequence[Sequence[str]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[int],
    Sequence[Sequence[Sequence[int]]],
]


@runtime_checkable
class _BertTokenizer(Protocol):
    model_max_length: int
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]

    def tokenize(self, text: str) -> list[str]:
        ...

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        ...


class _NERTrainer(Protocol):
    @property
    def args(self) -> _NERConfig:
        ...

    @property
    def vocab(self) -> MultiVocab:
        ...

    @property
    def model(self) -> NERTagger:
        ...

    def predict(
            self,
            batch: _NERBatch,
            unsort: bool = True,
        ) -> list[list[str]]:
        ...

    def get_known_tags(self) -> list[str]:
        ...


_NERFieldContents = Union[list[str], list[tuple[str, ...]]]


class _NERDocumentWriter(Protocol):
    def set(
            self,
            fields: Union[str, Sequence[str]],
            contents: _NERFieldContents,
            to_token: bool = False,
            to_sentence: bool = False,
        ) -> None:
        ...

    def build_ents(self) -> list[Span]:
        ...


def _write_predictions(
        document: _NERDocumentWriter,
        ner_tags: list[str],
        multi_ner_tags: list[tuple[str, ...]],
    ) -> int:
    document.set([doc_module.NER], ner_tags, to_token=True)
    document.set([doc_module.MULTI_NER], multi_ner_tags, to_token=True)
    return len(document.build_ents())


@register_processor(name=NER)
class NERProcessor(UDProcessor):
    _predict_tagset: dict[int, int]
    _trainer: Optional[_NERTrainer]
    _vocab: Optional[MultiVocab]
    trainers: Optional[list[_NERTrainer]]
    configs: list[_NERConfig]
    model_paths: list[str]

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT: ClassVar[set[str]] = {NER}
    # set of processor requirements for this processor
    REQUIRES_DEFAULT: ClassVar[set[str]] = {TOKENIZE}

    @staticmethod
    def _get_dependencies(
            config: _NERConfig,
            dep_name: str,
        ) -> list[Optional[str]]:
        configured_path = config.get(dep_name)
        if configured_path is not None:
            if not isinstance(configured_path, str):
                raise TypeError(f"{dep_name} must be a string")
            return [path if path else None for path in configured_path.split(";")]

        configured_dependencies = config.get('dependencies', [])
        if not isinstance(configured_dependencies, (list, tuple)):
            raise TypeError("dependencies must be a sequence of mappings")

        dependencies: list[Optional[str]] = []
        for dependency in configured_dependencies:
            if not isinstance(dependency, dict):
                raise TypeError("dependencies must be a sequence of mappings")
            configured_dependency = dependency.get(dep_name)
            if configured_dependency is not None and not isinstance(configured_dependency, str):
                raise TypeError(f"{dep_name} dependencies must be strings")
            dependencies.append(configured_dependency)
        return dependencies

    @staticmethod
    def _get_model_paths(config: _NERConfig) -> list[str]:
        configured_paths = config.get('model_path')
        if isinstance(configured_paths, str):
            return configured_paths.split(";")
        if not isinstance(configured_paths, (list, tuple)):
            raise TypeError("model_path must be a string or sequence of strings")

        model_paths: list[str] = []
        for model_path in configured_paths:
            if not isinstance(model_path, str):
                raise TypeError("model_path must be a string or sequence of strings")
            model_paths.append(model_path)
        return model_paths

    def _require_trainers(self) -> list[_NERTrainer]:
        trainers = self.trainers
        if not trainers:
            raise RuntimeError("Somehow there are no models loaded!")
        return trainers

    @staticmethod
    def _get_bert_tokenizer(
            model: NERTagger,
        ) -> Optional[_BertTokenizer]:
        tokenizer = getattr(model, "bert_tokenizer", None)
        if tokenizer is None:
            return None
        if not isinstance(tokenizer, _BertTokenizer):
            raise TypeError("NER model has an incompatible BERT tokenizer")
        return tokenizer

    def _set_up_model(
            self,
            config: _NERConfig,
            pipeline: Pipeline,
            device: _Device,
        ) -> None:
        # set up trainer
        model_paths = self._get_model_paths(config)

        charlm_forward_files = self._get_dependencies(config, 'forward_charlm_path')
        charlm_backward_files = self._get_dependencies(config, 'backward_charlm_path')
        pretrain_files = self._get_dependencies(config, 'pretrain_path')

        # allow predict_tagset to be specified as an int
        # (which only applies to the first model)
        # or as a string ";" separated list of ints
        self._predict_tagset = {}
        predict_tagset = config.get('predict_tagset', None)
        if predict_tagset:
            if isinstance(predict_tagset, int):
                self._predict_tagset[0] = predict_tagset
            elif isinstance(predict_tagset, str):
                for piece_idx, piece in enumerate(predict_tagset.split(";")):
                    if piece:
                        self._predict_tagset[piece_idx] = int(piece)
            else:
                raise TypeError("predict_tagset must be an integer or string")

        self.trainers = []
        for (model_path, pretrain_path, charlm_forward, charlm_backward) in zip(model_paths, pretrain_files, charlm_forward_files, charlm_backward_files):
            logger.debug("Loading %s with pretrain %s, forward charlm %s, backward charlm %s", model_path, pretrain_path, charlm_forward, charlm_backward)
            pretrain = pipeline.foundation_cache.load_pretrain(pretrain_path) if pretrain_path else None
            args: _NERConfig = {
                'charlm_forward_file': charlm_forward,
                'charlm_backward_file': charlm_backward,
            }

            predict_tagset = self._predict_tagset.get(len(self.trainers), None)
            if predict_tagset is not None:
                args['predict_tagset'] = predict_tagset

            try:
                trainer = Trainer(args=args, model_file=model_path, pretrain=pretrain, device=device, foundation_cache=pipeline.foundation_cache)
            except ForwardCharlmNotFoundError as e:
                raise ForwardCharlmNotFoundError("Could not find the forward charlm %s.  Please specify the correct path with ner_forward_charlm_path" % e.filename, e.filename) from None
            except BackwardCharlmNotFoundError as e:
                raise BackwardCharlmNotFoundError("Could not find the backward charlm %s.  Please specify the correct path with ner_backward_charlm_path" % e.filename, e.filename) from None
            trainer_interface: _NERTrainer = trainer
            self.trainers.append(trainer_interface)

        self._trainer = self._require_trainers()[0]
        self.model_paths = model_paths

    def _set_up_final_config(self, config: _NERConfig) -> None:
        """ Finalize the configurations for this processor, based off of values from a UD model. """
        # set configurations from loaded model
        trainers = self._require_trainers()
        self._vocab = trainers[0].vocab
        self.configs = []
        for trainer in trainers:
            loaded_args = trainer.args
            # filter out unneeded args from model
            loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
            loaded_args.update(config)
            self.configs.append(loaded_args)
        self._config = self.configs[0]

    def __str__(self) -> str:
        return "NERProcessor(%s)" % ";".join(self.model_paths)

    def mark_inactive(self) -> None:
        """ Drop memory intensive resources if keeping this processor around for reasons other than running it. """
        super().mark_inactive()
        self.trainers = None

    def process(self, document: Document) -> Document:
        trainers = self._require_trainers()
        with torch.no_grad():
            all_preds: list[list[list[str]]] = []
            for trainer, config in zip(trainers, self.configs):
                # set up a eval-only data loader and skip tag preprocessing
                batch_size = config.get('batch_size')
                if not isinstance(batch_size, int):
                    raise TypeError("NER batch_size must be an integer")
                batch = DataLoader(
                    document,
                    batch_size,
                    config,
                    vocab=trainer.vocab,
                    evaluation=True,
                    preprocess_tags=False,
                    bert_tokenizer=self._get_bert_tokenizer(trainer.model),
                )
                preds: list[list[str]] = []
                ner_batch: _NERBatch
                for ner_batch in batch:
                    batch_preds = trainer.predict(ner_batch)
                    preds.extend(batch_preds)
                all_preds.append(preds)
        # for each sentence, gather a list of predictions
        # merge those predictions into a single list
        # earlier models will have precedence
        preds = [
            merge_tags(sentence_predictions[0], *sentence_predictions[1:])
            for sentence_predictions in zip(*all_preds)
            if sentence_predictions
        ]
        ner_tags = [tag for sentence in preds for tag in sentence]
        multi_ner_tags = [
            tuple(token_predictions)
            for sentence_predictions in zip(*all_preds)
            for token_predictions in zip(*sentence_predictions)
        ]
        # collect entities into document attribute
        total = _write_predictions(document, ner_tags, multi_ner_tags)
        logger.debug(f'{total} entities found in document.')
        return document

    def bulk_process(self, docs: list[Document]) -> list[Document]:
        """
        NER processor has a collation step after running inference
        """
        docs = super().bulk_process(docs)
        for doc in docs:
            doc.build_ents()
        return docs

    def get_known_tags(self, model_idx: int = 0) -> list[str]:
        """
        Return the tags known by this model

        Removes the S-, B-, etc, and does not include O
        Specify model_idx if the processor  has more than one model
        """
        return self._require_trainers()[model_idx].get_known_tags()
