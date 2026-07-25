"""
Processor for performing lemmatization
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import compress
from typing import (
    ClassVar,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypedDict,
    Union,
)

import torch

from stanza.models.common.doc import (
    Document,
    LEMMA as LEMMA_FIELD,
    TEXT,
    UPOS,
)
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.lemma.data import DataLoader
from stanza.models.lemma.trainer import Trainer
from stanza.models.lemma.vocab import MultiVocab
from stanza.pipeline._constants import LEMMA, POS, TOKENIZE
from stanza.pipeline.processor import (
    ProcessorDevice,
    UDProcessor,
    register_processor,
)

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline

_ModelPath = Union[str, os.PathLike[str]]
_WordTag = tuple[str, Optional[str]]
_DictionaryEntry = tuple[str, Optional[str], str]
_LemmaBatchItem = Union[torch.Tensor, Sequence[int], Sequence[str]]

# Kept public for compatibility with callers which use the field selection.
WORD_TAGS = [TEXT, UPOS]


class _LemmaConfig(TypedDict, total=False):
    backward_charlm_path: Optional[_ModelPath]
    batch_size: int
    beam_size: int
    caseless: bool
    dict_only: bool
    ensemble_dict: bool
    forward_charlm_path: Optional[_ModelPath]
    model_path: _ModelPath
    pos: Union[bool, str]
    pretagged: Union[bool, str]
    pretrain_path: Optional[_ModelPath]
    sample_train: float
    skip_blank_lemmas: bool
    store_results: Union[bool, str]
    use_identity: Union[bool, str]


class _LemmaBatch(NamedTuple):
    source: torch.Tensor
    source_mask: torch.Tensor
    target_input: torch.Tensor
    target_output: torch.Tensor
    pos: torch.Tensor
    edits: torch.Tensor
    original_indices: Sequence[int]
    text: Sequence[str]


class _LoadedLemmaData(NamedTuple):
    document: Document
    vocab: MultiVocab
    batches: Iterator[Sequence[_LemmaBatchItem]]


class _RawLemmaDataLoader(Protocol):
    @property
    def doc(self) -> Document:
        ...

    @property
    def vocab(
            self,
        ) -> Union[MultiVocab, Mapping[str, str]]:
        ...

    def __iter__(self) -> Iterator[Sequence[_LemmaBatchItem]]:
        ...


class _LemmaDocumentWriter(Protocol):
    def set(
            self,
            fields: Sequence[str],
            contents: Sequence[str],
        ) -> None:
        ...


class _LemmaTrainer(Protocol):
    @property
    def pos_dict(
            self,
        ) -> Mapping[Optional[str], Mapping[str, str]]:
        ...

    def predict(
            self,
            batch: _LemmaBatch,
            beam_size: int = 1,
            vocab: Optional[MultiVocab] = None,
        ) -> tuple[Sequence[Optional[str]], Optional[Sequence[int]]]:
        ...

    def postprocess(
            self,
            words: Sequence[str],
            preds: Sequence[str],
            edits: Optional[Sequence[int]] = None,
        ) -> Sequence[Optional[str]]:
        ...

    def has_contextual_lemmatizers(self) -> bool:
        ...

    def update_contextual_preds(
            self,
            doc: Document,
            preds: Sequence[str],
        ) -> Sequence[Optional[str]]:
        ...

    def train_dict(
            self,
            triples: Sequence[_DictionaryEntry],
            update_word_dict: bool = True,
        ) -> None:
        ...

    def predict_dict(
            self,
            pairs: Sequence[_WordTag],
        ) -> Sequence[Optional[str]]:
        ...

    def skip_seq2seq(
            self,
            pairs: Sequence[_WordTag],
        ) -> Sequence[bool]:
        ...

    def ensemble(
            self,
            pairs: Sequence[_WordTag],
            other_preds: Sequence[str],
        ) -> Sequence[Optional[str]]:
        ...


class _LemmaTrainerFactory(Protocol):
    def __call__(
            self,
            *,
            args: Mapping[str, Optional[_ModelPath]],
            model_file: _ModelPath,
            device: Optional[ProcessorDevice],
            foundation_cache: FoundationCache,
            lemma_classifier_args: Mapping[str, Optional[_ModelPath]],
        ) -> _LemmaTrainer:
        ...


def _normalize_batch(
        raw_batch: Sequence[_LemmaBatchItem],
    ) -> _LemmaBatch:
    if len(raw_batch) != 8:
        raise TypeError("Lemma batches must contain eight fields")

    def require_tensor(
            value: _LemmaBatchItem,
            name: str,
        ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"The lemma {name} batch field must be a tensor")
        return value

    def require_ints(
            value: _LemmaBatchItem,
        ) -> tuple[int, ...]:
        if isinstance(value, torch.Tensor) or isinstance(value, (str, bytes)):
            raise TypeError("Lemma batch indices must be a sequence of integers")
        indices: list[int] = []
        for index in value:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    "Lemma batch indices must be a sequence of integers"
                )
            indices.append(index)
        return tuple(indices)

    def require_strings(
            value: _LemmaBatchItem,
        ) -> tuple[str, ...]:
        if isinstance(value, torch.Tensor) or isinstance(value, (str, bytes)):
            raise TypeError("Lemma batch text must be a sequence of strings")
        values: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    "Lemma batch text must be a sequence of strings"
                )
            values.append(item)
        return tuple(values)

    return _LemmaBatch(
        require_tensor(raw_batch[0], "source"),
        require_tensor(raw_batch[1], "source mask"),
        require_tensor(raw_batch[2], "target input"),
        require_tensor(raw_batch[3], "target output"),
        require_tensor(raw_batch[4], "POS"),
        require_tensor(raw_batch[5], "edits"),
        require_ints(raw_batch[6]),
        require_strings(raw_batch[7]),
    )


def _normalize_predictions(
        raw_predictions: Sequence[Optional[str]],
        source: str,
    ) -> list[str]:
    predictions: list[str] = []
    for prediction in raw_predictions:
        if not isinstance(prediction, str):
            raise TypeError(f"{source} lemma predictions must be strings")
        predictions.append(prediction)
    return predictions


def _load_lemma_data(
        loader: _RawLemmaDataLoader,
    ) -> _LoadedLemmaData:
    document = loader.doc
    vocab = loader.vocab
    if not isinstance(vocab, MultiVocab):
        raise TypeError("The lemma data loader did not contain a MultiVocab")
    batches: Iterator[Sequence[_LemmaBatchItem]] = iter(loader)
    return _LoadedLemmaData(document, vocab, batches)


def _set_lemmas(
        document: _LemmaDocumentWriter,
        predictions: Sequence[str],
    ) -> None:
    document.set([LEMMA_FIELD], predictions)


@register_processor(name=LEMMA)
class LemmaProcessor(UDProcessor):
    _config: _LemmaConfig
    _pipeline: Pipeline
    _pretagged: bool = False
    _requires: set[str]
    _trainer: Optional[_LemmaTrainer]
    _use_identity: Optional[bool] = None
    store_results: bool = False

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT: ClassVar[set[str]] = set([LEMMA])
    # set of processor requirements for this processor
    # pos will be added later for non-identity lemmatizerx
    REQUIRES_DEFAULT: ClassVar[set[str]] = set([TOKENIZE])
    # default batch size
    DEFAULT_BATCH_SIZE: ClassVar[int] = 5000

    @property
    def use_identity(self) -> Optional[bool]:
        return self._use_identity

    @property
    def config(self) -> _LemmaConfig:
        return self._config

    def _set_up_model(
            self,
            config: _LemmaConfig,
            pipeline: Pipeline,
            device: Optional[ProcessorDevice],
        ) -> None:
        if config.get('use_identity') in ['True', True]:
            self._use_identity = True
            self.store_results = False
            self._config = config
            self._config['batch_size'] = LemmaProcessor.DEFAULT_BATCH_SIZE
        else:
            # the lemmatizer only looks at one word when making
            # decisions, not the surrounding context
            # therefore, we can save some time by remembering what
            # we did the last time we saw any given word,pos
            # since a long running program will remember everything
            # (unless we go back and make it smarter)
            # we make this an option, not the default
            # TODO: need to update the cache to skip the contextual lemmatizer
            self.store_results = bool(config.get('store_results', False))
            self._use_identity = False
            model_path = config.get('model_path')
            if model_path is None:
                raise ValueError(
                    "A lemma model path is required outside identity mode"
                )
            args: dict[str, Optional[_ModelPath]] = {
                'charlm_forward_file': config.get(
                    'forward_charlm_path',
                    None,
                ),
                'charlm_backward_file': config.get(
                    'backward_charlm_path',
                    None,
                ),
            }
            lemma_classifier_args = dict(args)
            lemma_classifier_args['wordvec_pretrain_file'] = config.get('pretrain_path', None)
            trainer_factory: _LemmaTrainerFactory = Trainer
            self._trainer = trainer_factory(
                args=args,
                model_file=model_path,
                device=device,
                foundation_cache=pipeline.foundation_cache,
                lemma_classifier_args=lemma_classifier_args,
            )

    def _set_up_requires(self) -> None:
        self._pretagged = bool(self._config.get('pretagged', None))
        if self._pretagged:
            self._requires = set()
        elif self._config.get('pos') and not self.use_identity:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def _require_trainer(self) -> _LemmaTrainer:
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("The lemma processor model is not loaded")
        return trainer

    def _require_vocab(self) -> MultiVocab:
        vocab = self._vocab
        if not isinstance(vocab, MultiVocab):
            raise RuntimeError("The lemma processor vocabulary is not loaded")
        return vocab

    def _model_state(self) -> tuple[_LemmaTrainer, MultiVocab]:
        return self._require_trainer(), self._require_vocab()

    def process(self, document: Document) -> Document:
        batch_size = self._config.get('batch_size')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("The lemma batch size must be an integer")

        use_identity = self.use_identity
        if use_identity:
            batch = DataLoader(document, batch_size, self._config, evaluation=True, conll_only=True)
            batch_document = batch.doc
            preds: list[str] = [word.text for sent in batch_document.sentences for word in sent.words]
        else:
            trainer, vocab = self._model_state()
            batch = _load_lemma_data(DataLoader(
                document,
                batch_size,
                self._config,
                vocab=vocab,
                evaluation=True,
                expand_unk_vocab=True,
            ))
            batch_document = batch.document
            if self._config.get('dict_only', False):
                word_tags: list[_WordTag] = [
                    (word.text, word.upos)
                    for word in batch_document.iter_words()
                ]
                preds = _normalize_predictions(
                    trainer.predict_dict(word_tags),
                    "Dictionary",
                )
            else:
                word_tags = [
                    (word.text, word.upos)
                    for word in batch_document.iter_words()
                ]
                ensemble_dictionary = bool(
                    self._config.get('ensemble_dict', False)
                )
                skip: list[bool] = []
                if ensemble_dictionary:
                    # skip the seq2seq model when we can
                    skip = list(trainer.skip_seq2seq(word_tags))
                    # although there is no explicit use of caseless or lemma_caseless in this processor,
                    # it shows up in the config which gets passed to the DataLoader,
                    # possibly affecting its results
                    seq2seq_batch = _load_lemma_data(DataLoader(
                        document,
                        batch_size,
                        self._config,
                        vocab=vocab,
                        evaluation=True,
                        skip=skip,
                        expand_unk_vocab=True,
                    ))
                else:
                    seq2seq_batch = batch

                with torch.no_grad():
                    preds = []
                    edits: list[int] = []
                    seq2seq_vocab = seq2seq_batch.vocab
                    beam_size = self._config.get('beam_size')
                    if isinstance(beam_size, bool) or not isinstance(beam_size, int):
                        raise TypeError("The lemma beam size must be an integer")
                    for raw_batch in seq2seq_batch.batches:
                        lemma_batch = _normalize_batch(raw_batch)
                        raw_predictions, raw_edits = trainer.predict(
                            lemma_batch,
                            beam_size,
                            seq2seq_vocab,
                        )
                        preds.extend(_normalize_predictions(
                            raw_predictions,
                            "Sequence-to-sequence",
                        ))
                        if raw_edits is not None:
                            edits.extend(raw_edits)

                if ensemble_dictionary:
                    words = [word for word, _ in word_tags]
                    unskipped_words = [
                        word
                        for word, skipped in zip(words, skip)
                        if not skipped
                    ]
                    preds = _normalize_predictions(
                        trainer.postprocess(
                            unskipped_words,
                            preds,
                            edits=edits,
                        ),
                        "Postprocessed",
                    )
                    if self.store_results:
                        new_word_tags = compress(
                            word_tags,
                            (not skipped for skipped in skip),
                        )
                        new_predictions = [
                            (word, tag, lemma)
                            for (word, tag), lemma
                            in zip(new_word_tags, preds)
                        ]
                        trainer.train_dict(
                            new_predictions,
                            update_word_dict=False,
                        )
                    # expand seq2seq predictions to the same size as all words
                    prediction_index = 0
                    expanded_predictions: list[str] = []
                    for skipped in skip:
                        if skipped:
                            expanded_predictions.append('')
                        else:
                            expanded_predictions.append(
                                preds[prediction_index]
                            )
                            prediction_index += 1
                    preds = _normalize_predictions(
                        trainer.ensemble(
                            word_tags,
                            expanded_predictions,
                        ),
                        "Ensembled",
                    )
                else:
                    words = [
                        word.text
                        for word in batch_document.iter_words()
                    ]
                    preds = _normalize_predictions(
                        trainer.postprocess(words, preds, edits=edits),
                        "Postprocessed",
                    )

                if trainer.has_contextual_lemmatizers():
                    preds = _normalize_predictions(
                        trainer.update_contextual_preds(
                            batch_document,
                            preds,
                        ),
                        "Contextual",
                    )

        # map empty string lemmas to '_'
        preds = [max([(len(x), x), (0, '_')])[1] for x in preds]
        _set_lemmas(batch_document, preds)
        return batch_document
