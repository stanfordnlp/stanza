"""
Basic testing of part of speech tagging
"""

import numpy as np
import pytest
import stanza
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.depparse.trainer import GraphTrainer, TransitionTrainer
from stanza.models.pos.vocab import MultiVocab, WordVocab
from stanza.pipeline.depparse_processor import (
    DepparseProcessor,
    _flatten_document_predictions,
    _normalize_predictions,
)

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


def test_depparse_model_state_validation_without_model():
    processor = DepparseProcessor.__new__(DepparseProcessor)
    processor._trainer = None
    processor._pretrain = None
    processor._vocab = None

    with pytest.raises(RuntimeError, match="model is not loaded"):
        processor._require_trainer()
    with pytest.raises(RuntimeError, match="vocabulary is not loaded"):
        processor.get_known_relations()


def test_depparse_accepts_injected_trainers_without_model():
    for trainer_class in (GraphTrainer, TransitionTrainer):
        configured_trainer = trainer_class.__new__(trainer_class)
        configured_trainer.args = {}
        configured_trainer.vocab = MultiVocab({})
        processor = DepparseProcessor.__new__(DepparseProcessor)
        processor._trainer = None

        processor._set_up_model(
            {"trainer": configured_trainer},
            pipeline=None,
            device=None,
        )

        assert processor._require_trainer() is configured_trainer


def test_depparse_accepts_structural_trainer_without_model():
    class StructuralTrainer:
        def __init__(self):
            self.args = {}
            self.vocab = MultiVocab({})

        def predict(
                self,
                batch,
                unsort=True,
                resolve_head_constraints=False,
            ):
            return []

    configured_trainer = StructuralTrainer()
    processor = DepparseProcessor.__new__(DepparseProcessor)
    processor._trainer = None

    processor._set_up_model(
        {"trainer": configured_trainer},
        pipeline=None,
        device=None,
    )

    assert processor._require_trainer() is configured_trainer


def test_depparse_known_relations_without_model():
    processor = DepparseProcessor.__new__(DepparseProcessor)
    deprel_vocab = WordVocab([[("case",), ("root",)]])
    processor._vocab = MultiVocab({"deprel": deprel_vocab})

    assert processor.get_known_relations() == ["case", "root"]


def test_depparse_prediction_normalization_without_model():
    raw_predictions = [[(np.int64(0), "root"), (1, "case")]]

    assert _normalize_predictions(raw_predictions) == [
        [(0, "root"), (1, "case")]
    ]


def test_depparse_prediction_dimensions_without_model():
    document = stanza.Document([[{"text": "one"}, {"text": "two"}]])

    assert _flatten_document_predictions(
        document,
        [[(0, "root"), (1, "dep")]],
    ) == [(0, "root"), (1, "dep")]

    with pytest.raises(ValueError, match="one result per sentence"):
        _flatten_document_predictions(document, [])
    with pytest.raises(ValueError, match="2 words"):
        _flatten_document_predictions(document, [[(0, "root")]])
    with pytest.raises(ValueError, match="outside"):
        _flatten_document_predictions(
            document,
            [[(0, "root"), (3, "dep")]],
        )


class TestClassifier:
    @pytest.fixture(scope="class")
    def english_depparse(self):
        """
        Get a depparse_processor for English
        """
        nlp = stanza.Pipeline(
            processors='tokenize,pos,lemma,depparse',
            dir=TEST_MODELS_DIR,
            lang='en',
            download_method=None,
        )
        assert 'depparse' in nlp.processors
        return nlp.processors['depparse']

    def test_get_known_relations(self, english_depparse):
        """
        Test getting the known relations from a processor.

        Doesn't test that all the relations exist, since who knows what will change in the future
        """
        relations = english_depparse.get_known_relations()
        assert len(relations) > 5
        assert 'case' in relations
        for i in VOCAB_PREFIX:
            assert i not in relations
