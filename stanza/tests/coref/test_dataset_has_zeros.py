"""
Test the check for whether a coref training set has any zero anaphora
"""

import pytest

from stanza.models.coref.model import dataset_has_zeros

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_no_is_zero_field():
    # OntoNotes and the other non-UD converters don't write is_zero
    docs = [{"document_id": "a"}, {"document_id": "b"}]
    assert not dataset_has_zeros(docs)

def test_all_false():
    # convert_udcoref.py writes is_zero for every document, even with no zeros
    docs = [{"is_zero": [False, False]}, {"is_zero": []}, {"is_zero": [False]}]
    assert not dataset_has_zeros(docs)

def test_some_true():
    docs = [{"is_zero": [False, False]}, {"document_id": "b"}, {"is_zero": [False, True]}]
    assert dataset_has_zeros(docs)

def test_empty_dataset():
    assert not dataset_has_zeros([])
