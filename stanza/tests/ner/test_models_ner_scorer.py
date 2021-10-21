"""
Simple test of the scorer module for NER
"""

import pytest
import stanza

from stanza.tests import *
from stanza.models.ner.scorer import score_by_token, score_by_entity

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_ner_scorer():
    pred_sequences = [['O', 'S-LOC', 'O', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'O', 'E-ORG', 'O', 'B-PER', 'I-PER', 'E-PER']]
    gold_sequences = [['O', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'B-ORG', 'E-ORG', 'O', 'B-PER', 'E-PER', 'S-LOC']]
    
    token_p, token_r, token_f, confusion = score_by_token(pred_sequences, gold_sequences)
    assert pytest.approx(token_p, abs=0.00001) == 0.625
    assert pytest.approx(token_r, abs=0.00001) == 0.5
    assert pytest.approx(token_f, abs=0.00001) == 0.55555

    entity_p, entity_r, entity_f = score_by_entity(pred_sequences, gold_sequences)
    assert pytest.approx(entity_p, abs=0.00001) == 0.4
    assert pytest.approx(entity_r, abs=0.00001) == 0.33333
    assert pytest.approx(entity_f, abs=0.00001) == 0.36363
