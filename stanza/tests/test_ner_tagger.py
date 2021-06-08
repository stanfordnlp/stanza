"""
Basic testing of the NER tagger.
"""

import pytest
import stanza

from stanza.tests import *
from stanza.models.ner.scorer import score_by_token, score_by_entity

pytestmark = pytest.mark.pipeline

EN_DOC = "Chris Manning is a good man. He works in Stanford University."

EN_DOC_GOLD = """
<Span text=Chris Manning;type=PERSON;start_char=0;end_char=13>
<Span text=Stanford University;type=ORG;start_char=41;end_char=60>
""".strip()


def test_ner():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,ner', 'dir': TEST_MODELS_DIR, 'lang': 'en', 'logging_level': 'error'})
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD == '\n'.join([ent.pretty_print() for ent in doc.ents])


def test_ner_scorer():
    pred_sequences = [['O', 'S-LOC', 'O', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'O', 'E-ORG', 'O', 'B-PER', 'I-PER', 'E-PER']]
    gold_sequences = [['O', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'B-ORG', 'E-ORG', 'O', 'B-PER', 'E-PER', 'S-LOC']]
    
    token_p, token_r, token_f = score_by_token(pred_sequences, gold_sequences)
    assert pytest.approx(token_p, abs=0.00001) == 0.625
    assert pytest.approx(token_r, abs=0.00001) == 0.5
    assert pytest.approx(token_f, abs=0.00001) == 0.55555

    entity_p, entity_r, entity_f = score_by_entity(pred_sequences, gold_sequences)
    assert pytest.approx(entity_p, abs=0.00001) == 0.4
    assert pytest.approx(entity_r, abs=0.00001) == 0.33333
    assert pytest.approx(entity_f, abs=0.00001) == 0.36363
