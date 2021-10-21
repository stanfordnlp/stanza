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

