"""
Basic testing of part of speech tagging
"""

import pytest
import stanza
from stanza.models.common.vocab import VOCAB_PREFIX

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

class TestClassifier:
    @pytest.fixture(scope="class")
    def english_depparse(self):
        """
        Get a depparse_processor for English
        """
        nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma,depparse', 'dir': TEST_MODELS_DIR, 'lang': 'en'})
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
