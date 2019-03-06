"""
Basic testing of the English pipeline
"""

import pytest
import stanfordnlp

from tests import *


def setup_module(module):
    """Set up resources for all tests in this module"""
    safe_rm(EN_MODELS_DIR)
    stanfordnlp.download('en', resource_dir=TEST_WORKING_DIR, force=True)


def teardown_module(module):
    """Clean up resources after tests complete"""
    safe_rm(EN_MODELS_DIR)


# data for testing
EN_DOC = "Barack Obama was born in Hawaii.  He was elected president in 2008.  Obama attended Harvard."

EN_DOC_DEPENDENCY_PARSES_GOLD = """
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')

('He', '3', 'nsubj:pass')
('was', '3', 'aux:pass')
('elected', '0', 'root')
('president', '3', 'obj')
('in', '6', 'case')
('2008', '3', 'obl')
('.', '3', 'punct')

('Obama', '2', 'nsubj')
('attended', '0', 'root')
('Harvard', '2', 'obj')
('.', '2', 'punct')
""".strip()


@pytest.fixture(scope="module")
def processed_doc():
    """ Document created by running full English pipeline on a few sentences """
    nlp = stanfordnlp.Pipeline(models_dir=TEST_WORKING_DIR)
    return nlp(EN_DOC)


def test_dependency_parse(processed_doc):
    assert "\n\n".join([sent.dependencies_string() for sent in processed_doc.sentences]) == EN_DOC_DEPENDENCY_PARSES_GOLD
