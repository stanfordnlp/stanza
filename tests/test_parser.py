"""
Basic testing of dependency parser
"""

import stanfordnlp

from tests import *

def setup_module(module):
    """Set up resources for all tests in this module"""
    safe_rm(EN_MODELS_DIR)
    stanfordnlp.download('en', resource_dir=TEST_WORKING_DIR, force=True)


def teardown_module(module):
    """Clean up resources after tests complete"""
    safe_rm(EN_MODELS_DIR)


EN_DOC = "Joe Smith was born in California."

EN_DOC_GOLD = """
('Joe', '3', 'nsubj')
('Smith', '1', 'flat')
('lives', '0', 'root')
('in', '5', 'case')
('California', '3', 'obl')
('.', '3', 'punct')
""".strip()


def test_parser():
    nlp = stanfordnlp.Pipeline(
        **{'processors': 'tokenize,pos,lemma,depparse', 'models_dir': TEST_WORKING_DIR, 'lang': 'en'})
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD == '\n\n'.join([sent.dependencies_string() for sent in doc.sentences])
