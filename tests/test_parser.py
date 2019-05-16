"""
Basic testing of dependency parser
"""

import stanfordnlp

from tests import *


EN_DOC = "Joe Smith lives in California."

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
        **{'processors': 'tokenize,pos,lemma,depparse', 'models_dir': TEST_MODELS_DIR, 'lang': 'en'})
    doc = nlp(EN_DOC)
    assert EN_DOC_GOLD == '\n\n'.join([sent.dependencies_string() for sent in doc.sentences])
