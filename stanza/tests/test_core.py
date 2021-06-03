import pytest
import stanza

from stanza.tests import *

from stanza.pipeline import core

pytestmark = pytest.mark.pipeline

def test_pretagged():
    """
    Test that the pipeline does or doesn't build if pos is left out and pretagged is specified
    """
    nlp = stanza.Pipeline(lang='en', dir=TEST_MODELS_DIR, processors="tokenize,pos,lemma,depparse")
    with pytest.raises(core.PipelineRequirementsException):
        nlp = stanza.Pipeline(lang='en', dir=TEST_MODELS_DIR, processors="tokenize,lemma,depparse")
    nlp = stanza.Pipeline(lang='en', dir=TEST_MODELS_DIR, processors="tokenize,lemma,depparse", depparse_pretagged=True)
    nlp = stanza.Pipeline(lang='en', dir=TEST_MODELS_DIR, processors="tokenize,lemma,depparse", pretagged=True)
    # test that the module specific flag overrides the general flag
    nlp = stanza.Pipeline(lang='en', dir=TEST_MODELS_DIR, processors="tokenize,lemma,depparse", depparse_pretagged=True, pretagged=False)
