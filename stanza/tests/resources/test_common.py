"""
Test various resource downloading functions from resources/common.py
"""

import pytest
import tempfile

import stanza
from stanza.resources.common import process_pipeline_parameters

pytestmark = [pytest.mark.travis, pytest.mark.client]


def test_download_tokenize_mwt():
    with tempfile.TemporaryDirectory(dir=".") as test_dir:
        stanza.download("en", model_dir=test_dir, processors="tokenize", package="ewt", verbose=False)
        pipeline = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package="ewt")
        assert isinstance(pipeline, stanza.Pipeline)
        # mwt should be added to the list
        assert len(pipeline.loaded_processors) == 2

def test_process_pipeline_parameters():
    """
    Test a few options for specifying which processors to load
    """
    with tempfile.TemporaryDirectory(dir=".") as test_dir:
        lang, model_dir, package, processors = process_pipeline_parameters("en", test_dir, None, "tokenize,pos")
        assert processors == {"tokenize": "default", "pos": "default"}
        assert package == None

        lang, model_dir, package, processors = process_pipeline_parameters("en", test_dir, {"tokenize": "spacy"}, "tokenize,pos")
        assert processors == {"tokenize": "spacy", "pos": "default"}
        assert package == None

        lang, model_dir, package, processors = process_pipeline_parameters("en", test_dir, {"pos": "ewt"}, "tokenize,pos")
        assert processors == {"tokenize": "default", "pos": "ewt"}
        assert package == None

        lang, model_dir, package, processors = process_pipeline_parameters("en", test_dir, "ewt", "tokenize,pos")
        assert processors == {"tokenize": "ewt", "pos": "ewt"}
        assert package == None

