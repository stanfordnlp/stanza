"""
Test various resource downloading functions from resources/common.py
"""

import pytest
import tempfile

import stanza

pytestmark = [pytest.mark.travis, pytest.mark.client]


def test_download_tokenize_mwt():
    with tempfile.TemporaryDirectory(dir=".") as test_dir:
        stanza.download("en", model_dir=test_dir, processors="tokenize", package="ewt", verbose=False)
        pipeline = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package="ewt")
        assert isinstance(pipeline, stanza.Pipeline)
        # mwt should be added to the list
        assert len(pipeline.loaded_processors) == 2
