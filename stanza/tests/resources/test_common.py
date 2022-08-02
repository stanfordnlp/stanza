"""
Test various resource downloading functions from resources/common.py
"""

import os
import pytest
import tempfile

import stanza
from stanza.resources import common
from stanza.tests import TEST_WORKING_DIR

pytestmark = [pytest.mark.travis, pytest.mark.client]

def test_assert_file_exists():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        filename = os.path.join(test_dir, "test.txt")
        with pytest.raises(FileNotFoundError):
            common.assert_file_exists(filename)

        with open(filename, "w", encoding="utf-8") as fout:
            fout.write("Unban mox opal!")
        # MD5 of the fake model file, not any real model files in the system
        EXPECTED_MD5 = "44dbf21b4e89cea5184615a72a825a36"
        common.assert_file_exists(filename)
        common.assert_file_exists(filename, md5=EXPECTED_MD5)

        with pytest.raises(ValueError):
            common.assert_file_exists(filename, md5="12345")

        with pytest.raises(ValueError):
            common.assert_file_exists(filename, md5="12345", alternate_md5="12345")

        common.assert_file_exists(filename, md5="12345", alternate_md5=EXPECTED_MD5)


def test_download_tokenize_mwt():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("en", model_dir=test_dir, processors="tokenize", package="ewt", verbose=False)
        pipeline = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package="ewt")
        assert isinstance(pipeline, stanza.Pipeline)
        # mwt should be added to the list
        assert len(pipeline.loaded_processors) == 2

def test_download_non_default():
    """
    Test the download path for a single file rather than the default zip

    The expectation is that an NER model will also download two charlm models.
    If that layout changes on purpose, this test will fail and will need to be updated
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("en", model_dir=test_dir, processors="ner", package="ontonotes", verbose=False)
        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        en_dir = os.path.join(test_dir, 'en')
        en_dir_listing = sorted(os.listdir(en_dir))
        assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'ner', 'pretrain']
        assert os.listdir(os.path.join(en_dir, 'ner')) == ['ontonotes.pt']
        for i in en_dir_listing:
            assert len(os.listdir(os.path.join(en_dir, i))) == 1


def test_download_two_models():
    """
    Test the download path for two NER models

    The package system should now allow for multiple NER models to be
    specified, and a consequence of that is it should be possible to
    download two models at once

    The expectation is that the two different NER models both download
    a different forward & backward charlm.  If that changes, the test
    will fail.  Best way to update it will be two different models
    which download two different charlms
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("en", model_dir=test_dir, processors="ner", package={"ner": ["ontonotes", "anatem"]}, verbose=False)
        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        en_dir = os.path.join(test_dir, 'en')
        en_dir_listing = sorted(os.listdir(en_dir))
        assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'ner', 'pretrain']
        assert sorted(os.listdir(os.path.join(en_dir, 'ner'))) == ['anatem.pt', 'ontonotes.pt']
        for i in en_dir_listing:
            assert len(os.listdir(os.path.join(en_dir, i))) == 2


def test_process_pipeline_parameters():
    """
    Test a few options for specifying which processors to load
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        lang, model_dir, package, processors = common.process_pipeline_parameters("en", test_dir, None, "tokenize,pos")
        assert processors == {"tokenize": "default", "pos": "default"}
        assert package == None

        lang, model_dir, package, processors = common.process_pipeline_parameters("en", test_dir, {"tokenize": "spacy"}, "tokenize,pos")
        assert processors == {"tokenize": "spacy", "pos": "default"}
        assert package == None

        lang, model_dir, package, processors = common.process_pipeline_parameters("en", test_dir, {"pos": "ewt"}, "tokenize,pos")
        assert processors == {"tokenize": "default", "pos": "ewt"}
        assert package == None

        lang, model_dir, package, processors = common.process_pipeline_parameters("en", test_dir, "ewt", "tokenize,pos")
        assert processors == {"tokenize": "ewt", "pos": "ewt"}
        assert package == None

