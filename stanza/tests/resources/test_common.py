"""
Test various resource downloading functions from resources/common.py
"""

import hashlib
import json
import logging
import os
import pytest
import requests
import tempfile
from unittest.mock import patch

import stanza
from stanza.resources import common
from stanza.tests import TEST_MODELS_DIR, TEST_WORKING_DIR

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


FILE_CONTENT = b"Unban mox opal!"
# MD5 of FILE_CONTENT, verified independently
FILE_CONTENT_MD5 = hashlib.md5(FILE_CONTENT).hexdigest()

NON_HF_URL = "http://nlp.stanford.edu/software/stanza/fake_model.pt"
HF_URL = "https://huggingface.co/stanfordnlp/stanza-en/resolve/v1.13.0/models/ner/fake_model.pt"


class FakeResponse:
    """
    Minimal mock of a requests.Response sufficient for download_file.
    """
    def __init__(self, content=FILE_CONTENT, status_code=200):
        self.status_code = status_code
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def iter_content(self, chunk_size=131072):
        yield self._content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error", response=self
            )


def test_parse_hf_url():
    """
    Test that _parse_hf_url correctly identifies HF URLs and extracts components,
    and returns None for non-HF URLs.
    """
    result = common._parse_hf_url(HF_URL)
    assert result == ("stanfordnlp/stanza-en", "v1.13.0", "models/ner/fake_model.pt")

    assert common._parse_hf_url(NON_HF_URL) is None
    assert common._parse_hf_url("https://github.com/stanfordnlp/stanza") is None
    assert common._parse_hf_url("") is None


def test_download_file_non_hf():
    """
    Test the raw requests path in download_file using a non-HF URL.

    requests.get is mocked so no network traffic is generated.
    Verifies that chunked content is written correctly to the destination.
    """
    with patch("stanza.resources.common.requests.get", return_value=FakeResponse()):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            dest = os.path.join(test_dir, "fake_model.pt")
            status = common.download_file(NON_HF_URL, dest, proxies=None)
            assert status == 200
            assert os.path.exists(dest)
            assert common.get_md5(dest) == FILE_CONTENT_MD5


def test_download_file_non_hf_404():
    """
    Test that download_file raises on a 404 when raise_for_status=True,
    and does not raise when raise_for_status=False (default).
    """
    with patch("stanza.resources.common.requests.get", return_value=FakeResponse(content=b"", status_code=404)):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            dest = os.path.join(test_dir, "fake_model.pt")

            # default: no exception, but status code reflects the failure
            status = common.download_file(NON_HF_URL, dest, proxies=None)
            assert status == 404

            with pytest.raises(requests.exceptions.HTTPError):
                common.download_file(NON_HF_URL, dest, proxies=None, raise_for_status=True)


def test_download_file_hf_url_with_proxies():
    """
    Test that a HF URL with proxies set falls back to the raw requests path
    rather than going through hf_hub_download.
    """
    with patch("stanza.resources.common.requests.get", return_value=FakeResponse()) as mock_get:
        with patch("stanza.resources.common.huggingface_hub.hf_hub_download") as mock_hf:
            with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
                dest = os.path.join(test_dir, "fake_model.pt")
                proxies = {"https": "http://proxy.example.com:8080"}
                status = common.download_file(HF_URL, dest, proxies=proxies)
                assert status == 200
                mock_get.assert_called_once()
                mock_hf.assert_not_called()
                assert common.get_md5(dest) == FILE_CONTENT_MD5


def test_download_tokenize_mwt():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("en", model_dir=test_dir, processors="tokenize", package="ewt", verbose=False)
        pipeline = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package="ewt")
        assert isinstance(pipeline, stanza.Pipeline)
        # mwt should be added to the list
        assert len(pipeline.loaded_processors) == 2


def _fake_request_file(url, path, *args, **kwargs):
    """
    Stand-in for request_file: creates the destination file without hitting
    the network.  resources.json is handled by writing the real resources dict
    (already loaded from TEST_MODELS_DIR) into the temp dir before download()
    is called, so this mock only needs to stub out model .pt file downloads.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()

def test_download_non_default():
    """
    Test the download path for a single file rather than the default zip.
 
    The expectation is that an NER model will also download two charlm models
    and a pretrain.  If that layout changes on purpose, this test will fail
    and will need to be updated.
 
    The real resources.json is loaded from TEST_MODELS_DIR so the dependency
    resolution reflects the actual package structure.  request_file is mocked
    so no network traffic is generated.
    """
    resources = common.load_resources_json(TEST_MODELS_DIR)
 
    with patch("stanza.resources.common.request_file", side_effect=_fake_request_file):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            # Write the real resources.json into the temp dir so download()
            # can load it without fetching from the network.
            resources_path = os.path.join(test_dir, "resources.json")
            with open(resources_path, "w", encoding="utf-8") as fh:
                json.dump(resources, fh)

            stanza.download(
                "en",
                model_dir=test_dir,
                processors="ner",
                package="ontonotes_charlm",
                verbose=False,
                download_json=False,
            )

            assert sorted(os.listdir(test_dir)) == ["en", "resources.json"]
            en_dir = os.path.join(test_dir, "en")
            en_dir_listing = sorted(os.listdir(en_dir))
            assert en_dir_listing == ["backward_charlm", "forward_charlm", "ner", "pretrain"]
            assert os.listdir(os.path.join(en_dir, "ner")) == ["ontonotes_charlm.pt"]
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
    resources = common.load_resources_json(TEST_MODELS_DIR)

    with patch("stanza.resources.common.request_file", side_effect=_fake_request_file):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            resources_path = os.path.join(test_dir, "resources.json")
            with open(resources_path, "w", encoding="utf-8") as fh:
                json.dump(resources, fh)

            stanza.download("en", model_dir=test_dir, processors="ner", package={"ner": ["ontonotes_charlm", "anatem"]}, verbose=False, download_json=False)
            assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
            en_dir = os.path.join(test_dir, 'en')
            en_dir_listing = sorted(os.listdir(en_dir))
            assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'ner', 'pretrain']
            assert sorted(os.listdir(os.path.join(en_dir, 'ner'))) == ['anatem.pt', 'ontonotes_charlm.pt']
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

def test_language_resources():
    resources = common.load_resources_json(TEST_MODELS_DIR)

    # check that an unknown language comes back as None
    bad_lang = 'z'
    while bad_lang in resources and len(bad_lang) < 100:
        bad_lang = bad_lang + 'z'
    assert bad_lang not in resources
    assert common.get_language_resources(resources, bad_lang) == None

    # check the parameters of the test make sense
    # there should be 'zh' which is an alias of 'zh-hans'
    assert "zh" in resources
    assert "alias" in resources["zh"]
    assert resources["zh"]["alias"] == "zh-hans"

    # check that getting the resources for either 'zh' or 'zh-hans'
    # return the simplified Chinese resources
    zh_resources = common.get_language_resources(resources, "zh")
    assert "tokenize" in zh_resources
    assert "alias" not in zh_resources
    assert "Chinese" in zh_resources["lang_name"]

    zh_hans_resources = common.get_language_resources(resources, "zh-hans")
    assert zh_resources == zh_hans_resources


def test_download_restores_logging_level(tmp_path, monkeypatch):
    """download() must temporarily change the logger level, then restore it."""
    stanza.logger.setLevel(logging.WARNING)
    observed_level_during = []

    original_load = common.load_resources_json
    def capturing_load(model_dir):
        observed_level_during.append(stanza.logger.level)
        return original_load(model_dir)

    monkeypatch.setattr(common, 'load_resources_json', capturing_load)

    common.download('en', model_dir=TEST_MODELS_DIR, logging_level='DEBUG', processors=['tokenize'], download_json=False)

    # Level was actually changed to DEBUG during the call
    assert observed_level_during == [logging.DEBUG], (
        f"Expected DEBUG ({logging.DEBUG}) during download, got {observed_level_during}"
    )
    # And restored to WARNING afterwards
    assert stanza.logger.level == logging.WARNING, (
        f"Expected WARNING ({logging.WARNING}) after download, got {stanza.logger.level}"
    )
