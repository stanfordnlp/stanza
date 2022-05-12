import pytest
import tempfile

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

def test_download_missing_ner_model():
    """
    Test that the pipeline will automatically download missing models
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("en", model_dir=test_dir, processors="tokenize", package="combined", verbose=False)
        pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize,ner", package={"ner": ("ontonotes")})

        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        en_dir = os.path.join(test_dir, 'en')
        en_dir_listing = sorted(os.listdir(en_dir))
        assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'ner', 'tokenize']
        assert os.listdir(os.path.join(en_dir, 'ner')) == ['ontonotes.pt']


def test_download_missing_resources():
    """
    Test that the pipeline will automatically download missing models
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize,ner", package={"tokenize": "combined", "ner": "ontonotes"})

        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        en_dir = os.path.join(test_dir, 'en')
        en_dir_listing = sorted(os.listdir(en_dir))
        assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'ner', 'tokenize']
        assert os.listdir(os.path.join(en_dir, 'ner')) == ['ontonotes.pt']


def test_download_resources_overwrites():
    """
    Test that the DOWNLOAD_RESOURCES method overwrites an existing resources.json
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"})

        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        resources_path = os.path.join(test_dir, 'resources.json')
        mod_time = os.path.getmtime(resources_path)

        pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"})
        new_mod_time = os.path.getmtime(resources_path)
        assert mod_time != new_mod_time

def test_reuse_resources_overwrites():
    """
    Test that the REUSE_RESOURCES method does *not* overwrite an existing resources.json
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        pipe = stanza.Pipeline("en",
                               download_method=core.DownloadMethod.REUSE_RESOURCES,
                               model_dir=test_dir,
                               processors="tokenize",
                               package={"tokenize": "combined"})

        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        resources_path = os.path.join(test_dir, 'resources.json')
        mod_time = os.path.getmtime(resources_path)

        pipe = stanza.Pipeline("en",
                               download_method=core.DownloadMethod.REUSE_RESOURCES,
                               model_dir=test_dir,
                               processors="tokenize",
                               package={"tokenize": "combined"})
        new_mod_time = os.path.getmtime(resources_path)
        assert mod_time == new_mod_time


def test_download_not_repeated():
    """
    Test that a model is only downloaded once if it already matches the expected model from the resources file
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("en", model_dir=test_dir, processors="tokenize", package="combined")

        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        en_dir = os.path.join(test_dir, 'en')
        en_dir_listing = sorted(os.listdir(en_dir))
        assert en_dir_listing == ['tokenize']
        tokenize_path = os.path.join(en_dir, "tokenize", "combined.pt")
        mod_time = os.path.getmtime(tokenize_path)

        pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"})
        assert os.path.getmtime(tokenize_path) == mod_time



def test_download_fixed():
    """
    Test that a model is fixed if the existing model doesn't match the md5sum
    """
    for download_method in (core.DownloadMethod.REUSE_RESOURCES, core.DownloadMethod.DOWNLOAD_RESOURCES):
        with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
            stanza.download("en", model_dir=test_dir, processors="tokenize", package="combined")

            assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
            en_dir = os.path.join(test_dir, 'en')
            en_dir_listing = sorted(os.listdir(en_dir))
            assert en_dir_listing == ['tokenize']
            tokenize_path = os.path.join(en_dir, "tokenize", "combined.pt")

            with open(tokenize_path, "w") as fout:
                fout.write("Unban mox opal!")
            mod_time = os.path.getmtime(tokenize_path)

            pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"}, download_method=download_method)
            assert os.path.getmtime(tokenize_path) != mod_time

def test_limited_pipeline():
    """
    Test loading a pipeline, but then only using a couple processors
    """
    pipe = stanza.Pipeline(processors="tokenize,pos,lemma,depparse,ner", dir=TEST_MODELS_DIR)
    doc = pipe("John Bauer works at Stanford")
    assert all(word.upos is not None for sentence in doc.sentences for word in sentence.words)
    assert all(token.ner is not None for sentence in doc.sentences for token in sentence.tokens)

    doc = pipe("John Bauer works at Stanford", processors=["tokenize","pos"])
    assert all(word.upos is not None for sentence in doc.sentences for word in sentence.words)
    assert not any(token.ner is not None for sentence in doc.sentences for token in sentence.tokens)

    doc = pipe("John Bauer works at Stanford", processors="tokenize")
    assert not any(word.upos is not None for sentence in doc.sentences for word in sentence.words)
    assert not any(token.ner is not None for sentence in doc.sentences for token in sentence.tokens)

    doc = pipe("John Bauer works at Stanford", processors="tokenize,ner")
    assert not any(word.upos is not None for sentence in doc.sentences for word in sentence.words)
    assert all(token.ner is not None for sentence in doc.sentences for token in sentence.tokens)

    with pytest.raises(ValueError):
        # this should fail
        doc = pipe("John Bauer works at Stanford", processors="tokenize,depparse")
