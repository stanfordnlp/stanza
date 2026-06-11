import pytest
import shutil
import tempfile
from unittest.mock import patch

import stanza

from stanza.tests import *

from stanza.pipeline import core
from stanza.resources.common import get_md5, load_resources_json

pytestmark = pytest.mark.pipeline

def _fake_request_file(url, path, *args, **kwargs):
    """
    Stand-in for request_file: creates or touches the destination file without
    hitting the network.  This satisfies both "file exists" checks and mtime
    comparisons (the file is always re-written, so the mtime always advances).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith("resources.json"):
        # Copy the real resources.json so load_resources_json can parse it,
        # but open for write so the mtime is updated on every call.
        real_path = os.path.join(TEST_MODELS_DIR, "resources.json")
        with open(real_path, encoding="utf-8") as fin:
            data = fin.read()
        with open(path, "w", encoding="utf-8") as fout:
            fout.write(data)
    else:
        open(path, "wb").close()


def _seed_from_models_dir(test_dir, lang, model_rel_paths):
    """
    Populate test_dir with the real resources.json and copies of the real
    model files listed in model_rel_paths (e.g. ['tokenize/combined.pt',
    'mwt/combined.pt']).  This lets Pipeline load processors without any
    network access.

    We use shutil.copy2 rather than os.symlink because symlinks require
    elevated privileges on Windows.
    """
    real_resources = os.path.join(TEST_MODELS_DIR, "resources.json")
    shutil.copy(real_resources, os.path.join(test_dir, "resources.json"))
    for rel_path in model_rel_paths:
        src = os.path.join(TEST_MODELS_DIR, lang, rel_path)
        dst = os.path.join(test_dir, lang, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


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
    Test that the pipeline will automatically download missing models.

    Tokenize is pre-seeded (already "downloaded"); ner, charlm, and pretrain
    are absent and should be fetched automatically by Pipeline.  request_file
    is mocked so no network traffic is generated.  All models that Pipeline
    will load are restored from TEST_MODELS_DIR so torch can open them.
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        # Tokenize was already present; the rest are "downloaded" by Pipeline.
        # All of them need real content because Pipeline loads every processor.
        model_rel_paths = [
            "tokenize/combined.pt",
            "mwt/combined.pt",
            "ner/ontonotes-ww-multi_charlm.pt",
            "forward_charlm/1billion.pt",
            "backward_charlm/1billion.pt",
            "pretrain/conll17.pt",
        ]
        real_files = {
            os.path.join(test_dir, "en", rel): os.path.join(TEST_MODELS_DIR, "en", rel)
            for rel in model_rel_paths
        }

        def fake_request_file_with_restore(url, path, *args, **kwargs):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith("resources.json"):
                real_path = os.path.join(TEST_MODELS_DIR, "resources.json")
                with open(real_path, encoding="utf-8") as fin:
                    data = fin.read()
                with open(path, "w", encoding="utf-8") as fout:
                    fout.write(data)
            elif path in real_files:
                shutil.copy2(real_files[path], path)
            else:
                open(path, "wb").close()

        _seed_from_models_dir(test_dir, "en", ["tokenize/combined.pt", "mwt/combined.pt"])

        with patch("stanza.resources.common.request_file", side_effect=fake_request_file_with_restore):
            with patch("stanza.pipeline.core.download_resources_json", side_effect=lambda *a, **kw: fake_request_file_with_restore(None, os.path.join(a[0], "resources.json"))):
                pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize,ner", package={"ner": ("ontonotes-ww-multi_charlm")})

                assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
                en_dir = os.path.join(test_dir, 'en')
                en_dir_listing = sorted(os.listdir(en_dir))
                assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'mwt', 'ner', 'pretrain', 'tokenize']
                assert os.listdir(os.path.join(en_dir, 'ner')) == ['ontonotes-ww-multi_charlm.pt']


def test_download_missing_resources():
    """
    Test that the pipeline will automatically download missing models.

    No models are pre-seeded; everything including resources.json must be
    fetched by Pipeline.  request_file is mocked so no network traffic is
    generated.  All models that Pipeline will load are restored from
    TEST_MODELS_DIR so torch can open them.
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        model_rel_paths = [
            "tokenize/combined.pt",
            "mwt/combined.pt",
            "ner/ontonotes-ww-multi_charlm.pt",
            "forward_charlm/1billion.pt",
            "backward_charlm/1billion.pt",
            "pretrain/conll17.pt",
        ]
        real_files = {
            os.path.join(test_dir, "en", rel): os.path.join(TEST_MODELS_DIR, "en", rel)
            for rel in model_rel_paths
        }

        def fake_request_file_with_restore(url, path, *args, **kwargs):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith("resources.json"):
                real_path = os.path.join(TEST_MODELS_DIR, "resources.json")
                with open(real_path, encoding="utf-8") as fin:
                    data = fin.read()
                with open(path, "w", encoding="utf-8") as fout:
                    fout.write(data)
            elif path in real_files:
                shutil.copy2(real_files[path], path)
            else:
                open(path, "wb").close()

        with patch("stanza.resources.common.request_file", side_effect=fake_request_file_with_restore):
            with patch("stanza.pipeline.core.download_resources_json", side_effect=lambda *a, **kw: fake_request_file_with_restore(None, os.path.join(a[0], "resources.json"))):
                pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize,ner", package={"tokenize": "combined", "ner": "ontonotes-ww-multi_charlm"})

                assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
                en_dir = os.path.join(test_dir, 'en')
                en_dir_listing = sorted(os.listdir(en_dir))
                assert en_dir_listing == ['backward_charlm', 'forward_charlm', 'mwt', 'ner', 'pretrain', 'tokenize']
                assert os.listdir(os.path.join(en_dir, 'ner')) == ['ontonotes-ww-multi_charlm.pt']


def test_download_resources_overwrites():
    """
    Test that the DOWNLOAD_RESOURCES method overwrites an existing resources.json
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        model_rel_paths = ["tokenize/combined.pt", "mwt/combined.pt"]
        _seed_from_models_dir(test_dir, "en", model_rel_paths)

        real_files = {
            os.path.join(test_dir, "en", rel): os.path.join(TEST_MODELS_DIR, "en", rel)
            for rel in model_rel_paths
        }

        def fake_request_file_with_restore(url, path, *args, **kwargs):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith("resources.json"):
                real_path = os.path.join(TEST_MODELS_DIR, "resources.json")
                with open(real_path, encoding="utf-8") as fin:
                    data = fin.read()
                with open(path, "w", encoding="utf-8") as fout:
                    fout.write(data)
            elif path in real_files:
                shutil.copy2(real_files[path], path)
            else:
                open(path, "wb").close()

        with patch("stanza.resources.common.request_file", side_effect=fake_request_file_with_restore):
            with patch("stanza.pipeline.core.download_resources_json", side_effect=lambda *a, **kw: fake_request_file_with_restore(None, os.path.join(a[0], "resources.json"))):
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
        assert en_dir_listing == ['mwt', 'tokenize']
        tokenize_path = os.path.join(en_dir, "tokenize", "combined.pt")
        mod_time = os.path.getmtime(tokenize_path)

        pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"})
        assert os.path.getmtime(tokenize_path) == mod_time

def test_download_none():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        stanza.download("it", model_dir=test_dir, processors="tokenize", package="combined")
        stanza.download("it", model_dir=test_dir, processors="tokenize", package="vit")

        it_dir = os.path.join(test_dir, 'it')
        it_dir_listing = sorted(os.listdir(it_dir))
        assert sorted(it_dir_listing) == ['mwt', 'tokenize']
        combined_path = os.path.join(it_dir, "tokenize", "combined_nocharlm.pt")
        vit_path = os.path.join(it_dir, "tokenize", "vit.pt")

        assert os.path.exists(combined_path)
        assert os.path.exists(vit_path)

        combined_md5 = get_md5(combined_path)
        vit_md5 = get_md5(vit_path)
        # check that the models are different
        # otherwise the test is not testing anything
        assert combined_md5 != vit_md5

        shutil.copyfile(vit_path, combined_path)
        assert get_md5(combined_path) == vit_md5

        pipe = stanza.Pipeline("it", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"}, download_method=None)
        assert get_md5(combined_path) == vit_md5

        pipe = stanza.Pipeline("it", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"})
        assert get_md5(combined_path) != vit_md5


def check_download_method_updates(download_method):
    """
    Run a single test of creating a pipeline with a given download_method, checking that the model is updated
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        model_rel_paths = ["tokenize/combined.pt", "mwt/combined.pt"]
        _seed_from_models_dir(test_dir, "en", model_rel_paths)

        assert sorted(os.listdir(test_dir)) == ['en', 'resources.json']
        en_dir = os.path.join(test_dir, 'en')
        en_dir_listing = sorted(os.listdir(en_dir))
        assert en_dir_listing == ['mwt', 'tokenize']
        tokenize_path = os.path.join(en_dir, "tokenize", "combined.pt")

        # Build a mapping from each seeded destination path back to its real
        # source so the mock can restore real content when "re-downloading".
        real_files = {
            os.path.join(test_dir, "en", rel): os.path.join(TEST_MODELS_DIR, "en", rel)
            for rel in model_rel_paths
        }

        def fake_request_file_with_restore(url, path, *args, **kwargs):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if path.endswith("resources.json"):
                real_path = os.path.join(TEST_MODELS_DIR, "resources.json")
                with open(real_path, encoding="utf-8") as fin:
                    data = fin.read()
                with open(path, "w", encoding="utf-8") as fout:
                    fout.write(data)
            elif path in real_files:
                # Restore the real model so Pipeline can load it after "re-download"
                shutil.copy2(real_files[path], path)
            else:
                open(path, "wb").close()

        # Corrupt the tokenizer so the md5 check will trigger a re-download
        with open(tokenize_path, "w") as fout:
            fout.write("Unban mox opal!")
        mod_time = os.path.getmtime(tokenize_path)

        with patch("stanza.resources.common.request_file", side_effect=fake_request_file_with_restore):
            with patch("stanza.pipeline.core.download_resources_json", side_effect=lambda *a, **kw: fake_request_file_with_restore(None, os.path.join(a[0], "resources.json"))):
                pipe = stanza.Pipeline("en", model_dir=test_dir, processors="tokenize", package={"tokenize": "combined"}, download_method=download_method)
                assert os.path.getmtime(tokenize_path) != mod_time

def test_download_fixed():
    """
    Test that a model is fixed if the existing model doesn't match the md5sum
    """
    for download_method in (core.DownloadMethod.REUSE_RESOURCES, core.DownloadMethod.DOWNLOAD_RESOURCES):
        check_download_method_updates(download_method)

def test_download_strings():
    """
    Same as the test of the download_method, but tests that the pipeline works for string download_method
    """
    for download_method in ("reuse_resources", "download_resources"):
        check_download_method_updates(download_method)

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

@pytest.fixture(scope="module")
def unknown_language_name():
    resources = load_resources_json(model_dir=TEST_MODELS_DIR)
    name = "en"
    while name in resources:
        name = name + "z"
    assert name != "en"
    return name

def test_empty_unknown_language(unknown_language_name):
    """
    Check that there is an error for trying to load an unknown language
    """
    with pytest.raises(ValueError):
        pipe = stanza.Pipeline(unknown_language_name, model_dir=TEST_MODELS_DIR, download_method=None)

def test_unknown_language_tokenizer(unknown_language_name):
    """
    Test that loading tokenize works for an unknown language
    """
    base_pipe = stanza.Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize", download_method=None)
    # even if we one day add MWT to English, the tokenizer by itself should still work
    tokenize_processor = base_pipe.processors["tokenize"]

    pipe=stanza.Pipeline(unknown_language_name,
                         model_dir=TEST_MODELS_DIR,
                         processors="tokenize",
                         allow_unknown_language=True,
                         tokenize_model_path=tokenize_processor.config['model_path'],
                         download_method=None)
    doc = pipe("This is a test")
    words = [x.text for x in doc.sentences[0].words]
    assert words == ['This', 'is', 'a', 'test']


def test_unknown_language_mwt(unknown_language_name):
    """
    Test that loading tokenize & mwt works for an unknown language
    """
    base_pipe = stanza.Pipeline("fr", dir=TEST_MODELS_DIR, processors="tokenize,mwt", download_method=None)
    assert len(base_pipe.processors) == 2
    tokenize_processor = base_pipe.processors["tokenize"]
    mwt_processor = base_pipe.processors["mwt"]

    pipe=stanza.Pipeline(unknown_language_name,
                         model_dir=TEST_MODELS_DIR,
                         processors="tokenize,mwt",
                         allow_unknown_language=True,
                         tokenize_model_path=tokenize_processor.config['model_path'],
                         mwt_model_path=mwt_processor.config['model_path'],
                         download_method=None)
