"""
Tests for prepare_resources, which turns a directory of per-processor models into per-language models plus resources.json

Most of these tests replace the default_packages settings with a small
fake configuration, so that they don't depend on which languages and
models the real default_packages currently lists.  The fake
configuration has three languages:
  en: uses a charlm, with a default NER model
  fr: no charlm, has an MWT model
  multilingual: just the langid model
"""

import hashlib
import json
import os
import sys
import zipfile

import pytest

import stanza
import stanza.resources.prepare_resources as prepare_resources

from stanza.tests import *

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

# processor directory -> model filenames in the fake input directory
FAKE_MODELS = {
    "tokenize":        ["en_ewt_tokenizer.pt", "fr_gsd_tokenizer.pt"],
    "mwt":             ["fr_gsd_mwt_expander.pt"],
    "lemma":           ["en_ewt_nocharlm_lemmatizer.pt", "en_ewt_charlm_lemmatizer.pt", "fr_gsd_nocharlm_lemmatizer.pt"],
    "pos":             ["en_ewt_nocharlm_tagger.pt", "en_ewt_charlm_tagger.pt", "fr_gsd_nocharlm_tagger.pt"],
    "depparse":        ["en_ewt_nocharlm_parser.pt", "en_ewt_charlm_parser.pt", "fr_gsd_nocharlm_parser.pt"],
    "pretrain":        ["en_conll17_pretrain.pt", "fr_conll17_pretrain.pt"],
    "forward_charlm":  ["en_1billion_forward_charlm.pt"],
    "backward_charlm": ["en_1billion_backward_charlm.pt"],
    "ner":             ["en_ontonotes_charlm_nertagger.pt"],
    "langid":          ["multilingual_ud_langid.pt"],
}

FAKE_CONFIG = {
    "default_treebanks":     {"en": "ewt", "fr": "gsd", "multilingual": "ud"},
    "default_pretrains":     {"en": "conll17", "fr": "conll17"},
    "default_charlms":       {"en": "1billion"},
    "no_pretrain_languages": {"multilingual"},
    "default_ners":          {"en": "ontonotes_charlm"},
    "allowed_empty_languages": [],
}

# settings which are empty in the fake configuration
EMPTY_CONFIG = ["pos_pretrains", "depparse_pretrains", "ner_pretrains",
                "pos_charlms", "depparse_charlms", "lemma_charlms", "tokenizer_charlms", "ner_charlms",
                "default_tokenizer", "default_sentiment", "default_constituency",
                "optional_constituency", "optional_coref",
                "TRANSFORMERS", "TRANSFORMER_NICKNAMES", "LEMMA_CLASSIFIER_DATASETS"]

@pytest.fixture
def fake_config(monkeypatch):
    for name, value in FAKE_CONFIG.items():
        # copy so that a test which modifies the config doesn't affect other tests
        monkeypatch.setattr(prepare_resources, name, type(value)(value))
    for name in EMPTY_CONFIG:
        monkeypatch.setattr(prepare_resources, name, {})
    monkeypatch.setattr(prepare_resources, "known_nicknames", lambda: ["transformer"])

def model_contents(model):
    """
    Distinct, deterministic contents for each fake model file
    """
    return ("fake model %s\n" % model).encode() * 50

def write_model(input_dir, processor, model, contents=None):
    path = input_dir / processor / model
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model_contents(model) if contents is None else contents)
    return path

@pytest.fixture
def input_dir(tmp_path):
    input_dir = tmp_path / "input"
    for processor, models in FAKE_MODELS.items():
        for model in models:
            write_model(input_dir, processor, model)
    return input_dir

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"

def run_prepare(monkeypatch, input_dir, output_dir, *extra_args):
    monkeypatch.setattr(sys, "argv", ["prepare_resources", "--input_dir", str(input_dir), "--output_dir", str(output_dir), "--num_workers", "2"] + list(extra_args))
    prepare_resources.main()
    return read_resources(output_dir)

def read_resources(output_dir):
    with open(output_dir / "resources.json") as fin:
        return json.load(fin)

def md5_of(path):
    with open(path, "rb") as fin:
        return hashlib.md5(fin.read()).hexdigest()

class CopyCounter:
    """
    Wraps copy_file_with_md5 to record which models actually get copied
    """
    def __init__(self, monkeypatch):
        self.copied = []
        original = prepare_resources.copy_file_with_md5
        def counting_copy(src, dst):
            self.copied.append(os.path.basename(src))
            return original(src, dst)
        monkeypatch.setattr(prepare_resources, "copy_file_with_md5", counting_copy)

def assert_fails_with(monkeypatch, capsys, input_dir, output_dir, expected_messages, *extra_args):
    """
    Run prepare_resources, check that it exits, and that each of the expected messages is in the output
    """
    with pytest.raises(SystemExit):
        run_prepare(monkeypatch, input_dir, output_dir, *extra_args)
    out = capsys.readouterr().out
    for message in expected_messages:
        assert message in out
    return out


def test_split_model_name():
    # Basic test
    lang, package, processor = prepare_resources.split_model_name('ro_nonstandard_tagger.pt')
    assert lang == 'ro'
    assert package == 'nonstandard'
    assert processor == 'pos'

    # Check that nertagger is found even though it also ends with tagger
    # Check that ncbi_disease is correctly partitioned despite the extra _
    lang, package, processor = prepare_resources.split_model_name('en_ncbi_disease_nertagger.pt')
    assert lang == 'en'
    assert package == 'ncbi_disease'
    assert processor == 'ner'

    # assert that processors with _ in them are also okay
    lang, package, processor = prepare_resources.split_model_name('en_pubmed_forward_charlm.pt')
    assert lang == 'en'
    assert package == 'pubmed'
    assert processor == 'forward_charlm'

    # the language code can have a - in it
    lang, package, processor = prepare_resources.split_model_name('zh-hans_gsdsimp_nocharlm_parser.pt')
    assert lang == 'zh-hans'
    assert package == 'gsdsimp_nocharlm'
    assert processor == 'depparse'

def test_split_model_name_unknown_processor():
    with pytest.raises(AssertionError):
        prepare_resources.split_model_name('en_ewt_frobnicator.pt')

@pytest.mark.parametrize("package, expected", [
    ("ewt_nopretrain",           ("ewt", False, False)),
    ("ewt_nocharlm",             ("ewt", True, False)),
    ("ewt_charlm",               ("ewt", True, True)),
    ("ewt_charlm_finetuned",     ("ewt", True, True)),
    ("ewt_transformer",          ("ewt", True, True)),
    ("ewt",                      ("ewt", True, True)),
])
def test_split_package(package, expected):
    assert prepare_resources.split_package(package) == expected

def test_split_package_default_charlm():
    # tokenizers without a pretrain type in the name default to no charlm
    assert prepare_resources.split_package("ewt", default_use_charlm=False) == ("ewt", True, False)

def test_get_dependencies(fake_config):
    charlm_deps = [{'model': 'forward_charlm', 'package': '1billion'}, {'model': 'backward_charlm', 'package': '1billion'}]
    pretrain_deps = [{'model': 'pretrain', 'package': 'conll17'}]

    assert prepare_resources.get_dependencies("pos", "en", "ewt_charlm") == pretrain_deps + charlm_deps
    assert prepare_resources.get_dependencies("pos", "en", "ewt_nocharlm") == pretrain_deps
    assert prepare_resources.get_dependencies("depparse", "en", "ewt_charlm") == pretrain_deps + charlm_deps
    assert prepare_resources.get_dependencies("pos", "fr", "gsd_nocharlm") == pretrain_deps
    # fr has no charlm, so even a model named _charlm doesn't get one
    assert prepare_resources.get_dependencies("pos", "fr", "gsd_charlm") == pretrain_deps
    assert prepare_resources.get_dependencies("pos", "en", "ewt_nopretrain") == []

    # tokenizers only use the forward charlm, and only when the name says so
    assert prepare_resources.get_dependencies("tokenize", "en", "ewt") == []
    assert prepare_resources.get_dependencies("tokenize", "en", "ewt_charlm") == charlm_deps[:1]

    # the lemmatizer only uses a pretrain for the contextual lemma classifier datasets
    assert prepare_resources.get_dependencies("lemma", "en", "ewt_charlm") == charlm_deps
    assert prepare_resources.get_dependencies("lemma", "en", "ewt_nocharlm") == []

    assert not prepare_resources.get_dependencies("mwt", "fr", "gsd")

def test_missing_pretrain(fake_config):
    with pytest.raises(RuntimeError):
        prepare_resources.get_dependencies("pos", "de", "gsd_nocharlm")

def test_copy_file_with_md5(tmp_path):
    src = tmp_path / "src.pt"
    src.write_bytes(b"some model data" * 1000)
    os.utime(src, (1700000000, 1700000000))
    dst = tmp_path / "a" / "b" / "dst.pt"

    md5 = prepare_resources.copy_file_with_md5(str(src), str(dst))
    assert md5 == md5_of(src)
    assert md5 == prepare_resources.get_md5(str(dst))
    assert dst.read_bytes() == src.read_bytes()
    assert os.stat(dst).st_mtime == 1700000000
    assert os.listdir(dst.parent) == ["dst.pt"]

def test_run_in_threads_order():
    jobs = [(x,) for x in range(50)]
    assert prepare_resources.run_in_threads(lambda x: x * x, jobs, 4, "squares") == [x * x for x in range(50)]
    assert prepare_resources.run_in_threads(lambda x: x, [], 4, "nothing") == []

def test_run_in_threads_error():
    def fail_on_three(x):
        if x == 3:
            raise ValueError("three")
        return x
    with pytest.raises(ValueError, match="three"):
        prepare_resources.run_in_threads(fail_on_three, [(x,) for x in range(10)], 2, "failing")

def test_output_cache(tmp_path):
    output_dir = tmp_path / "output"
    src = tmp_path / "src.pt"
    src.write_bytes(b"model")
    dst = output_dir / "en" / "models" / "pos" / "ewt.pt"
    md5 = prepare_resources.copy_file_with_md5(str(src), str(dst))

    cache = prepare_resources.OutputCache(str(output_dir))
    assert cache.lookup_model(str(src), str(dst)) is None
    cache.store_model(str(src), str(dst), md5)
    cache.save()

    # a fresh cache reads back what was saved
    cache = prepare_resources.OutputCache(str(output_dir))
    assert cache.lookup_model(str(src), str(dst)) == md5
    # the entry belongs to that particular source file
    assert cache.lookup_model(str(tmp_path / "other.pt"), str(dst)) is None
    # use_existing=False ignores the saved entries, as with --force
    assert prepare_resources.OutputCache(str(output_dir), use_existing=False).lookup_model(str(src), str(dst)) is None

    # changing the source invalidates the entry
    src.write_bytes(b"new model")
    assert cache.lookup_model(str(src), str(dst)) is None

    # so does changing the destination
    md5 = prepare_resources.copy_file_with_md5(str(src), str(dst))
    cache.store_model(str(src), str(dst), md5)
    assert cache.lookup_model(str(src), str(dst)) == md5
    dst.write_bytes(b"corrupted")
    assert cache.lookup_model(str(src), str(dst)) is None

def test_output_cache_zip(tmp_path):
    zip_path = tmp_path / "default.zip"
    zip_path.write_bytes(b"zip data")
    members = [["pos/ewt.pt", "abc"], ["pretrain/conll17.pt", "def"]]

    cache = prepare_resources.OutputCache(str(tmp_path))
    cache.store_zip(str(zip_path), members, "zipmd5")
    assert cache.lookup_zip(str(zip_path), members) == "zipmd5"
    assert cache.lookup_zip(str(zip_path), [["pos/ewt.pt", "abc"], ["pretrain/conll17.pt", "xyz"]]) is None
    assert cache.lookup_zip(str(zip_path), members[:1]) is None
    zip_path.write_bytes(b"different zip data")
    assert cache.lookup_zip(str(zip_path), members) is None

def test_output_cache_corrupt(tmp_path, capsys):
    (tmp_path / prepare_resources.OutputCache.CACHE_NAME).write_text("{ not json")
    cache = prepare_resources.OutputCache(str(tmp_path))
    assert cache.entries == {}
    assert "WARNING" in capsys.readouterr().out


def test_full_build(monkeypatch, fake_config, input_dir, output_dir):
    resources = run_prepare(monkeypatch, input_dir, output_dir)

    # every model is copied into the per-language directories with the right md5
    for processor, models in FAKE_MODELS.items():
        for model in models:
            lang, package, _ = prepare_resources.split_model_name(model)
            output_path = output_dir / lang / "models" / processor / (package + ".pt")
            assert output_path.read_bytes() == model_contents(model)
            assert resources[lang][processor][package]['md5'] == md5_of(output_path)

    en = resources["en"]
    assert en["pos"]["ewt_charlm"]["dependencies"] == [{'model': 'pretrain', 'package': 'conll17'},
                                                        {'model': 'forward_charlm', 'package': '1billion'},
                                                        {'model': 'backward_charlm', 'package': '1billion'}]
    assert "dependencies" not in en["tokenize"]["ewt"]

    assert en["packages"]["default"] == {"tokenize": "ewt", "lemma": "ewt_nocharlm", "pos": "ewt_charlm",
                                         "depparse": "ewt_charlm", "ner": "ontonotes_charlm"}
    assert en["default_processors"] == en["packages"]["default"]
    assert en["packages"]["default_fast"] == {"tokenize": "ewt", "lemma": "ewt_nocharlm", "pos": "ewt_nocharlm",
                                              "depparse": "ewt_nocharlm", "ner": "ontonotes_charlm"}
    assert en["packages"]["default_accurate"]["lemma"] == "ewt_charlm"
    assert en["packages"]["ewt"] == {"tokenize": "ewt", "pos": "ewt_charlm", "lemma": "ewt_nocharlm", "depparse": "ewt_charlm"}

    fr = resources["fr"]
    assert fr["packages"]["default"] == {"tokenize": "gsd", "mwt": "gsd", "lemma": "gsd_nocharlm",
                                         "pos": "gsd_nocharlm", "depparse": "gsd_nocharlm"}

    assert resources["multilingual"]["packages"]["default"] == {"langid": "ud"}

    # the default.zip has the default models and their dependencies, in PROCESSORS order
    en_zip = output_dir / "en" / "models" / "default.zip"
    with zipfile.ZipFile(en_zip) as zin:
        assert zin.namelist() == ["tokenize/ewt.pt", "lemma/ewt_nocharlm.pt", "pos/ewt_charlm.pt", "depparse/ewt_charlm.pt",
                                  "pretrain/conll17.pt", "ner/ontonotes_charlm.pt",
                                  "forward_charlm/1billion.pt", "backward_charlm/1billion.pt"]
        assert zin.read("pos/ewt_charlm.pt") == model_contents("en_ewt_charlm_tagger.pt")
    assert en["default_md5"] == md5_of(en_zip)
    with zipfile.ZipFile(output_dir / "fr" / "models" / "default.zip") as zin:
        assert "mwt/gsd.pt" in zin.namelist()

    # language names and codes become aliases
    assert en["lang_name"] == "English"
    assert resources["english"] == {"alias": "en"}
    assert resources["french"] == {"alias": "fr"}
    assert resources["no"] == {"alias": "nb"}
    assert "{lang}" in resources["url"]

    assert (output_dir / prepare_resources.OutputCache.CACHE_NAME).exists()
    leftovers = [name for _, _, names in os.walk(output_dir) for name in names if name.endswith(".partial")]
    assert leftovers == []

def test_rerun_uses_cache(monkeypatch, fake_config, input_dir, output_dir):
    run_prepare(monkeypatch, input_dir, output_dir)
    first_json = (output_dir / "resources.json").read_text()

    counter = CopyCounter(monkeypatch)
    run_prepare(monkeypatch, input_dir, output_dir)
    assert counter.copied == []
    assert (output_dir / "resources.json").read_text() == first_json

def test_rerun_after_change(monkeypatch, fake_config, input_dir, output_dir):
    first = run_prepare(monkeypatch, input_dir, output_dir)
    fr_zip = output_dir / "fr" / "models" / "default.zip"
    fr_zip_stat = os.stat(fr_zip)

    changed = write_model(input_dir, "pos", "en_ewt_charlm_tagger.pt", b"retrained en pos model")
    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir)

    assert counter.copied == ["en_ewt_charlm_tagger.pt"]
    assert second["en"]["pos"]["ewt_charlm"]["md5"] == md5_of(changed)
    assert second["en"]["pos"]["ewt_charlm"]["md5"] != first["en"]["pos"]["ewt_charlm"]["md5"]

    # the en zip is rebuilt with the new model, the fr zip is left alone
    with zipfile.ZipFile(output_dir / "en" / "models" / "default.zip") as zin:
        assert zin.read("pos/ewt_charlm.pt") == b"retrained en pos model"
    assert second["en"]["default_md5"] != first["en"]["default_md5"]
    assert second["fr"]["default_md5"] == first["fr"]["default_md5"]
    assert os.stat(fr_zip).st_mtime_ns == fr_zip_stat.st_mtime_ns

def test_rerun_after_output_damaged(monkeypatch, fake_config, input_dir, output_dir):
    first = run_prepare(monkeypatch, input_dir, output_dir)
    (output_dir / "fr" / "models" / "pos" / "gsd_nocharlm.pt").write_bytes(b"oops")

    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir)
    assert counter.copied == ["fr_gsd_nocharlm_tagger.pt"]
    assert second == first

def test_force(monkeypatch, fake_config, input_dir, output_dir):
    first = run_prepare(monkeypatch, input_dir, output_dir)
    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir, "--force")
    assert len(counter.copied) == sum(len(models) for models in FAKE_MODELS.values())
    assert second == first

def test_check_only(monkeypatch, fake_config, input_dir, output_dir, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare_resources", "--input_dir", str(input_dir), "--output_dir", str(output_dir), "--check_only"])
    prepare_resources.main()
    assert "No problems found" in capsys.readouterr().out
    assert not output_dir.exists()

def test_lang(monkeypatch, fake_config, input_dir, output_dir):
    first = run_prepare(monkeypatch, input_dir, output_dir)

    # changes to en are ignored when only rebuilding fr
    write_model(input_dir, "pos", "en_ewt_charlm_tagger.pt", b"retrained en pos model")
    changed = write_model(input_dir, "pos", "fr_gsd_nocharlm_tagger.pt", b"retrained fr pos model")
    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir, "--lang", "fr")

    assert counter.copied == ["fr_gsd_nocharlm_tagger.pt"]
    assert second["en"] == first["en"]
    assert second["fr"]["pos"]["gsd_nocharlm"]["md5"] == md5_of(changed)
    assert second["fr"]["default_md5"] != first["fr"]["default_md5"]
    assert second["french"] == {"alias": "fr"}

def test_lang_removes_deleted_models(monkeypatch, fake_config, input_dir, output_dir, capsys):
    write_model(input_dir, "pos", "fr_extra_nocharlm_tagger.pt")
    first = run_prepare(monkeypatch, input_dir, output_dir)
    assert "extra_nocharlm" in first["fr"]["pos"]

    os.remove(input_dir / "pos" / "fr_extra_nocharlm_tagger.pt")
    capsys.readouterr()
    second = run_prepare(monkeypatch, input_dir, output_dir, "--lang", "fr")
    assert "extra_nocharlm" not in second["fr"]["pos"]

    # the old file is left in place, but reported
    out = capsys.readouterr().out
    assert "1 .pt file(s)" in out
    assert os.path.join("fr", "models", "pos", "extra_nocharlm.pt") in out
    assert (output_dir / "fr" / "models" / "pos" / "extra_nocharlm.pt").exists()

def test_no_unexpected_models(monkeypatch, fake_config, input_dir, output_dir, capsys):
    run_prepare(monkeypatch, input_dir, output_dir)
    run_prepare(monkeypatch, input_dir, output_dir)
    assert "not in resources.json" not in capsys.readouterr().out

def test_unexpected_models_reported(monkeypatch, fake_config, input_dir, output_dir, capsys):
    run_prepare(monkeypatch, input_dir, output_dir)
    stray = output_dir / "en" / "models" / "pos" / "old_model.pt"
    stray.write_bytes(b"leftover from some earlier release")
    os.remove(input_dir / "lemma" / "en_ewt_charlm_lemmatizer.pt")
    capsys.readouterr()

    unexpected = [os.path.join("en", "models", "lemma", "ewt_charlm.pt"),
                  os.path.join("en", "models", "pos", "old_model.pt")]

    # check_only lists them, as does a full run
    for extra_args in (["--check_only"], []):
        monkeypatch.setattr(sys, "argv", ["prepare_resources", "--input_dir", str(input_dir), "--output_dir", str(output_dir)] + extra_args)
        prepare_resources.main()
        out = capsys.readouterr().out
        assert "2 .pt file(s)" in out
        for path in unexpected:
            assert path in out
        assert stray.exists()
        assert (output_dir / unexpected[0]).exists()

def test_partial_files_removed(monkeypatch, fake_config, input_dir, output_dir, capsys):
    first = run_prepare(monkeypatch, input_dir, output_dir)
    partials = [output_dir / "en" / "models" / "pos" / "ewt_charlm.pt.partial",
                output_dir / "fr" / "models" / "default.zip.partial",
                output_dir / (prepare_resources.OutputCache.CACHE_NAME + ".partial")]
    for path in partials:
        path.write_bytes(b"interrupted")
    capsys.readouterr()

    # check_only reports them without removing anything
    monkeypatch.setattr(sys, "argv", ["prepare_resources", "--input_dir", str(input_dir), "--output_dir", str(output_dir), "--check_only"])
    prepare_resources.main()
    out = capsys.readouterr().out
    assert "Would remove 3 leftover partial file(s)" in out
    assert all(path.exists() for path in partials)

    # a real run removes them, and the cached models are still up to date
    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir)
    out = capsys.readouterr().out
    assert "Removing 3 leftover partial file(s)" in out
    assert os.path.join("en", "models", "pos", "ewt_charlm.pt.partial") in out
    assert not any(path.exists() for path in partials)
    assert counter.copied == []
    assert second == first

def test_partial_symlink_outside(monkeypatch, fake_config, input_dir, output_dir, tmp_path, capsys):
    """
    A .partial symlink pointing outside the output directory is reported, not removed
    """
    first = run_prepare(monkeypatch, input_dir, output_dir)
    target = tmp_path / "precious.partial"
    target.write_bytes(b"do not delete")
    link = output_dir / "en" / "models" / "pos" / "ewt_charlm.pt.partial"
    os.symlink(target, link)
    capsys.readouterr()

    second = run_prepare(monkeypatch, input_dir, output_dir)
    out = capsys.readouterr().out
    assert "not removing 1 partial file(s)" in out
    assert str(target) in out
    assert "Removing" not in out
    assert os.path.islink(link)
    assert target.read_bytes() == b"do not delete"
    assert second == first

def test_partial_symlink_inside(monkeypatch, fake_config, input_dir, output_dir, capsys):
    """
    A .partial symlink pointing inside the output directory is removed, and only the link goes away
    """
    first = run_prepare(monkeypatch, input_dir, output_dir)
    target = output_dir / "en" / "models" / "pos" / "ewt_charlm.pt"
    link = output_dir / "fr" / "models" / "pos" / "gsd_nocharlm.pt.partial"
    os.symlink(target, link)

    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir)
    assert not os.path.lexists(link)
    assert target.read_bytes() == model_contents("en_ewt_charlm_tagger.pt")
    assert counter.copied == []
    assert second == first

def test_symlinked_dir_not_walked(monkeypatch, fake_config, input_dir, output_dir, tmp_path, capsys):
    """
    A symlinked directory inside the output directory is not searched at all
    """
    run_prepare(monkeypatch, input_dir, output_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "something.partial").write_bytes(b"not ours")
    (elsewhere / "something.pt").write_bytes(b"not ours either")
    os.symlink(elsewhere, output_dir / "en" / "linked_dir")
    capsys.readouterr()

    run_prepare(monkeypatch, input_dir, output_dir)
    out = capsys.readouterr().out
    assert "partial file" not in out
    assert "not in resources.json" not in out
    assert (elsewhere / "something.partial").exists()

def test_output_dir_is_symlink(monkeypatch, fake_config, input_dir, tmp_path, capsys):
    """
    When the output directory is a symlink, partial files in the directory it points to are still removed
    """
    real_output = tmp_path / "real_output"
    real_output.mkdir()
    linked_output = tmp_path / "linked_output"
    os.symlink(real_output, linked_output)

    first = run_prepare(monkeypatch, input_dir, linked_output)
    assert (real_output / "resources.json").exists()
    partial = real_output / "en" / "models" / "pos" / "ewt_charlm.pt.partial"
    partial.write_bytes(b"interrupted")
    capsys.readouterr()

    second = run_prepare(monkeypatch, input_dir, linked_output)
    assert "Removing 1 leftover partial file(s)" in capsys.readouterr().out
    assert not partial.exists()
    assert second == first

def test_partial_files_kept_on_error(monkeypatch, fake_config, input_dir, output_dir, capsys):
    """
    A run which fails its checks doesn't touch the output directory at all
    """
    run_prepare(monkeypatch, input_dir, output_dir)
    partial = output_dir / "en" / "models" / "pos" / "ewt_charlm.pt.partial"
    partial.write_bytes(b"interrupted")
    write_model(input_dir, "tokenize", "de_gsd_tokenizer.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["de not in default treebanks"])
    assert partial.exists()

def test_packages_only(monkeypatch, fake_config, input_dir, output_dir):
    first = run_prepare(monkeypatch, input_dir, output_dir)

    # packages_only rebuilds the packages from the existing resources.json
    # add an MWT model to en in the resources to check that it shows up in the packages
    resources = read_resources(output_dir)
    resources["en"]["mwt"] = {"ewt": {"md5": "12345"}}
    prepare_resources.write_resources(resources, str(output_dir))

    counter = CopyCounter(monkeypatch)
    second = run_prepare(monkeypatch, input_dir, output_dir, "--packages_only")
    assert counter.copied == []
    assert second["en"]["packages"]["default"]["mwt"] == "ewt"
    assert second["en"]["default_md5"] == first["en"]["default_md5"]


def test_error_not_in_default_treebanks(monkeypatch, fake_config, input_dir, output_dir, capsys):
    write_model(input_dir, "tokenize", "de_gsd_tokenizer.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["de not in default treebanks"])
    assert not output_dir.exists()

def test_error_missing_dependency(monkeypatch, fake_config, input_dir, output_dir, capsys):
    os.remove(input_dir / "forward_charlm" / "en_1billion_forward_charlm.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir,
                      ["en pos ewt_charlm depends on forward_charlm 1billion, which does not exist",
                       "Processor forward_charlm package 1billion needed for en default.zip"])
    assert not output_dir.exists()

def test_error_missing_default_model(monkeypatch, fake_config, input_dir, output_dir, capsys):
    os.remove(input_dir / "ner" / "en_ontonotes_charlm_nertagger.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir,
                      ["en package default uses ner ontonotes_charlm, but there is no such model"])
    assert not output_dir.exists()

def test_error_missing_tokenizer(monkeypatch, fake_config, input_dir, output_dir, capsys):
    os.remove(input_dir / "tokenize" / "fr_gsd_tokenizer.pt")
    out = assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["No tokenizer models found for fr"])
    assert "Traceback" not in out
    assert not output_dir.exists()

def test_error_missing_langid(monkeypatch, fake_config, input_dir, output_dir, capsys):
    os.remove(input_dir / "langid" / "multilingual_ud_langid.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["No multilingual models found"])
    assert not output_dir.exists()

def test_error_bad_filename(monkeypatch, fake_config, input_dir, output_dir, capsys):
    write_model(input_dir, "pos", "garbage.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["pos/garbage.pt: cannot parse model name"])
    assert not output_dir.exists()

def test_error_unknown_language_code(monkeypatch, fake_config, input_dir, output_dir, capsys):
    # a language in default_treebanks but not in lcode2lang would be dropped by process_lcode
    assert "qqq" not in prepare_resources.lcode2lang
    prepare_resources.default_treebanks["qqq"] = "fake"
    prepare_resources.no_pretrain_languages.add("qqq")
    write_model(input_dir, "tokenize", "qqq_fake_tokenizer.pt")
    write_model(input_dir, "pos", "qqq_fake_nopretrain_tagger.pt")
    write_model(input_dir, "depparse", "qqq_fake_nopretrain_parser.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["qqq not found in lcode2lang"])
    assert not output_dir.exists()

def test_errors_reported_together(monkeypatch, fake_config, input_dir, output_dir, capsys):
    """
    Several unrelated problems are all reported in one run
    """
    write_model(input_dir, "tokenize", "de_gsd_tokenizer.pt")
    os.remove(input_dir / "ner" / "en_ontonotes_charlm_nertagger.pt")
    os.remove(input_dir / "langid" / "multilingual_ud_langid.pt")
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir,
                      ["de not in default treebanks",
                       "en package default uses ner ontonotes_charlm",
                       "No multilingual models found"])
    assert not output_dir.exists()

def test_error_keeps_existing_output(monkeypatch, fake_config, input_dir, output_dir, capsys):
    """
    A failed check leaves a previous build alone
    """
    run_prepare(monkeypatch, input_dir, output_dir)
    before = (output_dir / "resources.json").read_text()

    write_model(input_dir, "tokenize", "de_gsd_tokenizer.pt")
    counter = CopyCounter(monkeypatch)
    assert_fails_with(monkeypatch, capsys, input_dir, output_dir, ["de not in default treebanks"])
    assert counter.copied == []
    assert (output_dir / "resources.json").read_text() == before
