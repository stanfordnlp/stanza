"""
Tests for stanza.resources.list_installed.

All tests build temporary directories with fake .pt files so they never
touch the real Stanza cache or require downloaded models.
"""

import json
import os
import tempfile

import pytest

from stanza import __resources_version__

from stanza.resources.list_installed import (
    _dir_size_bytes,
    _format_rows,
    _human_size,
    _scan_cache_root,
    _scan_model_dir,
    _STANZA_CACHE_ROOT,
    _STANZA_TEST_CACHE_ROOT,
    list_installed,
)

# A version string distinct from __resources_version__, used in tests that
# need to simulate an older install alongside the current one.
_OLDER_VERSION = '1.10.0'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path, content=b''):
    """Create a file (and any parent dirs) with the given byte content."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(content)


def _make_model_dir(base, lang, processor, package, size=1024,
                    resources_json=None):
    """
    Create a minimal model directory structure under *base*::

        base/
            resources.json          (optional)
            <lang>/<processor>/<package>.pt

    Returns *base* for convenience.
    """
    _touch(os.path.join(base, lang, processor, f'{package}.pt'), b'x' * size)
    if resources_json is not None:
        with open(os.path.join(base, 'resources.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(resources_json, f)
    return base


def _make_versioned_cache(cache_root, version, lang, processor, package,
                          size=1024, resources_json=None, subdir='resources'):
    """
    Create a versioned directory tree under *cache_root*::

        cache_root/<version>/<subdir>/<lang>/<processor>/<package>.pt

    Use ``subdir='models'`` for test fixture caches.
    """
    model_dir = os.path.join(cache_root, version, subdir)
    return _make_model_dir(model_dir, lang, processor, package,
                           size=size, resources_json=resources_json)


def _no_custom_dir(monkeypatch):
    """Remove STANZA_RESOURCES_DIR from the environment."""
    monkeypatch.delenv('STANZA_RESOURCES_DIR', raising=False)


# ---------------------------------------------------------------------------
# _human_size
# ---------------------------------------------------------------------------

def test_human_size_bytes():
    assert _human_size(512) == '512.0 B'

def test_human_size_kilobytes():
    assert _human_size(2048) == '2.0 KB'

def test_human_size_megabytes():
    assert _human_size(3 * 1024 ** 2) == '3.0 MB'

def test_human_size_gigabytes():
    assert _human_size(1024 ** 3) == '1.0 GB'

def test_human_size_zero():
    assert _human_size(0) == '0.0 B'


# ---------------------------------------------------------------------------
# _dir_size_bytes
# ---------------------------------------------------------------------------

def test_dir_size_bytes_single_file():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(os.path.join(tmp, 'a.pt'), b'x' * 100)
        assert _dir_size_bytes(tmp) == 100

def test_dir_size_bytes_nested_files():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(os.path.join(tmp, 'sub', 'a.pt'), b'x' * 200)
        _touch(os.path.join(tmp, 'b.pt'), b'x' * 50)
        assert _dir_size_bytes(tmp) == 250

def test_dir_size_bytes_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert _dir_size_bytes(tmp) == 0


# ---------------------------------------------------------------------------
# _scan_model_dir
# ---------------------------------------------------------------------------

def test_scan_model_dir_nonexistent():
    assert _scan_model_dir('/this/does/not/exist') == []

def test_scan_model_dir_basic():
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt', size=100)
        rows = _scan_model_dir(tmp)
    assert len(rows) == 1
    assert rows[0]['lang'] == 'en'
    assert rows[0]['processor'] == 'pos'
    assert rows[0]['package'] == 'ewt'
    assert rows[0]['size_bytes'] == 100
    assert rows[0]['version'] is None

def test_scan_model_dir_version_propagated():
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt')
        rows = _scan_model_dir(tmp, version=_OLDER_VERSION)
    assert rows[0]['version'] == _OLDER_VERSION

def test_scan_model_dir_multiple_packages():
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt')
        _touch(os.path.join(tmp, 'en', 'pos', 'craft.pt'), b'y' * 50)
        rows = _scan_model_dir(tmp)
    assert {r['package'] for r in rows} == {'ewt', 'craft'}

def test_scan_model_dir_multiple_languages():
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt')
        _make_model_dir(tmp, 'de', 'pos', 'gsd')
        rows = _scan_model_dir(tmp)
    assert {r['lang'] for r in rows} == {'en', 'de'}

def test_scan_model_dir_resources_json_not_treated_as_lang():
    """resources.json at the top level must not appear as a language."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt',
                        resources_json={'url': 'http://example.com'})
        rows = _scan_model_dir(tmp)
    assert 'resources.json' not in {r['lang'] for r in rows}

def test_scan_model_dir_lang_name_from_resources_json():
    rj = {
        'url': 'http://example.com',
        'en': {'lang_name': 'English', 'pos': {'ewt': {'md5': 'abc'}}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt', resources_json=rj)
        rows = _scan_model_dir(tmp)
    assert rows[0]['lang_name'] == 'English'

def test_scan_model_dir_lang_name_falls_back_to_code():
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'sd', 'pos', 'isra')
        rows = _scan_model_dir(tmp)
    assert rows[0]['lang_name'] == 'sd'

def test_scan_model_dir_alias_resolved():
    """An aliased lang code should resolve to the target's lang_name."""
    rj = {
        'url': 'http://example.com',
        'german': {'alias': 'de'},
        'de': {'lang_name': 'German', 'pos': {'gsd': {'md5': 'abc'}}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'german', 'pos', 'gsd', resources_json=rj)
        rows = _scan_model_dir(tmp)
    assert rows[0]['lang_name'] == 'German'

def test_scan_model_dir_alias_cycle_terminates():
    """A malformed alias cycle must not hang."""
    rj = {
        'url': 'http://example.com',
        'a': {'alias': 'b'},
        'b': {'alias': 'a'},
    }
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'a', 'pos', 'pkg', resources_json=rj)
        rows = _scan_model_dir(tmp)
    assert len(rows) == 1

def test_scan_model_dir_no_pt_files_reports_directory():
    """A processor dir with no .pt files should appear as a directory row."""
    with tempfile.TemporaryDirectory() as tmp:
        subdir = os.path.join(tmp, 'en', 'pretrain', 'gsd')
        os.makedirs(subdir)
        _touch(os.path.join(subdir, 'model.bin'), b'z' * 200)
        rows = _scan_model_dir(tmp)
    assert len(rows) == 1
    assert rows[0]['package'] is None
    assert rows[0]['size_bytes'] == 200

def test_scan_model_dir_stray_files_in_lang_dir_ignored():
    """Non-directory entries inside a lang dir (e.g. default.zip) are skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt')
        _touch(os.path.join(tmp, 'en', 'default.zip'), b'z' * 100)
        rows = _scan_model_dir(tmp)
    assert len(rows) == 1
    assert rows[0]['processor'] == 'pos'


# ---------------------------------------------------------------------------
# _scan_cache_root
# ---------------------------------------------------------------------------

def test_scan_cache_root_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert _scan_cache_root(tmp) == []

def test_scan_cache_root_nonexistent():
    assert _scan_cache_root('/does/not/exist') == []

def test_scan_cache_root_multiple_versions():
    with tempfile.TemporaryDirectory() as cache_root:
        _make_versioned_cache(cache_root, _OLDER_VERSION, 'en', 'pos', 'ewt', size=10)
        _make_versioned_cache(cache_root, __resources_version__, 'de', 'pos', 'gsd', size=20)
        rows = _scan_cache_root(cache_root)
    assert {r['version'] for r in rows} == {_OLDER_VERSION, __resources_version__}
    assert len(rows) == 2

def test_scan_cache_root_ignores_dirs_without_resources_subdir():
    """A version dir that has no 'resources' subdirectory should be skipped."""
    with tempfile.TemporaryDirectory() as cache_root:
        os.makedirs(os.path.join(cache_root, 'junk', 'something'))
        _make_versioned_cache(cache_root, __resources_version__, 'en', 'pos', 'ewt')
        rows = _scan_cache_root(cache_root)
    assert len(rows) == 1
    assert rows[0]['version'] == __resources_version__


# ---------------------------------------------------------------------------
# _format_rows
# ---------------------------------------------------------------------------

def _make_row(lang='en', lang_name='English', processor='pos',
              package='ewt', size=1024, version='1.0.0'):
    return {
        'version': version,
        'lang': lang,
        'lang_name': lang_name,
        'processor': processor,
        'package': package,
        'path': f'/fake/{lang}/{processor}/{package}.pt',
        'size_bytes': size,
    }

def test_format_rows_headers():
    headers, _, _ = _format_rows([_make_row()])
    assert headers == ['Language', 'Code', 'Processor', 'Package', 'Size']

def test_format_rows_none_package_shown_as_directory():
    row = _make_row()
    row['package'] = None
    _, col_data, _ = _format_rows([row])
    assert '(directory)' in col_data[0]

def test_format_rows_widths_at_least_header_length():
    headers, _, widths = _format_rows([_make_row()])
    for h, w in zip(headers, widths):
        assert w >= len(h)

def test_format_rows_widths_reflect_data():
    row = _make_row(lang_name='A' * 50)
    _, _, widths = _format_rows([row])
    assert widths[0] == 50


# ---------------------------------------------------------------------------
# list_installed
# ---------------------------------------------------------------------------

def test_list_installed_custom_dir(monkeypatch):
    """When STANZA_RESOURCES_DIR is set, only that dir is scanned."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt')
        monkeypatch.setenv('STANZA_RESOURCES_DIR', tmp)
        rows = list_installed(model_dir=tmp, print_table=False)
    assert len(rows) == 1
    assert rows[0]['version'] is None

def test_list_installed_scans_all_versions(monkeypatch):
    """Without STANZA_RESOURCES_DIR, all versioned subdirs are scanned."""
    with tempfile.TemporaryDirectory() as cache_root:
        _make_versioned_cache(cache_root, _OLDER_VERSION, 'en', 'pos', 'ewt', size=10)
        _make_versioned_cache(cache_root, __resources_version__, 'en', 'pos', 'ewt', size=20)
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            cache_root)
        rows = list_installed(print_table=False)
    assert {r['version'] for r in rows} == {_OLDER_VERSION, __resources_version__}
    assert len(rows) == 2

def test_list_installed_empty_cache(monkeypatch):
    with tempfile.TemporaryDirectory() as cache_root:
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            cache_root)
        rows = list_installed(model_dir=cache_root, print_table=False)
    assert rows == []

def test_list_installed_nonexistent_custom_dir(monkeypatch):
    monkeypatch.setenv('STANZA_RESOURCES_DIR', '/does/not/exist')
    rows = list_installed(model_dir='/does/not/exist', print_table=False)
    assert rows == []

def test_list_installed_version_order(monkeypatch):
    """Versions should appear in sorted (ascending) order."""
    with tempfile.TemporaryDirectory() as cache_root:
        _make_versioned_cache(cache_root, __resources_version__, 'en', 'pos', 'ewt')
        _make_versioned_cache(cache_root, _OLDER_VERSION, 'en', 'pos', 'ewt')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            cache_root)
        rows = list_installed(print_table=False)
    assert rows[0]['version'] == _OLDER_VERSION
    assert rows[1]['version'] == __resources_version__

def test_list_installed_print_table_smoke(monkeypatch):
    """print_table=True should run without errors."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_model_dir(tmp, 'en', 'pos', 'ewt')
        monkeypatch.setenv('STANZA_RESOURCES_DIR', tmp)
        list_installed(model_dir=tmp, print_table=True)

def test_list_installed_multi_version_print_smoke(monkeypatch):
    with tempfile.TemporaryDirectory() as cache_root:
        _make_versioned_cache(cache_root, _OLDER_VERSION, 'en', 'pos', 'ewt')
        _make_versioned_cache(cache_root, __resources_version__, 'de', 'pos', 'gsd')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            cache_root)
        list_installed(print_table=True)


# ---------------------------------------------------------------------------
# list_installed: include_test_models
# ---------------------------------------------------------------------------

def test_include_test_models_excluded_by_default(monkeypatch):
    with tempfile.TemporaryDirectory() as model_cache, \
         tempfile.TemporaryDirectory() as test_cache:
        _make_versioned_cache(model_cache, __resources_version__, 'en', 'pos', 'ewt')
        _make_versioned_cache(test_cache, __resources_version__, 'en', 'pos', 'ewt',
                              subdir='models')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            model_cache)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_TEST_CACHE_ROOT',
                            test_cache)
        rows = list_installed(print_table=False)
    assert len(rows) == 1

def test_include_test_models_appended(monkeypatch):
    with tempfile.TemporaryDirectory() as model_cache, \
         tempfile.TemporaryDirectory() as test_cache:
        _make_versioned_cache(model_cache, __resources_version__, 'en', 'pos', 'ewt', size=10)
        _make_versioned_cache(test_cache, __resources_version__, 'de', 'pos', 'gsd', size=20,
                              subdir='models')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            model_cache)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_TEST_CACHE_ROOT',
                            test_cache)
        rows = list_installed(print_table=False, include_test_models=True)
    assert len(rows) == 2
    assert {r['lang'] for r in rows} == {'en', 'de'}

def test_include_test_models_empty_test_cache(monkeypatch):
    """include_test_models=True with no test fixtures should not raise."""
    with tempfile.TemporaryDirectory() as model_cache, \
         tempfile.TemporaryDirectory() as test_cache:
        _make_versioned_cache(model_cache, __resources_version__, 'en', 'pos', 'ewt')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            model_cache)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_TEST_CACHE_ROOT',
                            test_cache)
        rows = list_installed(print_table=False, include_test_models=True)
    assert len(rows) == 1
    assert rows[0]['lang'] == 'en'

def test_include_test_models_print_smoke(monkeypatch):
    with tempfile.TemporaryDirectory() as model_cache, \
         tempfile.TemporaryDirectory() as test_cache:
        _make_versioned_cache(model_cache, __resources_version__, 'en', 'pos', 'ewt')
        _make_versioned_cache(test_cache, __resources_version__, 'de', 'pos', 'gsd',
                              subdir='models')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            model_cache)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_TEST_CACHE_ROOT',
                            test_cache)
        list_installed(print_table=True, include_test_models=True)

def test_include_test_models_order(monkeypatch):
    """Test rows must come after model rows in the returned list."""
    with tempfile.TemporaryDirectory() as model_cache, \
         tempfile.TemporaryDirectory() as test_cache:
        _make_versioned_cache(model_cache, __resources_version__, 'en', 'pos', 'ewt')
        _make_versioned_cache(test_cache, __resources_version__, 'de', 'pos', 'gsd',
                              subdir='models')
        _no_custom_dir(monkeypatch)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_CACHE_ROOT',
                            model_cache)
        monkeypatch.setattr('stanza.resources.list_installed._STANZA_TEST_CACHE_ROOT',
                            test_cache)
        rows = list_installed(print_table=False, include_test_models=True)
    assert rows[0]['lang'] == 'en'
    assert rows[1]['lang'] == 'de'
