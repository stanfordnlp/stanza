"""
Utilities for inspecting locally installed Stanza models.

Can also be run directly::

    python -m stanza.resources.installed
    python -m stanza.resources.installed --include-test-models
"""

import os
import logging
from itertools import groupby

from platformdirs import user_cache_dir

from stanza.resources.common import DEFAULT_MODEL_DIR, load_resources_json, ResourcesFileNotFoundError
from stanza._version import __resources_version__

logger = logging.getLogger('stanza')

_SKIP_ENTRIES = {'resources.json'}

# The unversioned platform cache root, e.g. ~/.cache/stanza on Linux.
# Each subdirectory of this is expected to be a resources version, e.g. 1.10.0.
_STANZA_CACHE_ROOT = user_cache_dir('stanza', 'StanfordNLP')

# Cache root for test fixtures.  Mirrors the logic in stanza/tests/__init__.py:
#   TEST_DIR_BASE_NAME = 'stanza_test'
#   TEST_WORKING_DIR = os.getenv('STANZA_TEST_HOME') or
#                      user_cache_dir('stanza_test', 'StanfordNLP', __resources_version__)
# We use the unversioned root so we can show all versions, same as for models.
# Keep this in sync with stanza/tests/__init__.py if those constants change.
_STANZA_TEST_CACHE_ROOT = user_cache_dir('stanza_test', 'StanfordNLP')


def _dir_size_bytes(path):
    """Return the total byte-size of all regular files under *path*."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                total += os.path.getsize(fpath)
            except OSError:
                pass
    return total


def _human_size(nbytes):
    """Format *nbytes* as a human-readable string (e.g. '1.23 GB')."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024 or unit == 'TB':
            return f'{nbytes:.1f} {unit}'
        nbytes /= 1024


def _scan_model_dir(model_dir, version=None):
    """
    Scan a single *model_dir* and return a list of row dicts.

    Each dict has the keys: lang, lang_name, processor, package, path,
    size_bytes, and version (which may be None if unknown).
    """
    if not os.path.isdir(model_dir):
        return []

    try:
        resources = load_resources_json(model_dir)
    except (ResourcesFileNotFoundError, FileNotFoundError):
        resources = {}

    rows = []

    for lang in sorted(os.listdir(model_dir)):
        if lang in _SKIP_ENTRIES:
            continue
        lang_dir = os.path.join(model_dir, lang)
        if not os.path.isdir(lang_dir):
            continue

        # Follow alias chains in resources.json for the display name.
        lang_name = lang
        lang_entry = resources.get(lang, {})
        visited = set()
        while isinstance(lang_entry, dict) and 'alias' in lang_entry:
            alias = lang_entry['alias']
            if alias in visited:
                break
            visited.add(alias)
            lang_entry = resources.get(alias, {})
        if isinstance(lang_entry, dict):
            lang_name = lang_entry.get('lang_name', lang)

        for processor in sorted(os.listdir(lang_dir)):
            proc_dir = os.path.join(lang_dir, processor)
            if not os.path.isdir(proc_dir):
                continue

            pt_files = [f for f in os.listdir(proc_dir) if f.endswith('.pt')]

            if pt_files:
                for fname in sorted(pt_files):
                    fpath = os.path.join(proc_dir, fname)
                    try:
                        fsize = os.path.getsize(fpath)
                    except OSError:
                        fsize = 0
                    rows.append({
                        'version': version,
                        'lang': lang,
                        'lang_name': lang_name,
                        'processor': processor,
                        'package': fname[:-3],  # strip .pt
                        'path': fpath,
                        'size_bytes': fsize,
                    })
            else:
                # No .pt files directly; report the directory itself.
                rows.append({
                    'version': version,
                    'lang': lang,
                    'lang_name': lang_name,
                    'processor': processor,
                    'package': None,
                    'path': proc_dir,
                    'size_bytes': _dir_size_bytes(proc_dir),
                })

    return rows


def _scan_cache_root(cache_root, subdir='resources'):
    """
    Walk a versioned cache root and return all rows across all version subdirs.

    Expects the layout::

        <cache_root>/<version>/<subdir>/<lang>/<processor>/<package>.pt

    *subdir* is ``'resources'`` for the regular model cache and ``'models'``
    for the test fixture cache (``stanza_test``).
    """
    rows = []
    if os.path.isdir(cache_root):
        for entry in sorted(os.listdir(cache_root)):
            versioned_dir = os.path.join(cache_root, entry, subdir)
            if os.path.isdir(versioned_dir):
                rows.extend(_scan_model_dir(versioned_dir, version=entry))
    return rows


def list_installed(model_dir=DEFAULT_MODEL_DIR, print_table=True,
                   include_test_models=False):
    """
    Scan *model_dir* and report which Stanza models are installed locally.

    If *model_dir* is the default (i.e. ``STANZA_RESOURCES_DIR`` is not set),
    all versioned subdirectories found under the platform cache root are
    scanned, not just the current version.  This lets the user see models
    from older Stanza installs that may still be taking up disk space.

    When ``STANZA_RESOURCES_DIR`` is set the user has opted into a custom
    layout, so only that single directory is scanned.

    The directory layout expected under each versioned resources dir is::

        <version>/resources/
            resources.json          # optional; used for display names
            <lang>/                 # e.g. 'en', 'zh-hans'
                <processor>/        # e.g. 'pos', 'depparse', 'tokenize'
                    <package>.pt    # e.g. 'ewt.pt', 'gsd.pt'

    Parameters
    ----------
    model_dir : str
        Root directory that was passed as *model_dir* to ``stanza.download``
        (defaults to ``DEFAULT_MODEL_DIR``).
    print_table : bool
        When True (default) the results are printed as a formatted table in
        addition to being returned.
    include_test_models : bool
        When True, also scan the stanza_test cache (used by the Stanza test
        suite) and print those as a separate block.  Defaults to False since
        most users have no test fixtures installed.  The test rows are
        appended to the returned list after the regular model rows.

    Returns
    -------
    list of dict
        Each dict has the keys:

        ``version``
            Resources version string (e.g. ``'1.10.0'``), or ``None`` when
            scanning a custom ``STANZA_RESOURCES_DIR``.
        ``lang``
            Language code directory name (e.g. ``'en'``).
        ``lang_name``
            Human-readable language name from resources.json, or the lang
            code if resources.json is absent or the language is not listed.
        ``processor``
            Processor subdirectory name (e.g. ``'pos'``).
        ``package``
            Model filename without the ``.pt`` extension (e.g. ``'ewt'``),
            or ``None`` when the processor directory contains no ``.pt``
            files.
        ``path``
            Absolute path to the ``.pt`` file, or to the processor
            directory when *package* is ``None``.
        ``size_bytes``
            Size in bytes of the file (or total bytes under the directory
            when *package* is ``None``).
    """
    custom_dir = os.getenv('STANZA_RESOURCES_DIR')

    if custom_dir:
        # User has a custom layout; scan only that directory, version unknown.
        rows = _scan_model_dir(model_dir, version=None)
        if print_table:
            _print_installed_table(rows, model_dir, multi_version=False)
        return rows

    # Default layout: <cache_root>/<version>/resources/
    # Scan all version subdirectories we can find.
    all_rows = _scan_cache_root(_STANZA_CACHE_ROOT)

    if not all_rows:
        # Fallback: maybe the caller passed an explicit non-default model_dir
        all_rows = _scan_model_dir(model_dir, version=None)

    if include_test_models:
        test_rows = _scan_cache_root(_STANZA_TEST_CACHE_ROOT, subdir='models')
    else:
        test_rows = []

    if print_table:
        _print_installed_table(all_rows, _STANZA_CACHE_ROOT, multi_version=True)
        if include_test_models:
            if test_rows:
                _print_installed_table(test_rows, _STANZA_TEST_CACHE_ROOT,
                                       multi_version=True)
            else:
                print(f'\nNo test models found under {_STANZA_TEST_CACHE_ROOT}')
        combined = all_rows + test_rows
        total = sum(r['size_bytes'] for r in combined)
        label = 'All models + test fixtures' if test_rows else 'All versions'
        print(f'\n{label}: {len(combined)} model(s), {_human_size(total)} total')

    return all_rows + test_rows


def _format_rows(rows):
    """
    Return (headers, col_data, widths) for a flat list of model rows,
    without any version column (version is shown in the section header).
    """
    headers = ['Language', 'Code', 'Processor', 'Package', 'Size']
    col_data = [
        (
            r['lang_name'],
            r['lang'],
            r['processor'],
            r['package'] if r['package'] is not None else '(directory)',
            _human_size(r['size_bytes']),
        )
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row_vals in col_data:
        for i, val in enumerate(row_vals):
            widths[i] = max(widths[i], len(val))
    return headers, col_data, widths


def _print_section(rows, header_path, widths, sep='  '):
    """Print one version section: path header, table body, subtotal footer."""
    headers, col_data, _ = _format_rows(rows)

    # Recompute a rule sized to *widths* (may be wider than this section alone
    # when called from a multi-version print that pre-computed shared widths).
    rule = sep.join('-' * w for w in widths)
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))

    print(f'\n{header_path}')
    print(header_line)
    print(rule)

    prev_lang = None
    for row_vals in col_data:
        if prev_lang is not None and row_vals[1] != prev_lang:
            print()
        prev_lang = row_vals[1]
        print(sep.join(v.ljust(widths[i]) for i, v in enumerate(row_vals)))

    subtotal = sum(r['size_bytes'] for r in rows)
    print(f'  ({len(rows)} model(s), {_human_size(subtotal)})')


def _print_installed_table(rows, root_dir, multi_version=False):
    """Pretty-print the installed models as an aligned table."""
    if not rows:
        print(f'No models found under {root_dir}')
        return

    current = __resources_version__
    sep = '  '

    if multi_version:
        # Group rows by version and compute column widths across all sections
        # so that the tables line up visually.
        _, _, widths = _format_rows(rows)

        print(f'\nInstalled Stanza models under: {root_dir}')

        versions_seen = []
        for version, version_rows in groupby(rows, key=lambda r: r['version']):
            version_rows = list(version_rows)
            versions_seen.append(version)
            label = ' (current)' if version == current else ''
            path = os.path.join(root_dir, version or '', 'resources')
            _print_section(version_rows, f'{path}{label}', widths, sep)

    else:
        _, _, widths = _format_rows(rows)
        _print_section(rows, root_dir, widths, sep)
        print()  # trailing newline for single-dir output


if __name__ == '__main__':
    import argparse
    import sys

    # Set up a basic logger so any warnings from _scan_model_dir are visible.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING,
                        format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='List locally installed Stanza models.'
    )
    parser.add_argument(
        '--model-dir', default=None,
        help=(
            'Root model directory to scan (default: auto-detected from '
            'STANZA_RESOURCES_DIR or the platform cache).'
        )
    )
    parser.add_argument(
        '--include-test-models', action='store_true',
        help=(
            'Also scan the stanza_test cache used by the Stanza test suite '
            'and display those models in a separate block.'
        )
    )
    args = parser.parse_args()

    kwargs = {}
    if args.model_dir is not None:
        kwargs['model_dir'] = args.model_dir
    if args.include_test_models:
        kwargs['include_test_models'] = True

    list_installed(**kwargs)
