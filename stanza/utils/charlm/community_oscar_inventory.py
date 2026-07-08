"""
community_oscar_inventory.py

List the snapshots and languages available in the Community-OSCAR dataset
on HuggingFace, without downloading any text.  Uses the HF Hub file-listing
API to inspect the repo's directory structure.

Usage
-----
# List everything (human-readable language names shown by default)
python community_oscar_inventory.py

# Show only snapshots that contain a specific language
python community_oscar_inventory.py --lang sd

# Show only languages available in a specific snapshot
python community_oscar_inventory.py --snapshot 2024-22

# Show counts as a table (snapshots × languages)
python community_oscar_inventory.py --table

# Show bare codes instead of names (names are shown by default)
python community_oscar_inventory.py --codes_only

# Machine-readable JSON output
python community_oscar_inventory.py --json

General options
---------------
    --hf_token TOKEN       HF access token (or set HF_TOKEN env var)
    --lang LANG            Filter to a specific language code
    --snapshot SNAP        Filter to a specific snapshot
    --table                Print a snapshot × language presence table
    --json                 Print raw inventory as JSON instead of human text
    --codes_only           Show bare codes instead of human-readable names
                            (names are the default; requires Stanza on the
                            path, for stanza.models.common.constant)
    --timeout SECONDS      Fail fast if the HF API doesn't respond in time (default: 15)
    --no_cache             Skip the on-disk inventory cache for this run
    --refresh_cache        Ignore cached inventory, refetch, and update the cache
    --cache_ttl_hours N    How long a cached inventory stays valid (default: 24)

Requirements
------------
    pip install huggingface_hub
    Stanza must be importable (stanza.models.common.constant) for language
    names; falls back to codes automatically if unavailable. A code with no
    entry in Stanza's table prints as 'unknown' rather than the bare code.
"""

import argparse
import json
import logging
import os
import re
import socket
import sys
import time
from collections import defaultdict
from typing import Optional

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: 'huggingface_hub' package not found.  Install with:  pip install huggingface_hub",
          file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import RepoFolder
except ImportError:
    try:
        from huggingface_hub.hf_api import RepoFolder
    except ImportError:
        print(
            "ERROR: This script requires huggingface_hub >= 0.20 "
            "(needs RepoFolder / list_repo_tree, added in that release). "
            "Run:  pip install -U huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

try:
    from stanza.models.common.constant import langcode_to_lang, pretty_langcode_to_lang
    _HAVE_STANZA_CONSTANT = True
except ImportError:
    _HAVE_STANZA_CONSTANT = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ID = "oscar-corpus/community-oscar"
REPO_TYPE = "dataset"

# Where to cache the inventory result. Lives alongside HF's own blob cache
# so it's in a predictable, already-known location, but it's a separate
# file we manage ourselves -- huggingface_hub's disk cache only stores
# downloaded file *blobs*, never directory-listing API responses, so
# list_repo_tree() calls always hit the network fresh with no caching of
# their own.
_CACHE_DIR = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "community_oscar_inventory",
)
_CACHE_FILE = os.path.join(_CACHE_DIR, "inventory_cache.json")
_DEFAULT_CACHE_TTL_HOURS = 24

# Community-OSCAR meta files follow the pattern:
#   data/<snapshot>/<lang>_meta/<lang>_meta_part-XXXXX-....jsonl.zst
# We only need to see the directory names; the filename itself is irrelevant.
_META_PATH_RE = re.compile(r"^data/([^/]+)/([a-z]{2,3})_meta/")

# Codes where Community-OSCAR's own meaning either isn't in Stanza's table
# at all, or differs from what Stanza's table would say. These take
# precedence over both the 'multi' special case logic and the regular
# Stanza lookup.
#   - 'sh' / 'bh': Community-OSCAR (like the rest of OSCAR) classifies
#     documents with fastText's lid.176 language-ID model, and both codes
#     are real labels in that model's 176-language set -- not crawl noise.
#     Stanza's table omits them because they aren't used as UD treebank
#     codes, but that's a UD-specific decision, not a statement about what
#     the crawl bucket actually contains. 'sh' = Serbo-Croatian (the old
#     Wikipedia-era macro code spanning Bosnian/Croatian/Montenegrin/
#     Serbian); 'bh' = Bihari languages (ISO 639-1 collective code).
#   - 'zh': Stanza's table maps this to "Chinese (Simplified)" for UD
#     treebank purposes (UD splits zh-hans/zh-hant), but Community-OSCAR's
#     'zh' bucket is unsplit -- it's just "Chinese", not specifically the
#     simplified-script variant.
_LANG_NAME_OVERRIDES = {
    "sh": "Serbo-Croatian",
    "bh": "Bihari",
    "zh": "Chinese",
}


def lang_name(code: str) -> str:
    """
    Human-readable name for an OSCAR language code, via Stanza's own
    UD-aligned language table (stanza.models.common.constant).

    Special cases:
      - 'multi' is not a real language code (Community-OSCAR doesn't
        actually emit it, but other Stanza tooling sometimes uses it as a
        placeholder for multilingual data) -- pass it through as-is rather
        than reporting it as unknown.
      - _LANG_NAME_OVERRIDES above takes precedence over the general
        lookup for codes where Community-OSCAR's meaning needs to differ
        from Stanza's UD-oriented table.
      - Any other code not in Stanza's table returns 'unknown' rather than
        silently echoing the bare code back (which `pretty_langcode_to_lang`
        does on its own, and which can look like a real answer rather than
        a missing one -- e.g. a hypothetical 'xx_test' code would otherwise
        print as 'xx test').
    """
    if code == "multi":
        return "multi"
    if code in _LANG_NAME_OVERRIDES:
        return _LANG_NAME_OVERRIDES[code]
    if not _HAVE_STANZA_CONSTANT:
        return code
    # langcode_to_lang() returns the bare code unchanged when it has no
    # entry for it, so we check that first to distinguish "not found" from
    # a real lookup, rather than trusting pretty_langcode_to_lang's cosmetic
    # passthrough on an unrecognized code.
    if langcode_to_lang(code) == code:
        return "unknown"
    return pretty_langcode_to_lang(code)


def _handle_api_error(e: Exception, timeout: int) -> None:
    """Translate a dataset_info() failure into a clear, actionable message."""
    msg = str(e)
    if "401" in msg or "authentication" in msg.lower():
        log.error("HF token was rejected (401). Check that it's valid and not expired.")
    elif "403" in msg or "gated" in msg.lower():
        log.error(
            "Access denied (403). You likely haven't accepted the dataset's "
            "access agreement yet:\n"
            "  https://huggingface.co/datasets/oscar-corpus/community-oscar"
        )
    else:
        log.error(f"Could not reach dataset metadata within {timeout}s: {e}")


def preflight_check(hf_token: Optional[str], timeout: int) -> None:
    """
    Fail fast, with a clear message, instead of hanging indefinitely when
    there's no token or no network. (The actual auth/access-grant check
    happens as a side effect of the single dataset_info() call in
    fetch_inventory — no separate API call is made here, to keep total
    request count at one.)
    """
    if not hf_token:
        log.error(
            "No HF token provided. Community-OSCAR is a gated dataset — "
            "an access token is required even just to list files.\n"
            "  1. Accept the access agreement at "
            "https://huggingface.co/datasets/oscar-corpus/community-oscar\n"
            "  2. Get a token at https://huggingface.co/settings/tokens\n"
            "  3. Pass --hf_token TOKEN or set the HF_TOKEN environment variable."
        )
        sys.exit(1)

    # Basic reachability — this is what actually hangs forever otherwise.
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection(("huggingface.co", 443), timeout=timeout)
    except OSError as e:
        log.error(
            f"Cannot reach huggingface.co within {timeout}s ({e}). "
            "Check your network connection / proxy settings."
        )
        sys.exit(1)


def _load_cache(ttl_hours: float) -> Optional[dict[str, set[str]]]:
    """Return cached inventory if present and within TTL, else None."""
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        age_hours = (time.time() - payload["fetched_at"]) / 3600
        if age_hours > ttl_hours:
            log.info(f"Cache is {age_hours:.1f}h old (TTL {ttl_hours}h) — refreshing.")
            return None
        log.info(
            f"Using cached inventory from {age_hours:.1f}h ago "
            f"({_CACHE_FILE}). Pass --refresh-cache to force a refetch."
        )
        return {snap: set(langs) for snap, langs in payload["inventory"].items()}
    except (OSError, json.JSONDecodeError, KeyError) as e:
        log.warning(f"Could not read cache ({e}) — refetching.")
        return None


def _save_cache(inventory: dict[str, set[str]]) -> None:
    """Best-effort write of the inventory to the on-disk cache."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        payload = {
            "fetched_at": time.time(),
            "inventory": {snap: sorted(langs) for snap, langs in inventory.items()},
        }
        # Write to a temp file then rename, so a crash mid-write can't
        # leave a corrupt cache file behind for the next run to choke on.
        tmp_path = _CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, _CACHE_FILE)
    except OSError as e:
        log.warning(f"Could not write cache ({e}) — continuing without caching.")


def fetch_inventory(
    hf_token: Optional[str],
    timeout: int = 15,
    read_cache: bool = True,
    write_cache: bool = True,
    cache_ttl_hours: float = _DEFAULT_CACHE_TTL_HOURS,
) -> dict[str, set[str]]:
    """
    Return {snapshot: {lang, ...}} via a small, bounded number of HF API calls.

    This deliberately avoids two approaches that turned out to be unreliable
    for a repo this size:

      - dataset_info()'s `siblings` field: a single call, but the Hub API
        silently truncates the file list for large repos rather than
        erroring or paginating. Confirmed truncation mid-snapshot
        (e.g. cutting off partway through 2023-50's language list) and is
        consistent with a known, longstanding behavior reported upstream
        (huggingface/huggingface_hub#1814): `siblings` from dataset_info
        is not guaranteed complete for large repos.

      - list_repo_tree(recursive=True) from the repo root: correct and
        complete, but walks every directory including the individual
        *_meta/ folders, issuing one paginated request per language per
        snapshot — hundreds of calls.

    Instead: list `data/` non-recursively (1 call) to get snapshot folder
    names, then list each snapshot folder non-recursively (1 call each) to
    get language folder names. We never list inside a `{lang}_meta/`
    folder, since the language code is already in its directory name and
    we don't need individual filenames. Total calls ≈ 1 + (# snapshots).

    Results are cached to disk (see _CACHE_FILE) so reruns within
    cache_ttl_hours skip the network entirely. Pass read_cache=False to
    ignore any existing cache for this run (e.g. to force a refresh), or
    write_cache=False to avoid persisting this run's result.
    """
    if read_cache:
        cached = _load_cache(cache_ttl_hours)
        if cached is not None:
            return cached

    preflight_check(hf_token, timeout)

    api = HfApi(token=hf_token)
    log.info(f"Fetching snapshot list from {REPO_ID}/data (1 call) …")

    try:
        top_level = list(api.list_repo_tree(
            repo_id=REPO_ID,
            path_in_repo="data",
            recursive=False,
            repo_type=REPO_TYPE,
            token=hf_token,
        ))
    except Exception as e:
        _handle_api_error(e, timeout)
        sys.exit(1)

    snapshots = sorted(
        os.path.basename(item.path)
        for item in top_level
        if isinstance(item, RepoFolder)
    )
    log.info(f"Found {len(snapshots)} snapshot(s): {', '.join(snapshots)}")

    inventory: dict[str, set[str]] = defaultdict(set)
    for i, snapshot in enumerate(snapshots, 1):
        snap_path = f"data/{snapshot}"
        log.info(f"  [{i}/{len(snapshots)}] listing {snap_path} …")
        try:
            children = list(api.list_repo_tree(
                repo_id=REPO_ID,
                path_in_repo=snap_path,
                recursive=False,
                repo_type=REPO_TYPE,
                token=hf_token,
            ))
        except Exception as e:
            log.warning(f"  Failed to list {snap_path}: {e}  (skipping this snapshot)")
            continue

        for item in children:
            if not isinstance(item, RepoFolder):
                continue
            name = os.path.basename(item.path)
            # Folder names look like '<lang>_meta', e.g. 'sd_meta'.
            if name.endswith("_meta"):
                lang = name[: -len("_meta")]
                inventory[snapshot].add(lang)

    log.info(
        f"Parsed → {len(inventory)} snapshot(s), "
        f"{len({l for langs in inventory.values() for l in langs})} unique language(s), "
        f"~{1 + len(snapshots)} total API calls"
    )
    result = dict(inventory)
    if write_cache:
        _save_cache(result)
    return result


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_human(inventory: dict[str, set[str]],
                filter_lang: Optional[str],
                filter_snap: Optional[str],
                show_names: bool = False) -> None:
    snapshots = sorted(inventory, reverse=True)
    if filter_snap:
        snapshots = [s for s in snapshots if s == filter_snap]

    def fmt(code: str) -> str:
        return f"{code} ({lang_name(code)})" if show_names else code

    for snap in snapshots:
        langs = sorted(inventory[snap])
        if filter_lang:
            if filter_lang not in langs:
                continue
            print(f"{filter_lang}:{snap}")
        else:
            lang_str = "  ".join(fmt(l) for l in langs)
            print(f"{snap}  [{len(langs)} langs]")
            print(f"    {lang_str}")


def print_table(inventory: dict[str, set[str]],
                filter_lang: Optional[str],
                filter_snap: Optional[str],
                show_names: bool = False) -> None:
    """Print a compact presence matrix: rows = snapshots, columns = languages."""
    all_snapshots = sorted(inventory)
    all_langs = sorted({l for langs in inventory.values() for l in langs})

    if filter_snap:
        all_snapshots = [s for s in all_snapshots if s == filter_snap]
    if filter_lang:
        all_langs = [l for l in all_langs if l == filter_lang]

    if not all_snapshots or not all_langs:
        print("(no results after filtering)")
        return

    if show_names:
        # Names are long; a wide matrix is unreadable, so print a legend
        # instead of trying to cram names into column headers.
        print("Language legend:")
        for l in all_langs:
            print(f"  {l:6s} {lang_name(l)}")
        print()

    col_w = max(len(l) for l in all_langs)
    snap_w = max(len(s) for s in all_snapshots)

    # Header
    header = " " * (snap_w + 2) + "  ".join(l.ljust(col_w) for l in all_langs)
    print(header)
    print("-" * len(header))

    for snap in all_snapshots:
        langs_in_snap = inventory.get(snap, set())
        row = snap.ljust(snap_w) + "  "
        row += "  ".join(
            ("✓" if l in langs_in_snap else "·").ljust(col_w)
            for l in all_langs
        )
        print(row)


def print_json(inventory: dict[str, set[str]],
               filter_lang: Optional[str],
               filter_snap: Optional[str],
               show_names: bool = False) -> None:
    out = {}
    for snap, langs in sorted(inventory.items()):
        if filter_snap and snap != filter_snap:
            continue
        lang_list = sorted(langs)
        if filter_lang:
            if filter_lang not in lang_list:
                continue
            lang_list = [filter_lang]
        if show_names:
            out[snap] = {l: lang_name(l) for l in lang_list}
        else:
            out[snap] = lang_list
    print(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        metavar="TOKEN",
        help="HuggingFace access token. Defaults to HF_TOKEN env var.",
    )
    p.add_argument(
        "--lang",
        default=None,
        metavar="CODE",
        help="Filter output to a single ISO 639 language code, e.g. sd, ur, sl.",
    )
    p.add_argument(
        "--snapshot",
        default=None,
        metavar="SNAP",
        help="Filter output to a single snapshot, e.g. 2024-22.",
    )
    p.add_argument(
        "--codes_only",
        action="store_true",
        help=(
            "Show bare language codes instead of human-readable names. "
            "Names are shown by default via Stanza's language table "
            "(stanza.models.common.constant); falls back to codes "
            "automatically if Stanza isn't importable."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=15,
        metavar="SECONDS",
        help="Network/API timeout in seconds before failing fast. Default: 15.",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help=(
            "Skip the on-disk inventory cache entirely for this run "
            "(neither reads nor writes it)."
        ),
    )
    p.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Ignore any cached inventory and refetch, but still update the cache.",
    )
    p.add_argument(
        "--cache_ttl_hours",
        type=float,
        default=_DEFAULT_CACHE_TTL_HOURS,
        metavar="HOURS",
        help=(
            f"How long a cached inventory stays valid before refetching. "
            f"Default: {_DEFAULT_CACHE_TTL_HOURS}."
        ),
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--table",
        action="store_true",
        help="Print a snapshot × language presence matrix.",
    )
    mode.add_argument(
        "--json",
        action="store_true",
        help="Print inventory as JSON (useful for piping into other tools).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    show_names = not args.codes_only

    if show_names and not _HAVE_STANZA_CONSTANT:
        log.warning(
            "Language names requested (default) but "
            "stanza.models.common.constant could not be imported "
            "(Stanza not on the Python path / not installed) — showing "
            "codes only. Run from within the Stanza repo, or pass "
            "--codes_only to silence this warning."
        )
        show_names = False  # nothing to show; avoid passing a dead flag downstream

    # --no_cache: skip cache entirely (no read, no write)
    # --refresh_cache: ignore existing cache but still write a fresh one
    # default: read if fresh enough, always write after a real fetch
    inventory = fetch_inventory(
        args.hf_token,
        timeout=args.timeout,
        read_cache=not args.no_cache and not args.refresh_cache,
        write_cache=not args.no_cache,
        cache_ttl_hours=args.cache_ttl_hours,
    )

    if args.json:
        print_json(inventory, args.lang, args.snapshot, show_names=show_names)
    elif args.table:
        print_table(inventory, args.lang, args.snapshot, show_names=show_names)
    else:
        print_human(inventory, args.lang, args.snapshot, show_names=show_names)


if __name__ == "__main__":
    main()
