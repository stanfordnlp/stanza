"""
community_oscar_dedup.py

Download one or more Community-OSCAR slices, deduplicate across all of them
(by exact URL and by MinHash LSH near-duplicate detection), and write
plain-text output files suitable for Stanza's make_lm_data.py charlm pipeline.

Deduplication is stateful across slices: documents seen in an earlier slice
are suppressed in later ones, so the order of --slices matters (first
occurrence wins).

Usage
-----
python community_oscar_dedup.py \
    --slices sd:2024-22 sd:2024-18 ur:2024-22 \
    --output_dir /u/nlp/data/oscar_deduped \
    --minhash_threshold 0.7 \
    --hf_token YOUR_TOKEN   # or set HF_TOKEN env var

Output
------
One file per (lang, snapshot) pair, e.g.:
    /u/nlp/data/oscar_deduped/sd_2024-22.txt
    /u/nlp/data/oscar_deduped/sd_2024-18.txt
    /u/nlp/data/oscar_deduped/ur_2024-22.txt

Each file contains one document per line (newlines within a document are
replaced with a space), ready for make_lm_data.py --files.

Requirements
------------
    pip install datasketch huggingface_hub zstandard

Near-duplicate detection
------------------------
Uses MinHash LSH (datasketch library) on word-unigram sets, giving O(1)
query time regardless of corpus size. This replaces TLSH-based near-dedup,
which was O(n) per query and became prohibitively slow for large languages
like Urdu or Pashto (800+ seconds per slice at 200k seen documents).

The Jaccard similarity threshold (--minhash_threshold, default 0.7)
controls how similar two documents must be to be considered near-duplicates:
  - 1.0 = identical content only
  - 0.8 = near-identical (minor word changes, punctuation differences)
  - 0.7 = substantially same content (boilerplate additions, different lede)
  - 0.5 = significant overlap (50%+ shared content)
Lower values catch more near-duplicates but increase false positives.

Empirical validation
--------------------
The choice of dedup method has a measurable effect on downstream task
performance.  In experiments on Urdu dependency parsing and Sindhi NER,
MinHash at threshold 0.7 produced meaningfully better results than the
earlier TLSH-based dedup at threshold 50, even controlling for training
compute:

Urdu dependency parsing (dev / test LAS):
  MuRIL (large multilingual baseline):  90.12 / 91.65
  TLSH dedup,   short training:         88.96 / 90.95
  TLSH dedup,   long training (4x):     89.14 / 90.92
  MinHash dedup, short training:        89.58 / 91.44  <- beats TLSH long
  MinHash dedup, long training (4x):    89.32 / 91.01

The MinHash short run (5 epochs on the larger corpus) outperforms the
TLSH long run (80 epochs on the smaller corpus), suggesting that data
quality was the primary bottleneck rather than training duration or
model capacity.

Sindhi NER (dev / test F1):
  MuRIL-large (large multilingual baseline):  87.77 / 88.37
  TLSH dedup,   6-layer model:                84.49 / 84.86
  TLSH dedup,   12-layer model:               84.61 / 85.43
  MinHash dedup, 6-layer model:               85.05 / 85.88
  MinHash dedup, 8-layer model:               85.69 / 86.49  <- beats TLSH 12L

Again, the smaller MinHash model outperforms the larger TLSH model,
closing the gap to MuRIL-large from ~3.5 to ~2.0 F1 points.

To calibrate the threshold for a new language, use the companion script
community_oscar_inspect_similarity.py, which reports pairwise Jaccard
and TLSH similarity distributions on a random sample of document pairs.
A threshold of 0.7 has worked well for Urdu and Sindhi (validated on
downstream dependency parsing and NER respectively), and produced
well-diversified corpora for Pashto and Western Punjabi (verified via
inspect_similarity.py; downstream task evaluation pending). Languages
with very narrow web presence (few sources, heavy boilerplate) may
benefit from a slightly higher threshold.

HF access
---------
You must have accepted the Community-OSCAR gating agreement on HF.
Pass --hf_token or export HF_TOKEN before running.

Implementation note
--------------------
This deliberately does NOT use `datasets.load_dataset(streaming=True)`.
That path streams JSONL through `datasets`' PyArrow-backed batching layer,
which has a known class of bug where internal Arrow array slicing can go
out of bounds on certain inputs -- this isn't a Python exception you can
catch, it's a native `Check failed` assertion that aborts the whole
process (see e.g. huggingface/datasets#5531 and similar issues). Instead,
files matching each slice are listed and downloaded individually via
huggingface_hub (which also gives us real on-disk caching of the raw
files, a free side benefit), decompressed with the `zstandard` library,
and parsed line-by-line with the stdlib `json` module -- entirely outside
PyArrow's code path, so a malformed record raises a catchable Python
exception (skip and log) instead of crashing the interpreter.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print(
        "ERROR: 'datasketch' package not found.\n"
        "  Install with:  pip install datasketch",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("ERROR: 'huggingface_hub' package not found.  Install with:  pip install huggingface_hub",
          file=sys.stderr)
    sys.exit(1)

try:
    import zstandard
except ImportError:
    print("ERROR: 'zstandard' package not found.  Install with:  pip install zstandard",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationState:
    """Mutable state shared across all slices."""
    seen_urls: set = field(default_factory=set)
    # MinHash LSH parameters
    minhash_threshold: float = 0.7   # Jaccard similarity above which docs are near-dups
    minhash_num_perm: int = 128      # number of hash permutations (more = more accurate)

    # stats
    total_docs: int = 0
    dropped_url: int = 0
    dropped_minhash: int = 0
    kept: int = 0
    short_docs: int = 0   # docs too short to build a meaningful MinHash

    def __post_init__(self):
        self._lsh = MinHashLSH(
            threshold=self.minhash_threshold,
            num_perm=self.minhash_num_perm,
        )
        self._doc_counter = 0  # used to generate unique keys for the LSH index

    def _make_minhash(self, text: str) -> Optional[MinHash]:
        """
        Build a MinHash from word unigrams.

        Word unigrams (bag-of-words) are fast, language-agnostic (works on any
        whitespace-segmented script including Perso-Arabic), and robust to the
        kinds of near-duplicate variation common in web crawls: boilerplate
        additions, minor word substitutions, different ledes on the same article.
        Character n-grams give similar recall but are ~4x slower to compute.
        Returns None if the document is too short to hash meaningfully.
        """
        words = text.split()
        if len(words) < 5:
            return None
        m = MinHash(num_perm=self.minhash_num_perm)
        seen_words = set()
        for w in words:
            h = hash(w)
            if h not in seen_words:
                seen_words.add(h)
                m.update(h.to_bytes(8, 'little', signed=True))
        return m

    def is_duplicate(self, url: Optional[str], text: str) -> str:
        """
        Return 'url', 'minhash', or '' (not a duplicate).
        Side-effect: if not a duplicate, registers the URL and MinHash.

        Takes the raw document text rather than a pre-computed hash, since
        MinHash construction is fast (0.5ms) and the LSH index needs the
        MinHash object for both querying and insertion.
        """
        self.total_docs += 1

        # 1. Exact URL dedup
        if url:
            if url in self.seen_urls:
                self.dropped_url += 1
                return "url"

        # 2. MinHash LSH near-dedup
        m = self._make_minhash(text)
        if m is None:
            self.short_docs += 1
        else:
            results = self._lsh.query(m)
            if results:
                self.dropped_minhash += 1
                return "minhash"

        # Not a duplicate: register both signals
        if url:
            self.seen_urls.add(url)
        if m is not None:
            key = f'doc_{self._doc_counter}'
            self._doc_counter += 1
            self._lsh.insert(key, m)

        self.kept += 1
        return ""

    def report(self) -> str:
        pct = lambda n: f"{100*n/max(self.total_docs,1):.1f}%"
        return (
            f"total={self.total_docs}  kept={self.kept} ({pct(self.kept)})  "
            f"dropped_url={self.dropped_url} ({pct(self.dropped_url)})  "
            f"dropped_minhash={self.dropped_minhash} ({pct(self.dropped_minhash)})  "
            f"short_docs={self.short_docs} ({pct(self.short_docs)})"
        )


@dataclass
class SliceStats:
    """Summary numbers for one (lang, snapshot) slice, for the final report table."""
    lang: str
    snapshot: str
    total_docs: int = 0          # docs seen in this slice (post-parse, pre-dedup)
    kept_docs: int = 0           # docs from this slice actually written out
    dropped_url: int = 0
    dropped_minhash: int = 0
    short_docs: int = 0          # docs too short for meaningful MinHash
    parse_errors: int = 0
    word_count: int = 0          # whitespace-split word count of kept docs
    char_count: int = 0          # character count of kept docs (post-flatten)
    output_bytes: int = 0
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def parse_slice(s: str) -> tuple[str, str]:
    """Parse 'lang:snapshot' → ('lang', 'snapshot'), e.g. 'sd:2024-22'."""
    parts = s.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            f"Slice must be 'lang:snapshot', e.g. 'sd:2024-22'.  Got: {s!r}"
        )
    return parts[0], parts[1]


def _list_slice_files(api: HfApi, lang: str, snapshot: str, hf_token: Optional[str]) -> list[str]:
    """
    Return repo-relative paths of every *.jsonl.zst file for this slice,
    e.g. ['data/2024-38/sd_meta/sd_meta.jsonl.zst', ...]. Some slices ship
    as a single file, others as multiple part-NNNNN shards -- both are
    handled the same way since we just iterate whatever's listed.
    """
    folder = f"data/{snapshot}/{lang}_meta"
    items = api.list_repo_tree(
        repo_id="oscar-corpus/community-oscar",
        path_in_repo=folder,
        recursive=False,
        repo_type="dataset",
        token=hf_token,
    )
    return sorted(
        item.path for item in items
        if getattr(item, "path", "").endswith(".jsonl.zst")
    )


def _iter_jsonl_zst(path: str) -> Iterator[tuple[Optional[dict], Optional[str]]]:
    """
    Stream-decompress a .jsonl.zst file and parse it line by line.

    Yields (doc, error) pairs: doc is the parsed dict on success (error is
    None), or doc is None and error holds a message on a per-line failure
    -- letting the caller skip just that one malformed line and keep going,
    rather than the whole file/process dying the way the PyArrow streaming
    path did.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            # zstd frames don't align with text lines, so wrap in a text
            # stream that buffers correctly across decompression chunks.
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line_num, line in enumerate(text_stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), None
                except json.JSONDecodeError as e:
                    yield None, f"line {line_num}: {e}"


def process_slice(
    lang: str,
    snapshot: str,
    state: DeduplicationState,
    output_dir: Path,
    hf_token: Optional[str],
) -> SliceStats:
    """Download one Community-OSCAR slice's files, dedup against shared state, write output."""

    stats = SliceStats(lang=lang, snapshot=snapshot)

    log.info(f"Listing files for slice  lang={lang}  snapshot={snapshot}")
    api = HfApi(token=hf_token)
    try:
        repo_files = _list_slice_files(api, lang, snapshot, hf_token)
    except Exception as e:
        log.error(f"Failed to list files for slice {lang}:{snapshot} — {e}")
        return stats

    if not repo_files:
        log.warning(f"No .jsonl.zst files found for slice {lang}:{snapshot} -- skipping.")
        return stats
    log.info(f"  Found {len(repo_files)} file(s) for {lang}:{snapshot}")

    out_path = output_dir / f"{lang}_{snapshot}.txt"
    t0 = time.time()
    # Snapshot the shared dedup state's per-reason counters before this
    # slice starts, so we can compute *this slice's* drop counts even
    # though `state` accumulates across every slice in the run.
    dropped_url_before = state.dropped_url
    dropped_minhash_before = state.dropped_minhash
    short_docs_before = state.short_docs

    with out_path.open("w", encoding="utf-8") as fout:
        for repo_file in repo_files:
            log.info(f"  Downloading {repo_file} …")
            try:
                local_path = hf_hub_download(
                    repo_id="oscar-corpus/community-oscar",
                    filename=repo_file,
                    repo_type="dataset",
                    token=hf_token,
                )
            except Exception as e:
                log.error(f"  Failed to download {repo_file} — {e}  (skipping this file)")
                continue

            for doc, err in _iter_jsonl_zst(local_path):
                if err is not None:
                    stats.parse_errors += 1
                    if stats.parse_errors <= 5:
                        log.warning(f"  Skipping malformed JSON in {repo_file} — {err}")
                    elif stats.parse_errors == 6:
                        log.warning(f"  (suppressing further malformed-JSON warnings for {repo_file})")
                    continue

                stats.total_docs += 1

                # Community-OSCAR meta fields
                # Primary text field is 'content'; URL is in 'warc_headers' or 'metadata'
                text = doc.get("content", "")
                if not text or not text.strip():
                    continue

                url = _extract_url(doc)
                reason = state.is_duplicate(url, text)
                if reason:
                    continue

                # Flatten to a single line (charlm pipeline expects one doc per line)
                line = " ".join(text.split())
                fout.write(line + "\n")
                stats.kept_docs += 1
                stats.word_count += len(line.split())
                stats.char_count += len(line)

                if stats.total_docs % 10_000 == 0:
                    elapsed = time.time() - t0
                    log.info(
                        f"  [{lang}:{snapshot}]  docs={stats.total_docs}  "
                        f"kept={stats.kept_docs}  elapsed={elapsed:.0f}s"
                    )

    stats.dropped_url = state.dropped_url - dropped_url_before
    stats.dropped_minhash = state.dropped_minhash - dropped_minhash_before
    stats.short_docs = state.short_docs - short_docs_before
    stats.elapsed_sec = time.time() - t0
    try:
        stats.output_bytes = out_path.stat().st_size
    except OSError:
        pass

    log.info(
        f"Finished [{lang}:{snapshot}]  "
        f"total={stats.total_docs}  kept={stats.kept_docs}  parse_errors={stats.parse_errors}  "
        f"elapsed={stats.elapsed_sec:.0f}s  → {out_path}"
    )
    return stats


def _extract_url(doc: dict) -> Optional[str]:
    """
    Community-OSCAR stores the source URL in different places depending on
    the snapshot vintage.  Try the known locations in order.
    """
    # Direct field (newer snapshots)
    if "url" in doc and doc["url"]:
        return doc["url"]
    # Nested in warc_headers dict
    warc = doc.get("warc_headers") or {}
    if isinstance(warc, dict):
        for key in ("warc-target-uri", "WARC-Target-URI", "uri"):
            if warc.get(key):
                return warc[key]
    # Nested in metadata dict
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("url"):
        return meta["url"]
    return None



def _human_bytes(n: int) -> str:
    """Format a byte count compactly, e.g. 1536000 -> '1.5MB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def print_summary(all_stats: list[SliceStats]) -> None:
    """
    Print a per-slice table plus a running-cumulative column, so it's easy
    to see whether later slices are still adding meaningfully new data or
    diminishing returns have set in -- the question that originally
    motivated this summary.
    """
    if not all_stats:
        return

    header = (
        f"{'slice':<18} {'kept':>9} {'words':>11} {'chars':>12} "
        f"{'size':>8} {'dup_url':>8} {'dup_mh':>7} {'short%':>7} {'errs':>6} "
        f"{'cum_kept':>9} {'cum_words':>12}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    cum_kept = 0
    cum_words = 0
    cum_chars = 0
    cum_bytes = 0
    any_high_missing = False

    for s in all_stats:
        cum_kept += s.kept_docs
        cum_words += s.word_count
        cum_chars += s.char_count
        cum_bytes += s.output_bytes

        short_pct = (
            100 * s.short_docs / s.total_docs
            if s.total_docs > 0 else 0.0
        )
        if short_pct > 50:
            any_high_missing = True

        slice_label = f"{s.lang}:{s.snapshot}"
        print(
            f"{slice_label:<18} {s.kept_docs:>9,} {s.word_count:>11,} "
            f"{s.char_count:>12,} {_human_bytes(s.output_bytes):>8} "
            f"{s.dropped_url:>8,} {s.dropped_minhash:>7,} {short_pct:>6.1f}% "
            f"{s.parse_errors:>6,} {cum_kept:>9,} {cum_words:>12,}"
        )

    print(sep)
    print(
        f"{'TOTAL':<18} {cum_kept:>9,} {cum_words:>11,} {cum_chars:>12,} "
        f"{_human_bytes(cum_bytes):>8}"
    )
    print()

    if any_high_missing:
        print(
            "WARNING: at least one slice has >50% of documents too short for "
            "MinHash (see short% column). Near-duplicate detection is disabled "
            "for those documents -- they are kept but not indexed."
        )
        print()

    # Marginal-value hint: how much each successive slice added relative
    # to the running total before it, so diminishing returns are visible
    # at a glance rather than something you have to compute by hand.
    if len(all_stats) > 1:
        print("Marginal contribution per slice (relative to cumulative total so far):")
        running = 0
        for s in all_stats:
            pct = (s.word_count / running * 100) if running > 0 else float("inf")
            pct_str = "first slice" if running == 0 else f"+{pct:.1f}%"
            print(f"  {s.lang}:{s.snapshot:<12} {pct_str}")
            running += s.word_count
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--slices",
        nargs="+",
        required=True,
        metavar="LANG:SNAPSHOT",
        help=(
            "One or more slices to process, in the form lang:snapshot, "
            "e.g.  --slices sd:2024-22 sd:2024-18 ur:2024-22"
        ),
    )
    p.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Directory for output .txt files (created if absent).",
    )
    p.add_argument(
        "--minhash_threshold",
        type=float,
        default=0.7,
        metavar="FLOAT",
        help=(
            "Jaccard similarity threshold above which two documents are "
            "considered near-duplicates (and the later one dropped). "
            "Jaccard is computed over word-unigram sets: 1.0 = identical, "
            "0.0 = completely different. Higher values are more conservative "
            "(require more similarity to flag as duplicate -> more kept). "
            "Lower values are more aggressive (flag more pairs -> less kept). "
            "Typical values: 0.8 catches near-identical copies; 0.7 also "
            "catches syndicated articles with boilerplate differences; "
            "0.5 catches 50%%+ content overlap. Default: 0.7."
        ),
    )
    p.add_argument(
        "--minhash_num_perm",
        type=int,
        default=128,
        metavar="N",
        help=(
            "Number of MinHash permutations. More permutations = more accurate "
            "Jaccard estimation but slower hashing and more memory. 128 gives "
            "~2%% estimation error, which is well within dedup tolerance. "
            "Default: 128."
        ),
    )
    p.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        metavar="TOKEN",
        help=(
            "HuggingFace access token (required for gated dataset). "
            "Defaults to HF_TOKEN env var."
        ),
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    slices = [parse_slice(s) for s in args.slices]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.hf_token:
        log.warning(
            "No HF token provided.  If the dataset is gated this will fail. "
            "Pass --hf_token or set HF_TOKEN."
        )

    state = DeduplicationState(
        minhash_threshold=args.minhash_threshold,
        minhash_num_perm=args.minhash_num_perm,
    )

    log.info(
        f"Processing {len(slices)} slice(s)  "
        f"minhash_threshold={args.minhash_threshold}  "
        f"minhash_num_perm={args.minhash_num_perm}  "
        f"output_dir={output_dir}"
    )

    all_stats = []
    for lang, snapshot in slices:
        all_stats.append(process_slice(lang, snapshot, state, output_dir, args.hf_token))

    log.info(f"All slices done.  Global dedup stats: {state.report()}")
    print_summary(all_stats)


if __name__ == "__main__":
    main()
