"""
community_oscar_inspect_similarity.py

Sample random document pairs from a Community-OSCAR corpus file or a
processed dedup output file, and report pairwise similarity distributions
using MinHash (Jaccard / word-overlap) and/or TLSH (byte-level sequence
similarity). Having both gives a richer picture: they measure different
properties, and the correlation between them is informative in itself.

Either package can be absent -- the script reports whichever metrics are
available and warns about what's missing.

Usage
-----
# Check a raw Community-OSCAR source file (cached .jsonl.zst)
python inspect_similarity.py /path/to/ps_meta.jsonl.zst

# Check a processed dedup output file (.txt, one doc per line)
python inspect_similarity.py /path/to/ps_2024-38.txt

# Custom sample sizes and threshold markers
python inspect_similarity.py /path/to/ps_meta.jsonl.zst \\
    --n_docs 1000 --n_pairs 3000 \\
    --minhash_thresholds 0.5 0.6 0.7 0.8 0.9 \\
    --tlsh_thresholds 50 75 100 150 200

Requirements (both optional -- script runs with whichever are installed)
------------
    pip install datasketch   # for MinHash / Jaccard
    pip install py-tlsh      # for TLSH / byte-level similarity
    pip install zstandard    # required for .jsonl.zst input
"""

import argparse
import io
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    from datasketch import MinHash
    _HAVE_MINHASH = True
except ImportError:
    _HAVE_MINHASH = False

try:
    import tlsh as _tlsh_lib
    _HAVE_TLSH = True
except ImportError:
    _HAVE_TLSH = False

try:
    import zstandard
    _HAVE_ZST = True
except ImportError:
    _HAVE_ZST = False


# ---------------------------------------------------------------------------
# Text iteration
# ---------------------------------------------------------------------------

def iter_texts_zst(path: str, max_docs: int):
    """Stream text content from a Community-OSCAR .jsonl.zst file."""
    if not _HAVE_ZST:
        print("ERROR: zstandard not installed (pip install zstandard) -- "
              "cannot read .jsonl.zst files.", file=sys.stderr)
        sys.exit(1)
    dctx = zstandard.ZstdDecompressor()
    count = 0
    with open(path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get('content', '')
                if text and text.strip():
                    yield text.strip()
                    count += 1
                    if max_docs and count >= max_docs:
                        break


def iter_texts_txt(path: str, max_docs: int):
    """Stream lines from a processed dedup output .txt file."""
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            line = line.strip()
            if line:
                yield line


def collect_texts(path: str, n_docs: int) -> list[str]:
    """Read and return a sample of document texts from either file type."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    is_zst = path.endswith('.jsonl.zst') or path.endswith('.zst')
    is_txt = path.endswith('.txt')
    if not is_zst and not is_txt:
        print(f"ERROR: unrecognised file type (expected .jsonl.zst or .txt): {path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Reading {'raw OSCAR' if is_zst else 'dedup output'} file: {p.name}")
    print(f"Sampling up to {n_docs} documents...")

    texts = []
    short = 0
    iter_fn = iter_texts_zst if is_zst else iter_texts_txt
    for text in iter_fn(path, max_docs=n_docs * 2):
        if len(text.split()) < 5:
            short += 1
            continue
        texts.append(text)
        if len(texts) >= n_docs:
            break

    if short:
        print(f"  Skipped {short} documents with fewer than 5 words.")
    print(f"  Collected {len(texts)} documents.")
    return texts


# ---------------------------------------------------------------------------
# MinHash / Jaccard
# ---------------------------------------------------------------------------

def make_minhash(text: str, num_perm: int) -> 'MinHash':
    """Word-unigram MinHash, matching community_oscar_dedup.py's implementation."""
    m = MinHash(num_perm=num_perm)
    seen = set()
    for w in text.split():
        h = hash(w)
        if h not in seen:
            seen.add(h)
            m.update(h.to_bytes(8, 'little', signed=True))
    return m


def compute_jaccard_pairs(texts: list, pairs: list, num_perm: int) -> list[float]:
    """Compute MinHash Jaccard similarity for the given index pairs."""
    print(f"  Building {len(texts)} MinHash objects...")
    minhashes = [make_minhash(t, num_perm) for t in texts]
    print(f"  Computing Jaccard for {len(pairs)} pairs...")
    return sorted(minhashes[i].jaccard(minhashes[j]) for i, j in pairs)


# ---------------------------------------------------------------------------
# TLSH
# ---------------------------------------------------------------------------

def make_tlsh(text: str):
    """Compute a TLSH hash from document content. Returns None on failure."""
    h = _tlsh_lib.hash(text.encode('utf-8', errors='replace'))
    if not h or h == 'TNULL':
        return None
    t = _tlsh_lib.Tlsh()
    try:
        t.fromTlshStr(h)
    except Exception:
        return None
    return t


def compute_tlsh_pairs(texts: list, pairs: list) -> list[int]:
    """Compute TLSH diff scores for the given index pairs."""
    print(f"  Building {len(texts)} TLSH hashes...")
    hashes = [make_tlsh(t) for t in texts]
    valid = sum(1 for h in hashes if h is not None)
    if valid < len(hashes):
        print(f"  ({len(hashes) - valid} documents too short/simple for TLSH, skipped)")

    diffs = []
    print(f"  Computing TLSH diff for {len(pairs)} pairs...")
    for i, j in pairs:
        if hashes[i] is not None and hashes[j] is not None:
            diffs.append(hashes[i].diff(hashes[j]))
    return sorted(diffs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def percentile(values: list, pct: float):
    idx = min(int(len(values) * pct / 100), len(values) - 1)
    return values[idx]


def print_distribution(values: list, label: str):
    print(f"\n{label} distribution ({len(values)} pairs):")
    for name, pct in [("min", 0), ("p1", 1), ("p5", 5), ("p10", 10),
                       ("p25", 25), ("median", 50), ("p75", 75),
                       ("p90", 90), ("p99", 99), ("max", 100)]:
        print(f"  {name:>8}: {percentile(values, pct):.4f}"
              if isinstance(values[0], float)
              else f"  {name:>8}: {percentile(values, pct)}")


def print_threshold_table(values: list, thresholds: list, above: bool, label: str):
    """
    Report what fraction of random pairs would be flagged at each threshold.
    above=True for Jaccard (flag if >= threshold),
    above=False for TLSH diff (flag if < threshold).
    """
    direction = "ABOVE" if above else "BELOW"
    print(f"\nFraction of random pairs {direction} threshold ({label}):")
    for t in sorted(thresholds):
        frac = (sum(1 for v in values if v >= t) / len(values) if above
                else sum(1 for v in values if v < t) / len(values))
        if frac > 0.05:
            warn = "  <- WARNING: >5% of unrelated pairs would be flagged"
        elif frac > 0.01:
            warn = "  <- NOTE: >1% of unrelated pairs would be flagged"
        else:
            warn = ""
        print(f"  threshold={t}: {frac*100:.2f}% flagged{warn}")


def print_interpretation(jaccards, tlsh_diffs):
    print("\nInterpretation:")

    if jaccards:
        med_j = percentile(jaccards, 50)
        p90_j = percentile(jaccards, 90)
        if med_j < 0.1 and p90_j < 0.3:
            print("  MinHash: corpus looks well-diversified. Random pairs have low")
            print("  vocabulary overlap. Check the table above to confirm your")
            print("  chosen threshold has a low false-positive rate.")
        elif med_j < 0.2 and p90_j < 0.5:
            print("  MinHash: moderate vocabulary overlap between unrelated docs.")
            print("  If your chosen threshold is below 0.75, check the table above --")
            print("  you may be getting more false positives than expected.")
        else:
            print("  MinHash: WARNING — high Jaccard between random pairs.")
            print("  Many unrelated documents share vocabulary, likely due to")
            print("  heavy boilerplate or a narrow set of sources. Even a high")
            print("  threshold (0.8-0.9) may produce false positives for this corpus.")

    if tlsh_diffs:
        med_t = percentile(tlsh_diffs, 50)
        p10_t = percentile(tlsh_diffs, 10)
        if med_t > 150 and p10_t > 50:
            print("  TLSH: corpus looks well-diversified. Random pairs have high")
            print("  byte-level difference scores. Check the table above to confirm")
            print("  your chosen threshold has a low false-positive rate.")
        elif med_t > 100:
            print("  TLSH: moderate byte-level similarity between unrelated docs.")
            print("  If your chosen threshold is above 50, check the false-positive")
            print("  rate in the table above.")
        else:
            print("  TLSH: WARNING — low diff scores between random pairs.")
            print("  High byte-level similarity between unrelated docs. Even a low")
            print("  threshold (< 30) may produce false positives for this corpus.")

    if jaccards and tlsh_diffs:
        j_at_default = sum(1 for v in jaccards if v >= 0.7) / len(jaccards)
        t_at_default = sum(1 for v in tlsh_diffs if v < 50) / len(tlsh_diffs)
        # Only flag disagreement if at least one method is detecting something
        # non-trivial and they differ substantially -- 0.0% vs 0.0% is agreement,
        # not disagreement, and ratio comparison breaks down near zero.
        if max(j_at_default, t_at_default) > 0.005:
            ratio = j_at_default / max(t_at_default, 0.0001)
            if ratio > 5 or ratio < 0.2:
                print(f"\n  NOTE: MinHash and TLSH give quite different false-positive")
                print(f"  rates at their respective default thresholds (MinHash@0.7:")
                print(f"  {j_at_default*100:.1f}%, TLSH@50: {t_at_default*100:.1f}%).")
                print(f"  This is expected -- they measure different properties.")
                print(f"  Worth sampling a few pairs near each threshold to understand")
                print(f"  which better matches your dedup goals for this language.")
    print()


def print_report(path: str, jaccards, tlsh_diffs,
                 minhash_thresholds, tlsh_thresholds):
    print()
    print("=" * 60)
    print(f"SIMILARITY REPORT: {Path(path).name}")
    print("=" * 60)

    if not jaccards and not tlsh_diffs:
        print("No metrics available (install datasketch and/or py-tlsh).")
        return

    if jaccards:
        print_distribution(jaccards, "MinHash Jaccard similarity (word-unigram)")
        print_threshold_table(jaccards, minhash_thresholds, above=True, label="MinHash Jaccard")

    if tlsh_diffs:
        print_distribution(tlsh_diffs, "TLSH diff score (byte-level, lower = more similar)")
        print_threshold_table(tlsh_diffs, tlsh_thresholds, above=False, label="TLSH diff")

    print_interpretation(jaccards, tlsh_diffs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "path",
        help="Path to a .jsonl.zst Community-OSCAR file or a processed .txt output file.",
    )
    p.add_argument(
        "--n_docs", type=int, default=2000, metavar="N",
        help="Documents to sample. Default: 2000.",
    )
    p.add_argument(
        "--n_pairs", type=int, default=5000, metavar="N",
        help="Random pairs to evaluate. Default: 5000.",
    )
    p.add_argument(
        "--minhash_thresholds", nargs="+", type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9], metavar="FLOAT",
        help="Jaccard thresholds to report false-positive rates for. Default: 0.5 0.6 0.7 0.8 0.9.",
    )
    p.add_argument(
        "--tlsh_thresholds", nargs="+", type=int,
        default=[30, 50, 75, 100, 150], metavar="N",
        help="TLSH diff thresholds to report false-positive rates for. Default: 30 50 75 100 150.",
    )
    p.add_argument(
        "--num_perm", type=int, default=128, metavar="N",
        help="MinHash permutations. Should match --minhash_num_perm used in dedup. Default: 128.",
    )
    p.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed. Default: 42.",
    )
    p.add_argument(
        "--no_minhash", action="store_true",
        help="Skip MinHash/Jaccard even if datasketch is installed.",
    )
    p.add_argument(
        "--no_tlsh", action="store_true",
        help="Skip TLSH even if py-tlsh is installed.",
    )
    return p


def main():
    args = build_parser().parse_args()

    use_minhash = _HAVE_MINHASH and not args.no_minhash
    use_tlsh = _HAVE_TLSH and not args.no_tlsh

    if not use_minhash and not args.no_minhash:
        print("NOTE: datasketch not installed -- skipping MinHash/Jaccard metrics.",
              file=sys.stderr)
        print("      pip install datasketch", file=sys.stderr)
    if not use_tlsh and not args.no_tlsh:
        print("NOTE: py-tlsh not installed -- skipping TLSH metrics.", file=sys.stderr)
        print("      pip install py-tlsh", file=sys.stderr)
    if not use_minhash and not use_tlsh:
        print("ERROR: no similarity metrics available. Install at least one of:",
              file=sys.stderr)
        print("  pip install datasketch   (MinHash/Jaccard)", file=sys.stderr)
        print("  pip install py-tlsh      (TLSH/byte-level)", file=sys.stderr)
        sys.exit(1)

    texts = collect_texts(args.path, args.n_docs)
    if len(texts) < 2:
        print("ERROR: need at least 2 documents.", file=sys.stderr)
        sys.exit(1)

    # Sample pairs once, use for both metrics
    random.seed(args.seed)
    n = len(texts)
    pair_set = set()
    attempts = 0
    while len(pair_set) < args.n_pairs and attempts < args.n_pairs * 10:
        i, j = random.randrange(n), random.randrange(n)
        if i != j:
            pair_set.add((min(i, j), max(i, j)))
        attempts += 1
    pairs = list(pair_set)
    print(f"\nUsing {len(pairs)} random pairs for both metrics.")

    jaccards = compute_jaccard_pairs(texts, pairs, args.num_perm) if use_minhash else None
    tlsh_diffs = compute_tlsh_pairs(texts, pairs) if use_tlsh else None

    print_report(args.path, jaccards, tlsh_diffs,
                 args.minhash_thresholds, args.tlsh_thresholds)


if __name__ == "__main__":
    main()
