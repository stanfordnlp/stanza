"""
Create Stanza character LM train/dev/test data, by reading from txt files in each source corpus directory,
shuffling, splitting and saving into multiple smaller files (50MB by default) in a target directory.

This script assumes the following source directory structures:
    - {src_dir}/{language}/{corpus}/*.txt
It will read from all source .txt files and create the following target directory structures:
    - {tgt_dir}/{language}/{corpus}
and within each target directory, it will create the following files:
    - train/*.txt
    - dev.txt
    - test.txt
Args:
    - src_root: root directory of the source.
    - tgt_root: root directory of the target.
    - langs: a list of language codes to process; if specified, languages not in this list will be ignored.
Note: edit the {EXCLUDED_FOLDERS} variable to exclude more folders in the source directory.

Implementation note (shuffle/split):
    Earlier versions of this script shelled out to `cat`, `xzcat`, `zcat`, `shuf`, and `split` via
    subprocess(shell=True).  This relied on filenames never containing shell metacharacters, and did
    not work on Windows (no shuf/split/xzcat there).  This version avoids the shell entirely and
    instead does a two pass bucket shuffle:

      Pass 1: every source file (decompressed on the fly if .gz / .xz) is streamed line by line, and
              each line is assigned uniformly at random to one of N bucket files, where N is the same
              as the eventual number of train shards.  This scatters every source file's content
              proportionally across every bucket, which matters when there are few, large source files
              (e.g. one big Wikipedia dump and one big Common Crawl dump) -- without this step, naive
              chunked reading would put long runs of a single source into a single shard, badly skewing
              dev/test (which are just the first couple of shards).  Pass 1 only ever needs N file
              handles open for writing plus one for reading, so it stays well under OS fd limits
              regardless of how many source files there are (we've seen 400+ for some languages).
      Pass 2: each bucket (already approximately shard-sized by construction) is read fully into memory,
              shuffled locally with random.shuffle, and written out as the corresponding final shard.
              Because every bucket already contains a proportional mix of all source files, every shard
              -- including dev/test -- ends up with the same source mixture as the corpus as a whole.

    This trades one extra streaming read+write pass for: no shell dependency, Windows compatibility, and
    (with the bucket-then-local-shuffle approach) better dev/test distributional properties than a purely
    local chunk shuffle would give when source files are few and large.

    One narrow subprocess use remains, deliberately: estimating a target bucket/shard count for .xz
    *source* files shells out to `xz --robot -l path` (list form, no shell=True, so still safe
    regardless of filename contents) to read the file's size index in O(1) without a full
    decompression pass.  This is only used to size pass 1 -- not load-bearing for correctness -- and
    falls back to on-disk size (an underestimate for compressed files, which just makes shards a bit
    larger than split_size) if the xz binary isn't available.  Final .xz compression of output files
    uses the stdlib lzma module instead (pure Python, no subprocess, no xz binary dependency) --
    single threaded, so slower than `xz -T0` on multi-core machines for very large files, but that
    tradeoff was preferred over a multiprocessing-based parallel compressor for now.
"""

import argparse
import gzip
import lzma
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

from tqdm import tqdm

EXCLUDED_FOLDERS = ['raw_corpus']

# Read/write files in fixed-size text chunks to keep streaming I/O fast without
# loading whole files into memory.
IO_CHUNK_LINES = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", default="src", help="Root directory with all source files.  Expected structure is root dir -> language dirs -> package dirs -> text files to process")
    parser.add_argument("tgt_root", default="tgt", help="Root directory with all target files.")
    parser.add_argument("--langs", default="", help="A list of language codes to process.  If not set, all languages under src_root will be processed.")
    parser.add_argument("--packages", default="", help="A list of packages to process.  If not set, all packages under the languages found will be processed.")
    parser.add_argument("--no_xz_output", default=True, dest="xz_output", action="store_false", help="Output compressed xz files")
    parser.add_argument("--split_size", default=50, type=int, help="How large to make each split, in MB")
    parser.add_argument("--no_make_test_file", default=True, dest="make_test_file", action="store_false", help="Don't save a test file.  Honestly, we never even use it.  Best for low resource languages where every bit helps")
    parser.add_argument("--max_open_handles", default=100, type=int, help="Cap on simultaneously open bucket files in pass 1.  Mainly relevant if split_size is set very small, producing a huge number of buckets; keep this comfortably under the OS file descriptor limit (ulimit -n).")
    parser.add_argument("--bucket_compression", default=False, action="store_true", help="Compress intermediate bucket files (pass 1 output) with xz.  Saves disk space at the cost of extra CPU; off by default since buckets are temporary and disk is usually cheaper than CPU time.")
    args = parser.parse_args()

    print("Processing files:")
    print(f"source root: {args.src_root}")
    print(f"target root: {args.tgt_root}")
    print("")

    langs = []
    if len(args.langs) > 0:
        langs = args.langs.split(',')
        print("Only processing the following languages: " + str(langs))

    packages = []
    if len(args.packages) > 0:
        packages = args.packages.split(',')
        print("Only processing the following packages: " + str(packages))

    src_root = Path(args.src_root)
    tgt_root = Path(args.tgt_root)

    lang_dirs = os.listdir(src_root)
    lang_dirs = [l for l in lang_dirs if l not in EXCLUDED_FOLDERS]    # skip excluded
    lang_dirs = [l for l in lang_dirs if os.path.isdir(src_root / l)]  # skip non-directory
    if len(langs) > 0: # filter languages if specified
        lang_dirs = [l for l in lang_dirs if l in langs]
    print(f"{len(lang_dirs)} total languages found:")
    print(lang_dirs)
    print("")

    split_size = int(args.split_size * 1024 * 1024)

    for lang in lang_dirs:
        lang_root = src_root / lang
        data_dirs = os.listdir(lang_root)
        if len(packages) > 0:
            data_dirs = [d for d in data_dirs if d in packages]
        data_dirs = [d for d in data_dirs if os.path.isdir(lang_root / d)]
        print(f"{len(data_dirs)} total corpus found for language {lang}.")
        print(data_dirs)
        print("")

        for dataset_name in data_dirs:
            src_dir = lang_root / dataset_name
            tgt_dir = tgt_root / lang / dataset_name

            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            print(f"-> Processing {lang}-{dataset_name}")
            prepare_lm_data(src_dir, tgt_dir, lang, dataset_name, args.xz_output, split_size,
                             args.make_test_file, args.max_open_handles, args.bucket_compression)

        print("")


def open_text_read(path):
    """
    Open a .txt / .txt.gz / .txt.xz file for streaming text reading, regardless of compression.
    """
    if path.endswith(".txt"):
        return open(path, "rt", encoding="utf-8", errors="surrogateescape")
    elif path.endswith(".txt.xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="surrogateescape")
    elif path.endswith(".txt.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="surrogateescape")
    else:
        raise AssertionError("should not have found %s" % path)


def open_bucket_read(path, compress):
    if compress:
        return lzma.open(path + ".xz", "rt", encoding="utf-8", errors="surrogateescape")
    else:
        return open(path, "rt", encoding="utf-8", errors="surrogateescape")


def get_input_files(src_dir):
    src_dir = Path(src_dir)
    input_files = (sorted(src_dir.glob("*.txt")) +
                    sorted(src_dir.glob("*.txt.xz")) +
                    sorted(src_dir.glob("*.txt.gz")))
    return [str(f) for f in input_files]


def compress_file_to_xz(path):
    """
    Compress a file to .xz and remove the original, matching the behavior of the `xz` CLI run on
    a single file (in-place replace: path -> path + ".xz", original deleted).  Pure Python via the
    stdlib lzma module -- single-threaded, so slower than `xz -T0` on multi-core machines for large
    files, but removes the xz binary as a hard dependency for producing the final deliverable
    output files.  Binary mode + copyfileobj avoids any text encode/decode roundtrip, since we're
    just recompressing existing bytes, not transforming them.
    """
    xz_path = path + ".xz"
    with open(path, "rb") as fin, lzma.open(xz_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(path)


def gzip_uncompressed_size(path):
    """
    Gzip files store the uncompressed size mod 2**32 in their trailer (last 4 bytes) -- a O(1)
    lookup, no decompression needed.  The mod-2**32 wraparound means this is only exact for files
    under 4GB uncompressed; for anything larger it under-reports, which would just make our bucket
    count estimate low (buckets/shards end up bigger than split_size) -- a soft target miss, not a
    correctness problem, and per-source-file files over 4GB uncompressed are not the common case
    for the per-file granularity this script deals with.
    """
    import struct
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return struct.unpack("<I", f.read(4))[0]


def xz_uncompressed_size(path):
    """
    xz files store an index at the end with exact block sizes; `xz -l` reads just that index
    (O(1), no decompression) and reports the exact uncompressed size.  Falls back to on-disk size
    (an under-estimate) if the xz binary isn't available or parsing fails, since this is only used
    for the soft bucket-count target, not a correctness-critical value.
    """
    try:
        result = subprocess.run(["xz", "--robot", "-l", path], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("file\t"):
                return int(line.split("\t")[4])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass
    return os.path.getsize(path)


def measure_size_for_bucket_count(input_files):
    """
    Estimate the decompressed size of all input files cheaply (no full decompression pass) to
    decide how many buckets/shards to target.  Uses exact O(1) size lookups where the file format
    supports them (gzip trailer, xz index); falls back to on-disk size for plain .txt (exact) or
    if a lookup fails (under-estimate, soft target miss only).  This only sets the *target* bucket
    count -- pass 1 measures and reports the true total byte count as a side effect of scattering,
    which is what the minimum-size sanity check uses.
    """
    total_bytes = 0
    for f in input_files:
        if f.endswith(".txt"):
            total_bytes += os.path.getsize(f)
        elif f.endswith(".txt.gz"):
            total_bytes += gzip_uncompressed_size(f)
        elif f.endswith(".txt.xz"):
            total_bytes += xz_uncompressed_size(f)
        else:
            total_bytes += os.path.getsize(f)
    return total_bytes


def prepare_lm_data(src_dir, tgt_dir, lang, dataset_name, compress, split_size, make_test_file,
                     max_open_handles, bucket_compression):
    """
    Combine, shuffle and split data into smaller files, following a naming convention.

    Two pass bucket shuffle (see module docstring for rationale):
      Pass 1: scatter every line from every source file uniformly at random into one of N bucket
              files (single read pass over the source data), where N is estimated from total
              input size / split_size.  At most max_open_handles bucket files are held open
              concurrently; remaining buckets are flushed via brief open-append-close.
      Pass 2: read each bucket fully, shuffle its lines locally, write out as the corresponding
              final shard.  First one or two shards (post shuffle) become dev.txt / test.txt.
    """
    assert isinstance(src_dir, Path)
    assert isinstance(tgt_dir, Path)

    input_files = get_input_files(src_dir)
    if not input_files:
        print(f"--> No input files found in {src_dir}, skipping.")
        return

    on_disk_bytes = measure_size_for_bucket_count(input_files)
    num_buckets = max(1, round(on_disk_bytes / split_size))
    print(f"--> On-disk size: {on_disk_bytes/1024/1024/1024:.4f} GB, targeting ~{num_buckets} shard(s)")

    train_dir = tgt_dir / 'train'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    with tempfile.TemporaryDirectory(dir=tgt_dir) as tempdir:
        bucket_paths = [os.path.join(tempdir, f"bucket-{i:04d}.txt") for i in range(num_buckets)]

        print(f"--> Pass 1/2: scattering {len(input_files)} input file(s) across {num_buckets} bucket(s)...")
        actual_chars = scatter_into_buckets(input_files, bucket_paths, max_open_handles, bucket_compression)
        # Character count, not exact UTF-8 byte count (see scatter_into_buckets docstring) -- used
        # as an approximate proxy for the size floor below.  For predominantly Latin-script text
        # this undercounts true bytes only slightly if at all (most chars are 1-2 UTF-8 bytes), so
        # the floor check stays conservative rather than silently permissive.
        approx_gb = actual_chars / 1024 / 1024 / 1024
        print(f"--> Actual size (approx, char count): {approx_gb:.4f} GB")
        if approx_gb < 0.1:
            raise RuntimeError("Not enough data found to build a charlm.  At least 100MB data expected")

        print("--> Pass 2/2: shuffling each bucket and writing final shards...")
        shard_paths = []
        random.shuffle(bucket_paths)  # randomize which bucket becomes shard 0000, 0001, etc.
        shard_index = 0
        for bucket_path in tqdm(bucket_paths):
            lines = read_bucket_lines(bucket_path, bucket_compression)
            if not lines:
                continue
            random.shuffle(lines)
            shard_path = os.path.join(train_dir, f"{lang}-{dataset_name}-{shard_index:04d}.txt")
            with open(shard_path, "wt", encoding="utf-8", errors="surrogateescape") as fout:
                fout.writelines(lines)
            shard_paths.append(shard_path)
            shard_index += 1

        total = len(shard_paths)
        print(f"--> {total} total files generated.")
        if total < 3:
            raise RuntimeError("Something went wrong!  %d file(s) produced by shuffle and split, expected at least 3" % total)

        dev_file = f"{tgt_dir}/dev.txt"
        test_file = f"{tgt_dir}/test.txt"
        if make_test_file:
            print("--> Creating dev and test files...")
            shutil.move(shard_paths[0], dev_file)
            shutil.move(shard_paths[1], test_file)
            txt_files = [dev_file, test_file] + shard_paths[2:]
        else:
            print("--> Creating dev file...")
            shutil.move(shard_paths[0], dev_file)
            txt_files = [dev_file] + shard_paths[1:]

        if compress:
            print("--> Compressing files...")
            for txt_file in tqdm(txt_files):
                compress_file_to_xz(txt_file)

        print("--> Cleaning up...")
    print(f"--> All done for {lang}-{dataset_name}.\n")


def scatter_into_buckets(input_files, bucket_paths, max_open_handles, bucket_compression):
    """
    Pass 1: stream every input file exactly once and randomly assign each line to one bucket file.

    Bucket files are always written as plain uncompressed text during this pass (append mode is
    simple and well-supported for plain files; incrementally appending to an .xz stream is not a
    well-defined operation, since xz framing isn't designed for that).  If bucket_compression is
    requested, buckets are compressed in a separate pass *after* all scattering is done, when each
    bucket is finished and will only ever be read once in pass 2 -- at that point compressing it is
    just a single whole-file xz pass per bucket, no different in spirit from the final shard
    compression already done elsewhere in this script.

    To bound simultaneously open file descriptors at max_open_handles, we keep in-memory line
    buffers for every bucket, but only actually hold open OS file handles for up to
    max_open_handles buckets at a time ("hot" buckets).  When a buffer for a "cold" (not currently
    open) bucket needs to flush, we open it briefly in append mode, write, and close -- this keeps
    total *concurrently open* handles bounded by max_open_handles + 1 (the source file being read)
    while still only reading every source file once.

    Returns the total character count actually scattered (cheap len() per line, not an exact UTF-8
    byte count -- see note below), measured as a side effect of this pass to avoid a separate,
    redundant decompression pass just to learn the true input size.
    """
    num_buckets = len(bucket_paths)
    buffers = [[] for _ in range(num_buckets)]
    total_chars = 0

    # The first max_open_handles buckets stay open for the whole pass; the rest are flushed via
    # brief open-append-close, which is cheap relative to the cost of re-reading source data.
    num_hot = min(max_open_handles, num_buckets)
    hot_handles = [open(bucket_paths[i], "wt", encoding="utf-8", errors="surrogateescape")
                   for i in range(num_hot)]

    def flush(bucket_idx):
        if not buffers[bucket_idx]:
            return
        if bucket_idx < num_hot:
            hot_handles[bucket_idx].writelines(buffers[bucket_idx])
        else:
            with open(bucket_paths[bucket_idx], "at", encoding="utf-8", errors="surrogateescape") as fout:
                fout.writelines(buffers[bucket_idx])
        buffers[bucket_idx] = []

    # Cache random.random as a local to avoid repeated attribute lookup in the hot loop below, and
    # use it instead of random.randrange: randrange does extra bias-correction work (via
    # _randbelow_with_getrandbits) intended for cryptographically-uniform integer ranges, which we
    # don't need here -- int(random() * num_buckets) is uniform enough for bucket assignment and
    # measured roughly 2x faster per call at the scale this pass runs at (tens of millions of lines).
    _random = random.random

    try:
        for src_fn in tqdm(input_files, desc="scattering source files"):
            with open_text_read(src_fn) as fin:
                for line in fin:
                    # len(line) (character count) instead of len(line.encode(...)) (exact UTF-8
                    # byte count): this total only feeds the coarse >=100MB sanity-check floor in
                    # prepare_lm_data, not anything requiring byte precision, and per-line encode()
                    # calls were a measurable hot-loop cost at this line-count scale.  For
                    # predominantly Latin-script text (1-2 bytes/char in UTF-8) this tracks true
                    # byte size closely; even in the worst case (4 bytes/char) it only makes the
                    # floor check more conservative, never silently permissive.
                    total_chars += len(line)
                    bucket_idx = int(_random() * num_buckets)
                    buffers[bucket_idx].append(line)
                    if len(buffers[bucket_idx]) >= IO_CHUNK_LINES:
                        flush(bucket_idx)
        for bucket_idx in range(num_buckets):
            flush(bucket_idx)
    finally:
        for fh in hot_handles:
            fh.close()

    if bucket_compression:
        print("--> Compressing buckets...")
        for path in tqdm(bucket_paths):
            with open(path, "rt", encoding="utf-8", errors="surrogateescape") as fin, \
                 lzma.open(path + ".xz", "wt", encoding="utf-8", errors="surrogateescape") as fout:
                shutil.copyfileobj(fin, fout)
            os.remove(path)

    return total_chars


def read_bucket_lines(bucket_path, bucket_compression):
    with open_bucket_read(bucket_path, bucket_compression) as fin:
        return fin.readlines()


if __name__ == "__main__":
    main()
