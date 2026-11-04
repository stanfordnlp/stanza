"""
Convert a lemmatizer .pt file to the current dict format, without loading
the seq2seq model or any other heavyweight components.

Three dict formats have existed:
  v1 (<=1.13): legacy (word_dict, composite_dict) tuple, plain pickle
  v2 (1.14):   {pos: {word: lemma}}, gzip+pickle — security hole, no longer loadable
  v3 (1.15+):  {pos: {word: lemma}}, gzip+orjson — current format

This script converts v1 and v2 models to v3.  If you have a v2 model and
try to load it without converting, Stanza will raise an error directing
you here.

Usage:
    python3 convert_lemma_dict.py path/to/model.pt [--output path/to/output.pt]

If --output is not given, the input file is overwritten in place (after
writing to a .tmp sibling first, so the original is safe until the rename).

The checkpoint is loaded with torch.load(weights_only=True), the same
restriction Stanza uses when loading models.  The v2 dict is unpickled
with an unpickler that refuses to resolve any class or function, so a
crafted model cannot run code during conversion.  A legitimate v2 dict
is nothing but nested dicts of strings, which need no class lookups.
The decompressed v2 dict is also capped in size, so a gzip bomb fails
instead of exhausting memory.
"""

import argparse
import gzip
import io
import logging
import os
import pickle
from collections import defaultdict

import torch

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

import orjson

_POS_INDEPENDENT  = "*"
_DICTS_VERSION_LEGACY     = 1
_DICTS_VERSION_POS_PICKLE = 2   # gzip+pickle, replaced by v3
_DICTS_VERSION_POS        = 3   # gzip+json

# Upper bound on the decompressed size of a v2 dict.  Real lemma dicts
# are a few tens of MB at most.
_MAX_DICT_BYTES = 1 << 30


class _NoGlobalsUnpickler(pickle.Unpickler):
    """
    Unpickler which refuses every class and function lookup.

    All of pickle's routes to calling arbitrary code (GLOBAL,
    STACK_GLOBAL, INST, OBJ, EXT*) go through find_class, and
    persistent ids are rejected by the base class when persistent_load
    is not defined.  Plain dicts, lists, tuples, strings, and numbers
    are built directly by opcodes and still load.
    """
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name}: a Stanza lemma dict contains only strings")


def _bounded_gunzip(data, limit=_MAX_DICT_BYTES):
    """Decompress gzip bytes, raising ValueError if the result exceeds limit bytes."""
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
        raw = gz.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"decompressed dict exceeds {limit} bytes; refusing to continue")
    return raw


def _validate_pos_dict(pos_dict):
    """Check that pos_dict is {str: {str: str}}, raising ValueError otherwise."""
    if not isinstance(pos_dict, dict):
        raise ValueError(f"expected a dict of dicts, got {type(pos_dict).__name__}")
    for pos, words in pos_dict.items():
        if not isinstance(pos, str) or not isinstance(words, dict):
            raise ValueError(f"unexpected POS entry {pos!r}: {type(words).__name__}")
        for w, l in words.items():
            if not isinstance(w, str) or not isinstance(l, str):
                raise ValueError(f"unexpected entry under POS {pos!r}: {w!r} -> {l!r}")


def _load_v2_pos_dict(data):
    """
    Decode a v2 (gzip+pickle) dict without executing anything it contains.

    Raises pickle.UnpicklingError if the pickle references any class or
    function, and ValueError if it decompresses too large or does not
    have the {pos: {word: lemma}} shape.
    """
    raw = _bounded_gunzip(data)
    pos_dict = _NoGlobalsUnpickler(io.BytesIO(raw)).load()
    _validate_pos_dict(pos_dict)
    return pos_dict


def _legacy_dicts_to_pos_dict(word_dict, composite_dict):
    """
    Convert (word_dict, composite_dict) to {pos: {word: lemma}}.

    Composite entries that agree with word_dict are dropped — they are
    recoverable via the _POS_INDEPENDENT fallback at lookup time.
    """
    pos_dict = defaultdict(dict)
    for w, l in word_dict.items():
        pos_dict[_POS_INDEPENDENT][w] = l
    n_dropped = 0
    for (w, pos), l in composite_dict.items():
        if word_dict.get(w) != l:
            pos_dict[pos][w] = l
        else:
            n_dropped += 1
    logger.info("  composite entries dropped (redundant with word_dict): %d / %d  (%.1f%%)",
                n_dropped, len(composite_dict),
                100.0 * n_dropped / max(len(composite_dict), 1))
    return dict(pos_dict)


def _pack_pos_dict(pos_dict):
    """Serialize pos_dict to gzip-compressed JSON bytes."""
    raw = orjson.dumps(pos_dict)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
        gz.write(raw)
    return buf.getvalue()


def convert(input_path, output_path):
    logger.info("Loading checkpoint: %s", input_path)
    try:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"{input_path} contains objects outside the types a Stanza lemmatizer "
            "checkpoint uses, so it was not loaded.  Do not use this file.") from e

    dicts_version = checkpoint.get('dicts_version', _DICTS_VERSION_LEGACY)
    if dicts_version == _DICTS_VERSION_POS:
        changed = False
        if 'charlm_forward_file' not in checkpoint['config']:
            checkpoint['config']['charlm_forward_file'] = None
            changed = True
        if 'charlm_backward_file' not in checkpoint['config']:
            checkpoint['config']['charlm_backward_file'] = None
            changed = True
        if changed:
            tmp_path = output_path + ".tmp"
            logger.info("Writing to %s ...", tmp_path)
            torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, output_path)
            logger.info("Updated args and saved to %s", output_path)
        else:
            logger.info("Already in JSON format (dicts_version=%d), nothing to do.", dicts_version)
        return
    if dicts_version == _DICTS_VERSION_POS_PICKLE:
        # pickle-encoded v2 — re-encode as JSON v3
        logger.info("Found pickle-encoded v2 format — re-encoding as JSON (v3)...")
        try:
            pos_dict = _load_v2_pos_dict(checkpoint['dicts'])
        except (pickle.UnpicklingError, ValueError) as e:
            raise RuntimeError(
                f"The v2 dict in {input_path} is not a valid Stanza lemma dict ({e}).  "
                "It may have been tampered with; do not use this file.") from e
        checkpoint['dicts'] = _pack_pos_dict(pos_dict)
        checkpoint['dicts_version'] = _DICTS_VERSION_POS
        tmp_path = output_path + ".tmp"
        logger.info("Writing to %s ...", tmp_path)
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, output_path)
        logger.info("Saved to %s", output_path)
        return

    logger.info("Converting from legacy format (dicts_version=%d)...", dicts_version)
    word_dict, composite_dict = checkpoint['dicts']
    logger.info("  word_dict entries     : %d", len(word_dict))
    logger.info("  composite_dict entries: %d", len(composite_dict))

    pos_dict = _legacy_dicts_to_pos_dict(word_dict, composite_dict)
    logger.info("  pos_dict POS buckets  : %d  (including '%s')",
                len(pos_dict), _POS_INDEPENDENT)

    packed = _pack_pos_dict(pos_dict)
    logger.info("  packed dict size      : %.1f MB", len(packed) / 1024 / 1024)

    checkpoint['dicts'] = packed
    checkpoint['dicts_version'] = _DICTS_VERSION_POS

    # Write to a temp file first so the original is not clobbered on error
    tmp_path = output_path + ".tmp"
    logger.info("Writing to %s ...", tmp_path)
    torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, output_path)
    logger.info("Saved to %s", output_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input .pt lemmatizer checkpoint")
    ap.add_argument("--output", default=None,
                    help="Output path (default: overwrite input in place)")
    args = ap.parse_args()

    output = args.output if args.output else args.input
    convert(args.input, output)


if __name__ == "__main__":
    main()
