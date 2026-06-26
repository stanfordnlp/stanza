"""
Convert a lemmatizer .pt file from the legacy dict format to the new
pos_direct gzip-compressed format, without loading the seq2seq model
or any other heavyweight components.

Usage:
    python3 convert_lemma_dict.py path/to/model.pt [--output path/to/output.pt]

If --output is not given, the input file is overwritten in place (after
writing to a .tmp sibling first, so the original is safe until the rename).

This is useful for converting lemmatizers from version 1.13.0 to 1.14.0.
Otherwise, this script is not actually needed - the existing lemmatizer code
is already written to load in old formats without failing and to write out
the new format when saving.
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

_POS_INDEPENDENT  = "*"
_DICTS_VERSION_LEGACY = 1
_DICTS_VERSION_POS    = 2


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
    """Serialize pos_dict to gzip-compressed pickle bytes."""
    raw = pickle.dumps(pos_dict, protocol=4)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
        gz.write(raw)
    return buf.getvalue()


def convert(input_path, output_path):
    logger.info("Loading checkpoint: %s", input_path)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    dicts_version = checkpoint.get('dicts_version', _DICTS_VERSION_LEGACY)
    if dicts_version == _DICTS_VERSION_POS:
        logger.info("Already in new format (dicts_version=%d), nothing to do.", dicts_version)
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
