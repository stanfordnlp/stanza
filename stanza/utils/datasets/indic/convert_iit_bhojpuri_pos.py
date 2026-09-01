"""
Convert the IIT Bhojpuri POS-tagged corpus to the flat word/tag format
used by Stanza's POS training pipeline.

The source file lives in the Bhojpuri-Magahi-and-Maithili-Linguistic-Resources
repository (github.com/singhakr/Bhojpuri-Magahi-and-Maithili-Linguistic-Resources)
under bhojpuri/pos-tagged/bhojpuri-pos-tagged-ver-1.3.txt.  It is UTF-8 with
a file-level BOM and CRLF line endings.

Two annotation formats appear inside the file:

  flat      index \\t word \\t TAG \\t
  bracketed the word column is "((" and the tag column holds a chunk label
            (NP, VGF, VGNF, CCP, JJP, RBP, NEGP, BLK).  Tokens inside use
            sub-indices (1.1, 1.2, ...).  A "))" line with no tag closes the
            bracket.  702 sentences use this format.

This script:
  - Drops the "((" and "))" lines, keeping the tokens inside in file order
  - Drops any sentence containing a token with a missing or empty tag
  - Maps NNP:? -> NNP (3 tokens, all person names)
  - Drops tokens tagged NEG:? or "-" (2 tokens total)
  - Writes one token per line as word \\t tag, with a blank line between
    sentences, LF endings, and no BOM

Produces 14,780 sentences / 223,952 tokens.  479 sentences are dropped
because they contain tokens with missing or empty tags.

Source file: https://github.com/singhakr/Bhojpuri-Magahi-and-Maithili-Linguistic-Resources
"""

import argparse
import os
import re

from stanza.utils.default_paths import get_default_paths

SENTENCE_RE = re.compile(r"<Sentence [^>]*>")


def parse_sentences(path):
    """Read the XML-ish file and yield one sentence at a time.

    Each sentence is a list of non-empty lines (stripped of their trailing
    newline).  Lines that are pure markup (<document>, <head>, etc.) are
    silently skipped.
    """
    current = []
    inside = False

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if SENTENCE_RE.search(line):
                inside = True
                current = []
                continue
            if line.startswith("</Sentence>"):
                if inside and current:
                    yield current
                inside = False
                current = []
                continue
            if inside and line.strip():
                current.append(line.strip())


def convert(path, out_path):
    """Parse the tagged file and write the flat word/tag output."""
    sentences_kept = 0
    tokens_total = 0
    dropped_sentences = 0
    dropped_empty_tags = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for lines in parse_sentences(path):
            tokens = []
            has_empty_tag = False

            for line in lines:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                word = parts[1]
                tag = parts[2] if len(parts) > 2 else ""

                # Skip bracket delimiters
                if word in ("((", "))"):
                    continue

                # A missing or empty tag disqualifies the whole sentence
                if not tag:
                    has_empty_tag = True
                    dropped_empty_tags += 1
                    continue

                # Drop NEG:? and "-" tokens entirely
                if tag in ("NEG:?", "-"):
                    continue

                # Normalise the one known variant
                if tag == "NNP:?":
                    tag = "NNP"

                tokens.append((word, tag))

            if has_empty_tag:
                dropped_sentences += 1
                continue

            for word, tag in tokens:
                out.write("%s\t%s\n" % (word, tag))
            out.write("\n")

            sentences_kept += 1
            tokens_total += len(tokens)

    print("Kept %d sentences, %d tokens" % (sentences_kept, tokens_total))
    print("Dropped %d sentences (%d lines with empty tags)" % (
        dropped_sentences, dropped_empty_tags))


def main():
    paths = get_default_paths()
    output_dir = paths["POS_DATA_DIR"]
    default_output_filename = "bhojpuri_iit_pos.txt"
    default_input_path = os.path.join(paths["STANZA_EXTERN_DIR"], "bhojpuri", "Bhojpuri-Magahi-and-Maithili-Linguistic-Resources", "bhojpuri", "pos-tagged", "bhojpuri-pos-tagged-ver-1.3.txt")

    parser = argparse.ArgumentParser(
        description="Convert the IIT Bhojpuri POS corpus to flat word/tag format for Stanza training",
        formatter_class=argparse.RawTextHelpFormatter,  # for the long filenames
    )
    parser.add_argument(
        "--input", default=default_input_path,
        help="Path to bhojpuri-pos-tagged-ver-1.3.txt - defaults to %s" % default_input_path
    )
    parser.add_argument(
        "--output", default=default_output_filename,
        help="Output filename.  Defaults to %s" % default_output_filename
    )
    parser.add_argument(
        "--output_dir", default=output_dir,
        help="Output dir.  Defaults to %s" % output_dir
    )
    args = parser.parse_args()

    args.output = os.path.join(args.output_dir, args.output)

    print("Input:  %s" % args.input)
    print("Output: %s" % args.output)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
