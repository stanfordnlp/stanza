"""
Builds a self-training dataset from a single file.

Default is to assume one document of text per line.  If a line has
multiple sentences, they will be split using the stanza tokenizer.
"""

import argparse
import io
import logging
import os

import numpy as np

import stanza
from stanza.models.common import utils
from stanza.utils.datasets.constituency import selftrain

logger = logging.getLogger('stanza')
tqdm = utils.get_tqdm()

def parse_args():
    """
    Only specific argument for this script is the file to process
    """
    parser = argparse.ArgumentParser(
        description="Script that converts a single file of text to silver standard trees"
    )
    selftrain.common_args(parser)
    parser.add_argument(
        '--input_file',
        default="vi_part_1.aa",
        help='Path to the file to read'
    )

    args = parser.parse_args()
    return args


def read_file(input_file):
    """
    Read lines from an input file

    Takes care to avoid encoding errors at the end of Oscar files.
    The Oscar splits sometimes break a utf-8 character in half.
    """
    with open(input_file, "rb") as fin:
        text = fin.read()
    text = text.decode("utf-8", errors="replace")
    with io.StringIO(text) as fin:
        lines = fin.readlines()
    return lines


def main():
    args = parse_args()

    # TODO: make ssplit an argument
    ssplit_pipe = selftrain.build_ssplit_pipe(ssplit=True, lang=args.lang)
    tag_pipe = selftrain.build_tag_pipe(ssplit=False, lang=args.lang)
    parser_pipes = selftrain.build_parser_pipes(args.lang, args.models)

    # create a blank file.  we will append to this file so that partial results can be used
    with open(args.output_file, "w") as fout:
        pass

    docs = read_file(args.input_file)
    logger.info("Read %d lines from %s", len(docs), args.input_file)
    docs = selftrain.split_docs(docs, ssplit_pipe)

    # breaking into chunks lets us output partial results and see the
    # progress in log files
    accepted_trees = set()
    if len(docs) > 10000:
        chunks = tqdm(np.array_split(docs, 100), disable=False)
    else:
        chunks = [docs]
    for chunk in chunks:
        new_trees = selftrain.find_matching_trees(chunk, args.num_sentences, accepted_trees, tag_pipe, parser_pipes, shuffle=False, chunk_size=100)
        accepted_trees.update(new_trees)

        with open(args.output_file, "a") as fout:
            for tree in sorted(new_trees):
                fout.write(tree)
                fout.write("\n")

if __name__ == '__main__':
    main()
