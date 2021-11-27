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


def split_docs(docs, ssplit_pipe, max_len=140, max_word_len=100, chunk_size=2000):
    """
    Using the ssplit pipeline, break up the documents into sentences

    Filters out sentences which are too long or have words too long.

    This step is necessary because some web text has unstructured
    sentences which overwhelm the tagger, or even text with no
    whitespace which breaks the charlm in the tokenizer or tagger
    """
    raw_sentences = 0
    filtered_sentences = 0
    new_docs = []

    logger.info("Number of raw docs: %d", len(docs))
    for chunk_start in range(0, len(docs), chunk_size):
        chunk = docs[chunk_start:chunk_start+chunk_size]
        chunk = [stanza.Document([], text=t) for t in chunk]
        chunk = ssplit_pipe(chunk)
        sentences = [s for d in chunk for s in d.sentences]
        raw_sentences += len(sentences)
        sentences = [s for s in sentences if len(s.words) < max_len]
        sentences = [s for s in sentences if max(len(w.text) for w in s.words) < max_word_len]
        filtered_sentences += len(sentences)
        new_docs.extend([s.text for s in sentences])

    logger.info("Split sentences: %d", raw_sentences)
    logger.info("Sentences filtered for length: %d", filtered_sentences)
    return new_docs


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
    docs = split_docs(docs, ssplit_pipe)

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
