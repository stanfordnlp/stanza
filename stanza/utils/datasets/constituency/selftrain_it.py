"""Builds a self-training dataset from an Italian data source and two models

The idea is that the top down and the inorder parsers should make
somewhat different errors, so hopefully the sum of an 86 f1 parser and
an 85.5 f1 parser will produce some half-decent silver trees which can
be used as self-training so that a new model can do better than either.

One dataset used is PaCCSS, which has 63000 pairs of sentences:

http://www.italianlp.it/resources/paccss-it-parallel-corpus-of-complex-simple-sentences-for-italian/

PaCCSS-IT: A Parallel Corpus of Complex-Simple Sentences for Automatic Text Simplification
  Brunato, Dominique et al, 2016
  https://aclanthology.org/D16-1034

Even larger is the IT section of Europarl, which has 1900000 lines

https://www.statmt.org/europarl/

Europarl: A Parallel Corpus for Statistical Machine Translation
  Philipp Koehn
  https://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf
"""

import argparse
import logging
import os
import random

import stanza
from stanza.models.common import utils
from stanza.utils.datasets.constituency import selftrain

tqdm = utils.get_tqdm()
logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that converts a public IT dataset to silver standard trees"
    )
    selftrain.common_args(parser)
    parser.add_argument(
        '--input_dir',
        default='extern_data/italian',
        help='Path to the PaCCSS corpus and europarl corpus'
    )

    parser.set_defaults(lang="it")
    parser.set_defaults(models="saved_models/constituency/it_inorder.pt,saved_models/constituency/it_topdown.pt")
    parser.set_defaults(output_file="data/constituency/it_silver.mrg")

    args = parser.parse_args()
    return args

def get_paccss(input_dir):
    """
    Read the paccss dataset, which is two sentences per line
    """
    input_file = os.path.join(input_dir, "PaCCSS/data-set/PACCSS-IT.txt")
    with open(input_file) as fin:
        # the first line is a header line
        lines = fin.readlines()[1:]
    lines = [x.strip() for x in lines]
    lines = [x.split("\t")[:2] for x in lines if x]
    text = [y for x in lines for y in x]
    logger.info("Read %d sentences from %s", len(text), input_file)
    return text

def get_europarl(input_dir, ssplit_pipe):
    """
    Read the Europarl dataset

    This dataset needs to be tokenized and split into lines
    """
    input_file = os.path.join(input_dir, "europarl/europarl-v7.it-en.it")
    with open(input_file) as fin:
        # the first line is a header line
        lines = fin.readlines()[1:]
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    logger.info("Read %d docs from %s", len(lines), input_file)
    lines = selftrain.split_docs(lines, ssplit_pipe)
    return lines

def main():
    """
    Combine the two datasets, parse them, and write out the results
    """
    args = parse_args()

    ssplit_pipe = selftrain.build_ssplit_pipe(ssplit=True, lang=args.lang)
    tag_pipe = selftrain.build_tag_pipe(ssplit=False, lang=args.lang)
    parser_pipes = selftrain.build_parser_pipes(args.lang, args.models)

    docs = get_paccss(args.input_dir)
    docs.extend(get_europarl(args.input_dir, ssplit_pipe))

    new_trees = selftrain.find_matching_trees(docs, args.num_sentences, set(), tag_pipe, parser_pipes, shuffle=False, chunk_size=100)
    print("Found %d unique trees which are the same between models" % len(new_trees))
    with open(args.output_file, "w") as fout:
        for tree in sorted(new_trees):
            fout.write(tree)
            fout.write("\n")


if __name__ == '__main__':
    main()
