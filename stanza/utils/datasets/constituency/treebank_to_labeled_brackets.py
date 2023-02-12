"""
Converts a PTB file to a format where all the brackets have labels on the start and end bracket.

Such a file should be suitable for training an LM
"""

import argparse
import logging
import sys

from stanza.models.constituency import tree_reader
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logger = logging.getLogger('stanza.constituency')

def main():
    parser = argparse.ArgumentParser(
        description="Script that converts a PTB treebank into a labeled bracketed file suitable for LM training"
    )

    parser.add_argument(
        'ptb_file',
        help='Where to get the original PTB format treebank'
    )
    parser.add_argument(
        'label_file',
        help='Where to write the labeled bracketed file'
    )
    parser.add_argument(
        '--separator',
        default="_",
        help='What separator to use in place of spaces',
    )
    parser.add_argument(
        '--no_separator',
        dest='separator',
        action='store_const',
        const=None,
        help="Don't use a separator"
    )

    args = parser.parse_args()

    treebank = tree_reader.read_treebank(args.ptb_file)
    logger.info("Writing %d trees to %s", len(treebank), args.label_file)

    tree_format = "{:%sL}\n" % args.separator if args.separator else "{:L}\n"
    with open(args.label_file, "w", encoding="utf-8") as fout:
        for tree in tqdm(treebank):
            fout.write(tree_format.format(tree))

if __name__ == '__main__':
    main()
