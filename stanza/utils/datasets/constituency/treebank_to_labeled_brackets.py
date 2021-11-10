"""
Converts a PTB file to a format where all the brackets have labels on the start and end bracket.

Such a file should be suitable for training an LM
"""

import argparse
import logging
import sys

from stanza.models.common import utils
from stanza.models.constituency import tree_reader

tqdm = utils.get_tqdm()

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

    args = parser.parse_args()

    treebank = tree_reader.read_treebank(args.ptb_file)
    logger.info("Writing %d trees to %s", len(treebank), args.label_file)

    with open(args.label_file, "w", encoding="utf-8") as fout:
        for tree in tqdm(treebank):
            fout.write("{:L}\n".format(tree))

if __name__ == '__main__':
    main()
