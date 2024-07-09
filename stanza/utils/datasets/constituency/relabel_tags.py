"""
Retag an S-expression tree with a new set of POS tags

Also includes an option to write the new trees as bracket_labels
(essentially, skipping the treebank_to_labeled_brackets step)
"""

import argparse
import logging

from stanza import Pipeline
from stanza.models.constituency import retagging
from stanza.models.constituency import tree_reader
from stanza.models.constituency.utils import retag_trees

logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser(description="Script that retags a tree file")
    parser.add_argument('--lang', default='vi', type=str, help='Language')
    parser.add_argument('--input_file', default='data/constituency/vi_vlsp21_train.mrg', help='File to retag')
    parser.add_argument('--output_file', default='vi_vlsp21_train_retagged.mrg', help='Where to write the retagged trees')
    retagging.add_retag_args(parser)

    parser.add_argument('--bracket_labels', action='store_true', help='Write the trees as bracket labels instead of S-expressions')

    args = parser.parse_args()
    args = vars(args)
    retagging.postprocess_args(args)

    return args

def main():
    args = parse_args()

    retag_pipeline = retagging.build_retag_pipeline(args)

    train_trees = tree_reader.read_treebank(args['input_file'])
    logger.info("Retagging %d trees using %s", len(train_trees), args['retag_package'])
    train_trees = retag_trees(train_trees, retag_pipeline, args['retag_xpos'])
    tree_format = "{:L}" if args['bracket_labels'] else "{}"
    with open(args['output_file'], "w") as fout:
        for tree in train_trees:
            fout.write(tree_format.format(tree))
            fout.write("\n")

if __name__ == '__main__':
    main()
