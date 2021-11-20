"""
Retag an S-expression tree with a new set of POS tags

Also includes an option to write the new trees as bracket_labels
(essentially, skipping the treebank_to_labeled_brackets step)
"""

import argparse
import logging

from stanza import Pipeline
from stanza.models.constituency import tree_reader
from stanza.models.constituency.utils import retag_trees

logger = logging.getLogger('stanza')

def parse_args():
    parser = argparse.ArgumentParser(description="Script that retags a tree file")
    parser.add_argument('--lang', default='vi', type=str, help='Language')
    parser.add_argument('--input_file', default='data/constituency/vi_vlsp21_train.mrg', help='File to retag')
    parser.add_argument('--output_file', default='vi_vlsp21_train_retagged.mrg', help='Where to write the retagged trees')
    parser.add_argument('--retag_package', default="default", help='Which tagger shortname to use when retagging trees.  None for no retagging.  Retagging is recommended, as gold tags will not be available at pipeline time')
    parser.add_argument('--retag_method', default='upos', choices=['xpos', 'upos'], help='Which tags to use when retagging')

    parser.add_argument('--bracket_labels', action='store_true', help='Write the trees as bracket labels instead of S-expressions')

    args = parser.parse_args()
    args = vars(args)

    if args['retag_method'] == 'xpos':
        args['retag_xpos'] = True
    elif args['retag_method'] == 'upos':
        args['retag_xpos'] = False
    else:
        raise ValueError("Unknown retag method {}".format(xpos))

    return args

def main():
    args = parse_args()

    # TODO: refactor with constituency_parser?
    if '_' in args['retag_package']:
        lang, package = args['retag_package'].split('_', 1)
    else:
        lang = args['lang']
        package = args['retag_package']
    retag_pipeline = Pipeline(lang=lang, processors="tokenize, pos", tokenize_pretokenized=True, pos_package=package, pos_tqdm=True)

    train_trees = tree_reader.read_treebank(args['input_file'])
    logger.info("Retagging %d trees using %s_%s", len(train_trees), lang, package)
    train_trees = retag_trees(train_trees, retag_pipeline, args['retag_xpos'])
    tree_format = "{:L}" if args['bracket_labels'] else "{}"
    with open(args['output_file'], "w") as fout:
        for tree in train_trees:
            fout.write(tree_format.format(tree))
            fout.write("\n")

if __name__ == '__main__':
    main()
