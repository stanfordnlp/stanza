"""
Split a constituency dataset randomly into 90/10 splits

TODO: add a function to rotate the pieces of the split so that each
training instance gets seen once
"""

import argparse
import os
import random

from stanza.models.constituency import tree_reader
from stanza.utils.datasets.constituency.utils import copy_dev_test
from stanza.utils.default_paths import get_default_paths

def write_trees(base_path, dataset_name, trees):
    output_path = os.path.join(base_path, "%s_train.mrg" % dataset_name)
    with open(output_path, "w", encoding="utf-8") as fout:
        for tree in trees:
            fout.write("%s\n" % tree)


def main():
    parser = argparse.ArgumentParser(description="Split a standard dataset into 90/10 proportions of train so there is held out training data")
    parser.add_argument('--dataset', type=str, default="id_icon", help='dataset to split')
    parser.add_argument('--base_dataset', type=str, default=None, help='output name for base dataset')
    parser.add_argument('--holdout_dataset', type=str, default=None, help='output name for holdout dataset')
    parser.add_argument('--ratio', type=float, default=0.1, help='Number of trees to hold out')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    args = parser.parse_args()

    if args.base_dataset is None:
        args.base_dataset = args.dataset + "-base"
        print("--base_dataset not set, using %s" % args.base_dataset)

    if args.holdout_dataset is None:
        args.holdout_dataset = args.dataset + "-holdout"
        print("--holdout_dataset not set, using %s" % args.holdout_dataset)

    base_path = get_default_paths()["CONSTITUENCY_DATA_DIR"]
    copy_dev_test(base_path, args.dataset, args.base_dataset)
    copy_dev_test(base_path, args.dataset, args.holdout_dataset)

    train_file = os.path.join(base_path, "%s_train.mrg" % args.dataset)
    print("Reading %s" % train_file)
    trees = tree_reader.read_tree_file(train_file)

    base_train = []
    holdout_train = []

    random.seed(args.seed)

    for tree in trees:
        if random.random() < args.ratio:
            holdout_train.append(tree)
        else:
            base_train.append(tree)

    write_trees(base_path, args.base_dataset, base_train)
    write_trees(base_path, args.holdout_dataset, holdout_train)

if __name__ == '__main__':
    main()

