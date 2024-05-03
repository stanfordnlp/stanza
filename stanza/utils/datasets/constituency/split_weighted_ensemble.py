"""
Read in a dataset and split the train portion into pieces

One chunk of the train will be the original dataset.

Others will be a sampling from the original dataset of the same size,
but sampled with replacement, with the goal being to get a random
distribution of trees with some reweighting of the original trees.
"""

import argparse
import os
import random

from stanza.models.constituency import tree_reader
from stanza.models.constituency.parse_tree import Tree
from stanza.utils.datasets.constituency.utils import copy_dev_test
from stanza.utils.default_paths import get_default_paths

def main():
    parser = argparse.ArgumentParser(description="Split a standard dataset into 1 base section and N-1 random redraws of training data")
    parser.add_argument('--dataset', type=str, default="id_icon", help='dataset to split')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of splits')
    args = parser.parse_args()

    random.seed(args.seed)

    base_path = get_default_paths()["CONSTITUENCY_DATA_DIR"]
    train_file = os.path.join(base_path, "%s_train.mrg" % args.dataset)
    print("Reading %s" % train_file)
    train_trees = tree_reader.read_tree_file(train_file)

    # For datasets with low numbers of certain constituents in the train set,
    # we could easily find ourselves in a situation where all of the trees
    # with a specific constituent have been randomly shuffled away from
    # a random shuffle
    # An example of this is there are 3 total trees with SQ in id_icon
    # Therefore, we have to take a little care to guarantee at least one tree
    # for each constituent type is in a random slice
    # TODO: this doesn't compensate for transition schemes with compound transitions,
    # such as in_order_compound.  could do a similar boosting with one per transition type
    constituents = sorted(Tree.get_unique_constituent_labels(train_trees))
    con_to_trees = {con: list() for con in constituents}
    for tree in train_trees:
        tree_cons = Tree.get_unique_constituent_labels(tree)
        for con in tree_cons:
            con_to_trees[con].append(tree)
    for con in constituents:
        print("%d trees with %s" % (len(con_to_trees[con]), con))

    for i in range(args.num_splits):
        dataset_name = "%s-random-%d" % (args.dataset, i)

        copy_dev_test(base_path, args.dataset, dataset_name)
        if i == 0:
            train_dataset = train_trees
        else:
            train_dataset = []
            for con in constituents:
                train_dataset.extend(random.choices(con_to_trees[con], k=2))
            needed_trees = len(train_trees) - len(train_dataset)
            if needed_trees > 0:
                print("%d trees already chosen.  Adding %d more" % (len(train_dataset), needed_trees))
                train_dataset.extend(random.choices(train_trees, k=needed_trees))
        output_filename = os.path.join(base_path, "%s_train.mrg" % dataset_name)
        print("Writing {} trees to {}".format(len(train_dataset), output_filename))
        Tree.write_treebank(train_dataset, output_filename)


if __name__ == '__main__':
    main()

