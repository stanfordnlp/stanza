"""
Utilities for the processing of constituency treebanks
"""

import os

from stanza.models.constituency import parse_tree

SHARDS = ("train", "dev", "test")

def write_dataset(datasets, output_dir, dataset_name):
    for dataset, shard in zip(datasets, SHARDS):
        output_filename = os.path.join(output_dir, "%s_%s.mrg" % (dataset_name, shard))
        print("Writing {} trees to {}".format(len(dataset), output_filename))
        parse_tree.Tree.write_treebank(dataset, output_filename)

def split_treebank(treebank, train_size, dev_size):
    """
    Split a treebank deterministically
    """
    train_end = int(len(treebank) * train_size)
    dev_end = int(len(treebank) * (train_size + dev_size))
    return treebank[:train_end], treebank[train_end:dev_end], treebank[dev_end:]
