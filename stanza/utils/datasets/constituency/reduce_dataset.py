"""
Cut short the training portion of a constituency dataset.

One could think this script isn't necessary, as shuf | head would work,
but some treebanks use multiple lines for representing trees.
Thus it is necessary to actually intelligently read the trees.

Run with

python3  stanza/utils/datasets/constituency/reduce_dataset.py --input zh-hans_ctb-51b --output zh-hans_ctb5k
"""

import argparse
import os
import random

from stanza.models.constituency import tree_reader
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.constituency.utils import SHARDS, write_dataset

def main():
    parser = argparse.ArgumentParser(description="Script that cuts a treebank down to size")
    parser.add_argument('--input', type=str, default=None, help='Input treebank')
    parser.add_argument('--output', type=str, default=None, help='Output treebank')
    parser.add_argument('--size', type=int, default=5000, help='How many trees')
    args = parser.parse_args()

    random.seed(1234)

    paths = default_paths.get_default_paths()
    output_directory = paths["CONSTITUENCY_DATA_DIR"]

    # data/constituency/en_ptb3_train.mrg
    input_filenames = [os.path.join(output_directory, "%s_%s.mrg" % (args.input, shard)) for shard in SHARDS]
    output_filenames = ["%s_%s.mrg" % (args.output, shard) for shard in SHARDS]
    shrink_datasets = [True, False, False]

    datasets = []
    for input_filename, shrink in zip(input_filenames, shrink_datasets):
        treebank = tree_reader.read_treebank(input_filename)
        if shrink:
            random.shuffle(treebank)
            treebank = treebank[:args.size]
        datasets.append(treebank)
    write_dataset(datasets, output_directory, args.output)

if __name__ == '__main__':
    main()
