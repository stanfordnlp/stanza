"""
Turns a constituency treebank into a POS dataset with the tags as the upos column

The constituency treebank first has to be converted from the original
data to PTB style trees.  This script converts trees from the
CONSTITUENCY_DATA_DIR folder to a conllu dataset in the POS_DATA_DIR folder.

Note that this doesn't pay any attention to whether or not the tags actually are upos.
Also not possible: using this for tokenization.

TODO: upgrade the POS model to handle xpos datasets with no upos, then make upos/xpos an option here

To run this:
  python3 stanza/utils/training/run_pos.py vi_vlsp22

"""

import argparse
import os
import shutil
import sys

from stanza.models.constituency import tree_reader
import stanza.utils.default_paths as default_paths
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

SHARDS = ("train", "dev", "test")

def convert_file(in_file, out_file, upos):
    print("Reading %s" % in_file)
    trees = tree_reader.read_tree_file(in_file)
    print("Writing %s" % out_file)
    with open(out_file, "w") as fout:
        for tree in tqdm(trees):
            tree = tree.simplify_labels()
            text = " ".join(tree.leaf_labels())
            fout.write("# text = %s\n" % text)

            for pt_idx, pt in enumerate(tree.yield_preterminals()):
                # word index
                fout.write("%d\t" % (pt_idx+1))
                # word
                fout.write("%s\t" % pt.children[0].label)
                # don't know the lemma
                fout.write("_\t")
                # always put the tag, whatever it is, in the upos (for now)
                if upos:
                    fout.write("%s\t_\t" % pt.label)
                else:
                    fout.write("_\t%s\t" % pt.label)
                # don't have any features
                fout.write("_\t")
                # so word 0 fake dep on root, everyone else fake dep on previous word
                fout.write("%d\t" % pt_idx)
                if pt_idx == 0:
                    fout.write("root")
                else:
                    fout.write("dep")
                fout.write("\t_\t_\n")
            fout.write("\n")

def convert_treebank(short_name, upos, output_name, paths):
    in_dir = paths["CONSTITUENCY_DATA_DIR"]
    in_files = [os.path.join(in_dir, "%s_%s.mrg" % (short_name, shard)) for shard in SHARDS]
    for in_file in in_files:
        if not os.path.exists(in_file):
            raise FileNotFoundError("Cannot find expected datafile %s" % in_file)

    out_dir = paths["POS_DATA_DIR"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if output_name is None:
        output_name = short_name
    out_files = [os.path.join(out_dir, "%s.%s.in.conllu" % (output_name, shard)) for shard in SHARDS]
    gold_files = [os.path.join(out_dir, "%s.%s.gold.conllu" % (output_name, shard)) for shard in SHARDS]

    for in_file, out_file in zip(in_files, out_files):
        convert_file(in_file, out_file, upos)
    for out_file, gold_file in zip(out_files, gold_files):
        shutil.copy2(out_file, gold_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Which dataset to process from trees to POS")
    parser.add_argument("--upos", action="store_true", default=False, help="Store tags on the UPOS")
    parser.add_argument("--xpos", dest="upos", action="store_false", help="Store tags on the XPOS")
    parser.add_argument("--output_name", default=None, help="What name to give the output dataset.  If blank, will use the dataset arg")
    args = parser.parse_args()

    paths = default_paths.get_default_paths()

    convert_treebank(args.dataset, args.upos, args.output_name, paths)
