"""
Turns a constituency treebank into a POS dataset with the tags as the upos column

Note that this doesn't pay any attention to whether or not the tags actually are upos

Also not possible: using this for tokenization

TODO: upgrade the POS model to handle xpos datasets with no upos, then make upos/xpos an option here

To run this:
  python3 stanza/utils/training/run_pos.py vi_vlsp22
"""

import os
import shutil
import sys

from stanza.models.common.utils import get_tqdm
from stanza.models.constituency import tree_reader
import stanza.utils.default_paths as default_paths

tqdm = get_tqdm()

SHARDS = ("train", "dev", "test")

def convert_file(in_file, out_file):
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
                # always xpos (for now)
                fout.write("%s\t" % pt.label)
                # don't know upos or features
                fout.write("_\t_\t")
                # so word 0 fake dep on root, everyone else fake dep on previous word
                fout.write("%d\t" % pt_idx)
                if pt_idx == 0:
                    fout.write("root")
                else:
                    fout.write("dep")
                fout.write("\t_\t_\n")
            fout.write("\n")

def convert_treebank(short_name, paths):
    in_dir = paths["CONSTITUENCY_DATA_DIR"]
    in_files = [os.path.join(in_dir, "%s_%s.mrg" % (short_name, shard)) for shard in SHARDS]
    for in_file in in_files:
        if not os.path.exists(in_file):
            raise FileNotFoundError("Cannot find expected datafile %s" % in_file)

    out_dir = paths["POS_DATA_DIR"]
    out_files = [os.path.join(out_dir, "%s.%s.in.conllu" % (short_name, shard)) for shard in SHARDS]
    gold_files = [os.path.join(out_dir, "%s.%s.gold.conllu" % (short_name, shard)) for shard in SHARDS]

    for in_file, out_file in zip(in_files, out_files):
        convert_file(in_file, out_file)
    for out_file, gold_file in zip(out_files, gold_files):
        shutil.copy2(out_file, gold_file)

if __name__ == '__main__':
    paths = default_paths.get_default_paths()

    convert_treebank(sys.argv[1], paths)
