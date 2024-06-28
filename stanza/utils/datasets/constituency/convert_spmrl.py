import os

from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_reader import read_treebank
from stanza.utils.default_paths import get_default_paths

SHARDS = ("train", "dev", "test")

def add_root(tree):
    if tree.label.startswith("NN"):
        tree = Tree("NP", tree)
    if tree.label.startswith("NE"):
        tree = Tree("PN", tree)
    elif tree.label.startswith("XY"):
        tree = Tree("VROOT", tree)
    return Tree("ROOT", tree)

def convert_spmrl(input_directory, output_directory, short_name):
    for shard in SHARDS:
        tree_filename = os.path.join(input_directory, shard, shard + ".German.gold.ptb")
        trees = read_treebank(tree_filename, tree_callback=add_root)
        output_filename = os.path.join(output_directory, "%s_%s.mrg" % (short_name, shard))
        with open(output_filename, "w", encoding="utf-8") as fout:
            for tree in trees:
                fout.write(str(tree))
                fout.write("\n")
        print("Wrote %d trees to %s" % (len(trees), output_filename))

if __name__ == '__main__':
    paths = get_default_paths()
    output_directory = paths["CONSTITUENCY_DATA_DIR"]
    input_directory = "extern_data/constituency/spmrl/SPMRL_SHARED_2014/GERMAN_SPMRL/gold/ptb"
    convert_spmrl(input_directory, output_directory, "de_spmrl")


