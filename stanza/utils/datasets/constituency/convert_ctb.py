import glob
import os

from stanza.models.constituency import tree_reader
from stanza.utils.datasets.constituency.utils import write_dataset

def filenum_to_shard(filenum):
    if filenum >= 1 and filenum <= 815:
        return 0
    if filenum >= 1001 and filenum <= 1136:
        return 0

    if filenum >= 886 and filenum <= 931:
        return 1
    if filenum >= 1148 and filenum <= 1151:
        return 1

    if filenum >= 816 and filenum <= 885:
        return 2
    if filenum >= 1137 and filenum <= 1147:
        return 2

    raise ValueError("Unhandled filenum %d" % filenum)

def convert_ctb(input_dir, output_dir, dataset_name):
    input_files = glob.glob(os.path.join(input_dir, "*"))

    # train, dev, test
    datasets = [[], [], []]

    sorted_filenames = []
    for input_filename in input_files:
        base_filename = os.path.split(input_filename)[1]        
        filenum = int(os.path.splitext(base_filename)[0].split("_")[1])
        sorted_filenames.append((filenum, input_filename))
    sorted_filenames.sort()

    for filenum, filename in sorted_filenames:
        line_idx = -1
        with open(filename, errors='ignore', encoding="gb2312") as fin:
            trees = []
            in_tree = False
            for line_idx, line in enumerate(fin):
                #print(line, line.startswith("<S"), line.startswith("</S"))
                if line.startswith("<S"):
                    tree_id = line.strip().split("=")[1][:-1]
                    in_tree = True
                    tree_text = []
                elif line.startswith("</S"):
                    in_tree = False
                    if filenum == 414 and tree_id == "4366":
                        print("SKIPPING A BROKEN TREE IN %d" % filenum)
                        continue
                    else:
                        trees.append("".join(tree_text))
                    tree_text = []
                elif in_tree:
                    tree_text.append(line)

            trees = "\n".join(trees)
            trees = tree_reader.read_trees(trees)
            trees = [t.prune_none().simplify_labels() for t in trees]

            assert len(trees) > 0
            assert len(tree_text) == 0

            shard = filenum_to_shard(filenum)
            datasets[shard].extend(trees)


    write_dataset(datasets, output_dir, dataset_name)
