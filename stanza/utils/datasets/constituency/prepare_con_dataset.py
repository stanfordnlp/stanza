"""Converts raw data files from their original format (dataset dependent) into PTB trees.

The operation of this script depends heavily on the dataset in question.
The common result is that the data files go to data/constituency and are in PTB format.

it_turin
  A combination of Evalita competition from 2011 and the ParTUT trees
  More information is available in convert_it_turin
vlsp09 is the 2009 constituency treebank:
  Nguyen Phuong Thai, Vu Xuan Luong, Nguyen Thi Minh Huyen, Nguyen Van Hiep, Le Hong Phuong
    Building a Large Syntactically-Annotated Corpus of Vietnamese
    Proceedings of The Third Linguistic Annotation Workshop
    In conjunction with ACL-IJCNLP 2009, Suntec City, Singapore, 2009
  This can be obtained by contacting vlsp.resources@gmail.com

da_arboretum
  Ekhard Bick
    Arboretum, a Hybrid Treebank for Danish
    https://www.researchgate.net/publication/251202293_Arboretum_a_Hybrid_Treebank_for_Danish
  Available here for a license fee:
    http://catalog.elra.info/en-us/repository/browse/ELRA-W0084/
  Internal to Stanford, please contact Chris Manning and/or John Bauer
  The file processed is the tiger xml, although there are some edits
    needed in order to make it functional for our parser
  The treebank comes as a tar.gz file, W0084.tar.gz
  untar this file in $CONSTITUENCY_HOME/danish
  then move the extracted folder to "arboretum"
    $CONSTITUENCY_HOME/danish/W0084/... becomes
    $CONSTITUENCY_HOME/danish/arboretum/...
"""

import os
import random
import sys
import tempfile

import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.constituency.convert_arboretum import convert_tiger_treebank
from stanza.utils.datasets.constituency.convert_it_turin import convert_it_turin
import stanza.utils.datasets.constituency.vtb_convert as vtb_convert
import stanza.utils.datasets.constituency.vtb_split as vtb_split

SHARDS = ("train", "dev", "test")

def process_it_turin(paths):
    """
    Convert the it_turin dataset
    """
    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "italian")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    convert_it_turin(input_dir, output_dir)

def process_vlsp09(paths):
    """
    Processes the VLSP 2009 dataset, discarding or fixing trees when needed
    """
    short_name = "vi_vlsp09"
    vlsp_path = os.path.join(paths["CONSTITUENCY_BASE"], "vietnamese", "VietTreebank_VLSP_SP73", "Kho ngu lieu 10000 cay cu phap")
    with tempfile.TemporaryDirectory() as tmp_output_path:
        vtb_convert.convert_dir(vlsp_path, tmp_output_path)
        vtb_split.split_files(tmp_output_path, paths["CONSTITUENCY_DATA_DIR"], short_name)

def process_vlsp21(paths):
    """
    Processes the VLSP 2021 dataset, which is just a single file
    """
    short_name = "vi_vlsp21"
    vlsp_file = os.path.join(paths["CONSTITUENCY_BASE"], "vietnamese", "VLSP_2021", "VTB_VLSP21_tree.txt")
    if not os.path.exists(vlsp_file):
        raise FileNotFoundError("Could not find the 2021 dataset in the expected location of {} - CONSTITUENCY_BASE == {}".format(vlsp_file, paths["CONSTITUENCY_BASE"]))
    with tempfile.TemporaryDirectory() as tmp_output_path:
        vtb_convert.convert_files([vlsp_file], tmp_output_path)
        # This produces a tiny test set, just as a placeholder until the actual test set is released
        vtb_split.split_files(tmp_output_path, paths["CONSTITUENCY_DATA_DIR"], short_name, train_size=0.9, dev_size=0.1)
    _, _, test_file = vtb_split.create_paths(paths["CONSTITUENCY_DATA_DIR"], short_name)
    with open(test_file, "w"):
        # create an empty test file - currently we don't have actual test data for VLSP 21
        pass

def split_treebank(treebank, train_size, dev_size):
    train_end = int(len(treebank) * train_size)
    dev_end = int(len(treebank) * (train_size + dev_size))
    return treebank[:train_end], treebank[train_end:dev_end], treebank[dev_end:]

def process_arboretum(paths, dataset_name):
    """
    Processes the Danish dataset, Arboretum
    """
    assert dataset_name == 'da_arboretum'

    arboretum_file = os.path.join(paths["CONSTITUENCY_BASE"], "danish", "arboretum", "arboretum.tiger", "arboretum.tiger")
    if not os.path.exists(arboretum_file):
        raise FileNotFoundError("Unable to find input file for Arboretum.  Expected in {}".format(arboretum_file))

    treebank = convert_tiger_treebank(arboretum_file)
    datasets = split_treebank(treebank, 0.8, 0.1)
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    output_filename = os.path.join(output_dir, "%s.mrg" % dataset_name)
    print("Writing {} trees to {}".format(len(treebank), output_filename))
    with open(output_filename, "w", encoding="utf-8") as fout:
        for tree in treebank:
            fout.write("{}".format(tree))
            fout.write("\n")
    for dataset, shard in zip(datasets, SHARDS):
        output_filename = os.path.join(output_dir, "%s_%s.mrg" % (dataset_name, shard))
        with open(output_filename, "w", encoding="utf-8") as fout:
            for tree in dataset:
                fout.write("{}".format(tree))
                fout.write("\n")


def main(dataset_name):
    paths = default_paths.get_default_paths()

    random.seed(1234)

    if dataset_name == 'it_turin':
        process_it_turin(paths)
    elif dataset_name == 'vi_vlsp09':
        process_vlsp09(paths)
    elif dataset_name == 'vi_vlsp21':
        process_vlsp21(paths)
    elif dataset_name == 'da_arboretum':
        process_arboretum(paths, dataset_name)
    else:
        raise ValueError(f"dataset {dataset_name} currently not handled")

if __name__ == '__main__':
    main(sys.argv[1])


