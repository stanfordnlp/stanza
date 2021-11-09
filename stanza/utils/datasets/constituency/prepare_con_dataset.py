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
"""

import os
import random
import sys
import tempfile

import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.constituency.convert_it_turin import convert_it_turin
import stanza.utils.datasets.constituency.vtb_convert as vtb_convert
import stanza.utils.datasets.constituency.vtb_split as vtb_split

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

def main(dataset_name):
    paths = default_paths.get_default_paths()

    random.seed(1234)

    if dataset_name == 'it_turin':
        process_it_turin(paths)
    elif dataset_name == 'vi_vlsp09':
        process_vlsp09(paths)
    elif dataset_name == 'vi_vlsp21':
        process_vlsp21(paths)
    else:
        raise ValueError(f"dataset {dataset_name} currently not handled")

if __name__ == '__main__':
    main(sys.argv[1])


