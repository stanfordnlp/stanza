"""Converts raw data files from their original format (dataset dependent) into PTB trees.

The operation of this script depends heavily on the dataset in question.
The common result is that the data files go to data/constituency and are in PTB format.

it_turin
  A combination of Evalita competition from 2011 and the ParTUT trees
  More information is available in convert_it_turin

it_vit
  The original for the VIT UD Dataset
  The UD version has a lot of corrections, so we try to apply those as much as possible
  In fact, we applied some corrections of our own back to UD based on this treebank.
    The first version which had those corrections is UD 2.10
    Versions of UD before that won't work
    Hopefully versions after that work
    Set UDBASE to a path such that $UDBASE/UD_Italian-VIT is the UD version
  The constituency labels are generally not very understandable, unfortunately
    Some documentation is available here:
      https://core.ac.uk/download/pdf/223148096.pdf
      https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.423.5538&rep=rep1&type=pdf
  Available from ELRA:
    http://catalog.elra.info/en-us/repository/browse/ELRA-W0040/

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
  untar this file in $CONSTITUENCY_BASE/danish
  then move the extracted folder to "arboretum"
    $CONSTITUENCY_BASE/danish/W0084/... becomes
    $CONSTITUENCY_BASE/danish/arboretum/...

tr_starlang
  A dataset in three parts from the Starlang group in Turkey:
  Neslihan Kara, Büşra Marşan, et al
    Creating A Syntactically Felicitous Constituency Treebank For Turkish
    https://ieeexplore.ieee.org/document/9259873
  git clone the following three repos
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank-15
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank2-15
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank2-20
  Put them in
    $CONSTITUENCY_BASE/turkish
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset tr_starlang

ja_alt
  Asian Language Treebank produced a treebank for Japanese:
    Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch, Eiichiro Sumita
    Introducing the Asian Language Treebank
    http://www.lrec-conf.org/proceedings/lrec2016/pdf/435_Paper.pdf
  Download
    https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/Japanese-ALT-20210218.zip
  unzip this in $CONSTITUENCY_BASE/japanese
  this should create a directory $CONSTITUENCY_BASE/japanese/Japanese-ALT-20210218
  In this directory, also download the following:
    https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-train.txt
    https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-dev.txt
    https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-test.txt
  In particular, there are two files with a bunch of bracketed parses,
    Japanese-ALT-Draft.txt and Japanese-ALT-Reviewed.txt
  The first word of each of these lines is SNT.80188.1 or something like that
  This correlates with the three URL-... files, telling us whether the
    sentence belongs in train/dev/test
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset ja_alt

pt_cintil
  CINTIL treebank for Portuguese, available at ELRA:
    https://catalogue.elra.info/en-us/repository/browse/ELRA-W0055/
  It can also be obtained from here:
    https://hdl.handle.net/21.11129/0000-000B-D2FE-A
  Produced at U Lisbon
    António Branco; João Silva; Francisco Costa; Sérgio Castro
      CINTIL TreeBank Handbook: Design options for the representation of syntactic constituency
    Silva, João; António Branco; Sérgio Castro; Ruben Reis
      Out-of-the-Box Robust Parsing of Portuguese
    https://portulanclarin.net/repository/extradocs/CINTIL-Treebank.pdf
    http://www.di.fc.ul.pt/~ahb/pubs/2011bBrancoSilvaCostaEtAl.pdf
  If at Stanford, ask John Bauer or Chris Manning for the data
  Otherwise, purchase it from ELRA or find it elsewhere if possible
  Either way, unzip it in
    $CONSTITUENCY_BASE/portuguese to the CINTIL directory
    so for example, the final result might be
    extern_data/constituency/portuguese/CINTIL/CINTIL-Treebank.xml
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset pt_cintil
"""

import argparse
import os
import random
import sys
import tempfile

from stanza.models.constituency import parse_tree
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.constituency import utils
from stanza.utils.datasets.constituency.convert_alt import convert_alt
from stanza.utils.datasets.constituency.convert_arboretum import convert_tiger_treebank
from stanza.utils.datasets.constituency.convert_cintil import convert_cintil_treebank
from stanza.utils.datasets.constituency.convert_it_turin import convert_it_turin
from stanza.utils.datasets.constituency.convert_it_vit import convert_it_vit
from stanza.utils.datasets.constituency.convert_starlang import read_starlang
from stanza.utils.datasets.constituency.utils import SHARDS, write_dataset
import stanza.utils.datasets.constituency.vtb_convert as vtb_convert
import stanza.utils.datasets.constituency.vtb_split as vtb_split

class UnknownDatasetError(ValueError):
    def __init__(self, dataset, text):
        super().__init__(text)
        self.dataset = dataset

def process_it_turin(paths, dataset_name, *args):
    """
    Convert the it_turin dataset
    """
    assert dataset_name == 'it_turin'
    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "italian")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    convert_it_turin(input_dir, output_dir)

def process_it_vit(paths, dataset_name, *args):
    # needs at least UD 2.11 or this will not work
    # in the meantime, the git version of VIT will suffice
    assert dataset_name == 'it_vit'
    convert_it_vit(paths, dataset_name)

def process_vlsp09(paths, dataset_name, *args):
    """
    Processes the VLSP 2009 dataset, discarding or fixing trees when needed
    """
    assert dataset_name == 'vi_vlsp09'
    vlsp_path = os.path.join(paths["CONSTITUENCY_BASE"], "vietnamese", "VietTreebank_VLSP_SP73", "Kho ngu lieu 10000 cay cu phap")
    with tempfile.TemporaryDirectory() as tmp_output_path:
        vtb_convert.convert_dir(vlsp_path, tmp_output_path)
        vtb_split.split_files(tmp_output_path, paths["CONSTITUENCY_DATA_DIR"], dataset_name)

def process_vlsp21(paths, dataset_name, *args):
    """
    Processes the VLSP 2021 dataset, which is just a single file
    """
    assert dataset_name == 'vi_vlsp21'
    vlsp_file = os.path.join(paths["CONSTITUENCY_BASE"], "vietnamese", "VLSP_2021", "VTB_VLSP21_tree.txt")
    if not os.path.exists(vlsp_file):
        raise FileNotFoundError("Could not find the 2021 dataset in the expected location of {} - CONSTITUENCY_BASE == {}".format(vlsp_file, paths["CONSTITUENCY_BASE"]))
    with tempfile.TemporaryDirectory() as tmp_output_path:
        vtb_convert.convert_files([vlsp_file], tmp_output_path)
        # This produces a 0 length test set, just as a placeholder until the actual test set is released
        vtb_split.split_files(tmp_output_path, paths["CONSTITUENCY_DATA_DIR"], dataset_name, train_size=0.9, dev_size=0.1)
    _, _, test_file = vtb_split.create_paths(paths["CONSTITUENCY_DATA_DIR"], dataset_name)
    with open(test_file, "w"):
        # create an empty test file - currently we don't have actual test data for VLSP 21
        pass


def process_vlsp22(paths, dataset_name, *args):
    """
    Processes the VLSP 2022 dataset, which is four separate files for some reason
    """
    assert dataset_name == 'vi_vlsp22'

    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', default='VLSP_2022', type=str, help='Where to find the data - allows for using previous versions, if needed')
    parser.add_argument('--no_convert_brackets', default=True, action='store_false', dest='convert_brackets', help="Don't convert the VLSP parens RKBT & LKBT to PTB parens")
    parser.add_argument('--n_splits', default=None, type=int, help='Split the data into this many pieces.  Relevant as there is no set training/dev split and no official test data yet, so this allows for N models on N different dev sets')
    parser.add_argument('--test_split', default=False, action='store_true', help='Split 1/10th of the data as a test split as well.  Useful for experimental results')
    args = parser.parse_args(args=list(*args))

    if os.path.exists(args.subdir):
        vlsp_dir = args.subdir
    else:
        vlsp_dir = os.path.join(paths["CONSTITUENCY_BASE"], "vietnamese", args.subdir)
    if not os.path.exists(vlsp_dir):
        raise FileNotFoundError("Could not find the 2022 dataset in the expected location of {} - CONSTITUENCY_BASE == {}".format(vlsp_dir, paths["CONSTITUENCY_BASE"]))
    vlsp_files = os.listdir(vlsp_dir)
    vlsp_files = [os.path.join(vlsp_dir, x) for x in vlsp_files if not x.endswith(".zip")]
    vlsp_files.sort()
    if len(vlsp_files) == 0:
        raise FileNotFoundError("No tree files found in {}".format(vlsp_dir))
    print("Loaded files from {}".format(vlsp_dir))
    print("Procesing:\n  {}".format("\n  ".join(vlsp_files)))
    with tempfile.TemporaryDirectory() as tmp_output_path:
        vtb_convert.convert_files(vlsp_files, tmp_output_path, verbose=True, fix_errors=True, convert_brackets=args.convert_brackets)
        # This produces a 0 length test set, just as a placeholder until the actual test set is released
        if args.n_splits:
            test_size = 0.1 if args.test_split else 0.0
            dev_size = (1.0 - test_size) / args.n_splits
            train_size = 1.0 - test_size - dev_size
            for rotation in range(args.n_splits):
                # there is a shuffle inside the split routine,
                # so we need to reset the random seed each time
                random.seed(1234)
                rotation_name = "%s-%d-%d" % (dataset_name, rotation, args.n_splits)
                if args.test_split:
                    rotation_name = rotation_name + "t"
                vtb_split.split_files(tmp_output_path, paths["CONSTITUENCY_DATA_DIR"], rotation_name, train_size=train_size, dev_size=dev_size, rotation=(rotation, args.n_splits))
                _, _, test_file = vtb_split.create_paths(paths["CONSTITUENCY_DATA_DIR"], rotation_name)
                if not os.path.exists(test_file):
                    with open(test_file, "w"):
                        # create an empty test file - currently we don't have actual test data for VLSP 22
                        pass
        else:
            test_size = 0.1 if args.test_split else 0.0
            dev_size = 0.1
            train_size = 1.0 - test_size - dev_size
            if args.test_split:
                dataset_name = dataset_name + "t"
            vtb_split.split_files(tmp_output_path, paths["CONSTITUENCY_DATA_DIR"], dataset_name, train_size=train_size, dev_size=dev_size)
            _, _, test_file = vtb_split.create_paths(paths["CONSTITUENCY_DATA_DIR"], dataset_name)
            if not os.path.exists(test_file):
                with open(test_file, "w"):
                    # create an empty test file - currently we don't have actual test data for VLSP 22
                    pass

def process_arboretum(paths, dataset_name, *args):
    """
    Processes the Danish dataset, Arboretum
    """
    assert dataset_name == 'da_arboretum'

    arboretum_file = os.path.join(paths["CONSTITUENCY_BASE"], "danish", "arboretum", "arboretum.tiger", "arboretum.tiger")
    if not os.path.exists(arboretum_file):
        raise FileNotFoundError("Unable to find input file for Arboretum.  Expected in {}".format(arboretum_file))

    treebank = convert_tiger_treebank(arboretum_file)
    datasets = utils.split_treebank(treebank, 0.8, 0.1)
    output_dir = paths["CONSTITUENCY_DATA_DIR"]

    output_filename = os.path.join(output_dir, "%s.mrg" % dataset_name)
    print("Writing {} trees to {}".format(len(treebank), output_filename))
    parse_tree.Tree.write_treebank(treebank, output_filename)

    write_dataset(datasets, output_dir, dataset_name)


def process_starlang(paths, dataset_name, *args):
    """
    Convert the Turkish Starlang dataset to brackets
    """
    assert dataset_name == 'tr_starlang'

    PIECES = ["TurkishAnnotatedTreeBank-15",
              "TurkishAnnotatedTreeBank2-15",
              "TurkishAnnotatedTreeBank2-20"]

    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    chunk_paths = [os.path.join(paths["CONSTITUENCY_BASE"], "turkish", piece) for piece in PIECES]
    datasets = read_starlang(chunk_paths)

    write_dataset(datasets, output_dir, dataset_name)

def process_ja_alt(paths, dataset_name, *args):
    """
    Convert and split the ALT dataset

    TODO: could theoretically extend this to MY or any other similar dataset from ALT
    """
    lang, source = dataset_name.split("_", 1)
    assert lang == 'ja'
    assert source == 'alt'

    PIECES = ["Japanese-ALT-Draft.txt", "Japanese-ALT-Reviewed.txt"]
    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "japanese", "Japanese-ALT-20210218")
    input_files = [os.path.join(input_dir, input_file) for input_file in PIECES]
    split_files = [os.path.join(input_dir, "URL-%s.txt" % shard) for shard in SHARDS]
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    output_files = [os.path.join(output_dir, "%s_%s.mrg" % (dataset_name, shard)) for shard in SHARDS]
    convert_alt(input_files, split_files, output_files)

def process_pt_cintil(paths, dataset_name, *args):
    """
    Convert and split the PT Cintil dataset
    """
    lang, source = dataset_name.split("_", 1)
    assert lang == 'pt'
    assert source == 'cintil'

    input_file = os.path.join(paths["CONSTITUENCY_BASE"], "portuguese", "CINTIL", "CINTIL-Treebank.xml")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    datasets = convert_cintil_treebank(input_file)

    write_dataset(datasets, output_dir, dataset_name)

DATASET_MAPPING = {
    'da_arboretum': process_arboretum,

    'it_turin':     process_it_turin,
    'it_vit':       process_it_vit,

    'ja_alt':       process_ja_alt,

    'pt_cintil':    process_pt_cintil,

    'tr_starlang':  process_starlang,

    'vi_vlsp09':    process_vlsp09,
    'vi_vlsp21':    process_vlsp21,
    'vi_vlsp22':    process_vlsp22,
}

def main(dataset_name, *args):
    paths = default_paths.get_default_paths()

    random.seed(1234)

    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name, *args)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_con_dataset")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])


