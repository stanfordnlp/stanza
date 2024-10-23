"""Converts raw data files from their original format (dataset dependent) into PTB trees.

The operation of this script depends heavily on the dataset in question.
The common result is that the data files go to data/constituency and are in PTB format.

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

en_ptb3-revised is an updated version of PTB with NML and stuff
  put LDC2015T13 in $CONSTITUENCY_BASE/english
  the directory name may look like LDC2015T13_eng_news_txt_tbnk-ptb_revised
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset en_ptb3-revised

  All this needs to do is concatenate the various pieces

  @article{ptb_revised,
    title= {Penn Treebank Revised: English News Text Treebank LDC2015T13},
    journal= {},
    author= {Ann Bies and Justin Mott and Colin Warner},
    year= {2015},
    url= {https://doi.org/10.35111/xpjy-at91},
    doi= {10.35111/xpjy-at91},
    isbn= {1-58563-724-6},
    dcmi= {text},
    languages= {english},
    language= {english},
    ldc= {LDC2015T13},
  }

id_icon
  ICON: Building a Large-Scale Benchmark Constituency Treebank
    for the Indonesian Language
    Ee Suan Lim, Wei Qi Leong, Ngan Thanh Nguyen, Dea Adhista,
    Wei Ming Kng, William Chandra Tjhi, Ayu Purwarianti
    https://aclanthology.org/2023.tlt-1.5.pdf
  Available at https://github.com/aisingapore/seacorenlp-data
  git clone the repo in $CONSTITUENCY_BASE/seacorenlp
  so there is now a directory
    $CONSTITUENCY_BASE/seacorenlp/seacorenlp-data
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset id_icon

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

vlsp09 is the 2009 constituency treebank:
  Nguyen Phuong Thai, Vu Xuan Luong, Nguyen Thi Minh Huyen, Nguyen Van Hiep, Le Hong Phuong
    Building a Large Syntactically-Annotated Corpus of Vietnamese
    Proceedings of The Third Linguistic Annotation Workshop
    In conjunction with ACL-IJCNLP 2009, Suntec City, Singapore, 2009
  This can be obtained by contacting vlsp.resources@gmail.com

vlsp22 is the 2022 constituency treebank from the VLSP bakeoff
  there is an official test set as well
  you may be able to obtain both of these by contacting vlsp.resources@gmail.com
  NGUYEN Thi Minh Huyen, HA My Linh, VU Xuan Luong, PHAN Thi Hue,
  LE Van Cuong, NGUYEN Thi Luong, NGO The Quyen
    VLSP 2022 Challenge: Vietnamese Constituency Parsing
    to appear in Journal of Computer Science and Cybernetics.

vlsp23 is the 2023 update to the constituency treebank from the VLSP bakeoff
  the vlsp22 code also works for the new dataset,
    although some effort may be needed to update the tags
  As of late 2024, the test set is available on request at vlsp.resources@gmail.com
  Organize the directory
    $CONSTITUENCY_BASE/vietnamese/VLSP_2023
      $CONSTITUENCY_BASE/vietnamese/VLSP_2023/Trainingset
      $CONSTITUENCY_BASE/vietnamese/VLSP_2023/test

zh_ctb-51 is the 5.1 version of CTB
  put LDC2005T01U01_ChineseTreebank5.1 in $CONSTITUENCY_BASE/chinese
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset zh_ctb-51

  @article{xue_xia_chiou_palmer_2005,
           title={The Penn Chinese TreeBank: Phrase structure annotation of a large corpus},
           volume={11},
           DOI={10.1017/S135132490400364X},
           number={2},
           journal={Natural Language Engineering},
           publisher={Cambridge University Press},
           author={XUE, NAIWEN and XIA, FEI and CHIOU, FU-DONG and PALMER, MARTA},
           year={2005},
           pages={207–238}}

zh_ctb-51b is the same dataset, but using a smaller dev/test set
  in our experiments, this is substantially easier

zh_ctb-90 is the 9.0 version of CTB
  put LDC2016T13 in $CONSTITUENCY_BASE/chinese
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset zh_ctb-90

  the splits used are the ones from the file docs/ctb9.0-file-list.txt
    included in the CTB 9.0 release

SPMRL adds several treebanks
  https://www.spmrl.org/
  https://www.spmrl.org/sancl-posters2014.html
  Currently only German is converted, the German version being a
    version of the Tiger Treebank
  python3 -m stanza.utils.datasets.constituency.prepare_con_dataset de_spmrl  

en_mctb is a multidomain test set covering five domains other than newswire
  https://github.com/RingoS/multi-domain-parsing-analysis
  Challenges to Open-Domain Constituency Parsing

  @inproceedings{yang-etal-2022-challenges,
    title = "Challenges to Open-Domain Constituency Parsing",
    author = "Yang, Sen  and
      Cui, Leyang and
      Ning, Ruoxi and
      Wu, Di and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.11",
    doi = "10.18653/v1/2022.findings-acl.11",
    pages = "112--127",
  }

  This conversion replaces the top bracket from top -> ROOT and puts an extra S
    bracket on any roots with more than one node.
"""

import argparse
import os
import random
import sys
import tempfile

from tqdm import tqdm

from stanza.models.constituency import parse_tree
import stanza.utils.default_paths as default_paths
from stanza.models.constituency import tree_reader
from stanza.models.constituency.parse_tree import Tree
from stanza.server import tsurgeon
from stanza.utils.datasets.common import UnknownDatasetError
from stanza.utils.datasets.constituency import utils
from stanza.utils.datasets.constituency.convert_alt import convert_alt
from stanza.utils.datasets.constituency.convert_arboretum import convert_tiger_treebank
from stanza.utils.datasets.constituency.convert_cintil import convert_cintil_treebank
import stanza.utils.datasets.constituency.convert_ctb as convert_ctb
from stanza.utils.datasets.constituency.convert_it_turin import convert_it_turin
from stanza.utils.datasets.constituency.convert_it_vit import convert_it_vit
from stanza.utils.datasets.constituency.convert_spmrl import convert_spmrl
from stanza.utils.datasets.constituency.convert_starlang import read_starlang
from stanza.utils.datasets.constituency.utils import SHARDS, write_dataset
import stanza.utils.datasets.constituency.vtb_convert as vtb_convert
import stanza.utils.datasets.constituency.vtb_split as vtb_split

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
    assert dataset_name == 'vi_vlsp22' or dataset_name == 'vi_vlsp23'

    if dataset_name == 'vi_vlsp22':
        default_subdir = 'VLSP_2022'
        default_make_test_split = False
        updated_tagset = False
    elif dataset_name == 'vi_vlsp23':
        default_subdir = os.path.join('VLSP_2023', 'Trainingdataset')
        default_make_test_split = False
        updated_tagset = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', default=default_subdir, type=str, help='Where to find the data - allows for using previous versions, if needed')
    parser.add_argument('--no_convert_brackets', default=True, action='store_false', dest='convert_brackets', help="Don't convert the VLSP parens RKBT & LKBT to PTB parens")
    parser.add_argument('--n_splits', default=None, type=int, help='Split the data into this many pieces.  Relevant as there is no set training/dev split, so this allows for N models on N different dev sets')
    parser.add_argument('--test_split', default=default_make_test_split, action='store_true', help='Split 1/10th of the data as a test split as well.  Useful for experimental results.  Less relevant since there is now an official test set')
    parser.add_argument('--no_test_split', dest='test_split', action='store_false', help='Split 1/10th of the data as a test split as well.  Useful for experimental results.  Less relevant since there is now an official test set')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed to use when splitting')
    args = parser.parse_args(args=list(*args))

    if os.path.exists(args.subdir):
        vlsp_dir = args.subdir
    else:
        vlsp_dir = os.path.join(paths["CONSTITUENCY_BASE"], "vietnamese", args.subdir)
    if not os.path.exists(vlsp_dir):
        raise FileNotFoundError("Could not find the {} dataset in the expected location of {} - CONSTITUENCY_BASE == {}".format(dataset_name, vlsp_dir, paths["CONSTITUENCY_BASE"]))
    vlsp_files = os.listdir(vlsp_dir)
    vlsp_train_files = [os.path.join(vlsp_dir, x) for x in vlsp_files if x.startswith("file") and not x.endswith(".zip")]
    vlsp_train_files.sort()
        
    if dataset_name == 'vi_vlsp22':
        vlsp_test_files = [os.path.join(vlsp_dir, x) for x in vlsp_files if x.startswith("private") and not x.endswith(".zip")]
    elif dataset_name == 'vi_vlsp23':
        vlsp_test_dir = os.path.abspath(os.path.join(vlsp_dir, os.pardir, "test"))
        vlsp_test_files = os.listdir(vlsp_test_dir)
        vlsp_test_files = [os.path.join(vlsp_test_dir, x) for x in vlsp_test_files if x.endswith(".csv")]

    if len(vlsp_train_files) == 0:
        raise FileNotFoundError("No train files (files starting with 'file') found in {}".format(vlsp_dir))
    if not args.test_split and len(vlsp_test_files) == 0:
        raise FileNotFoundError("No test files found in {}".format(vlsp_dir))
    print("Loading training files from {}".format(vlsp_dir))
    print("Procesing training files:\n  {}".format("\n  ".join(vlsp_train_files)))
    with tempfile.TemporaryDirectory() as train_output_path:
        vtb_convert.convert_files(vlsp_train_files, train_output_path, verbose=True, fix_errors=True, convert_brackets=args.convert_brackets, updated_tagset=updated_tagset)
        # This produces a 0 length test set, just as a placeholder until the actual test set is released
        if args.n_splits:
            test_size = 0.1 if args.test_split else 0.0
            dev_size = (1.0 - test_size) / args.n_splits
            train_size = 1.0 - test_size - dev_size
            for rotation in range(args.n_splits):
                # there is a shuffle inside the split routine,
                # so we need to reset the random seed each time
                random.seed(args.seed)
                rotation_name = "%s-%d-%d" % (dataset_name, rotation, args.n_splits)
                if args.test_split:
                    rotation_name = rotation_name + "t"
                vtb_split.split_files(train_output_path, paths["CONSTITUENCY_DATA_DIR"], rotation_name, train_size=train_size, dev_size=dev_size, rotation=(rotation, args.n_splits))
        else:
            test_size = 0.1 if args.test_split else 0.0
            dev_size = 0.1
            train_size = 1.0 - test_size - dev_size
            if args.test_split:
                dataset_name = dataset_name + "t"
            vtb_split.split_files(train_output_path, paths["CONSTITUENCY_DATA_DIR"], dataset_name, train_size=train_size, dev_size=dev_size)

    if not args.test_split:
        print("Procesing test files:\n  {}".format("\n  ".join(vlsp_test_files)))
        with tempfile.TemporaryDirectory() as test_output_path:
            vtb_convert.convert_files(vlsp_test_files, test_output_path, verbose=True, fix_errors=True, convert_brackets=args.convert_brackets, updated_tagset=updated_tagset)
            if args.n_splits:
                for rotation in range(args.n_splits):
                    rotation_name = "%s-%d-%d" % (dataset_name, rotation, args.n_splits)
                    vtb_split.split_files(test_output_path, paths["CONSTITUENCY_DATA_DIR"], rotation_name, train_size=0, dev_size=0)
            else:
                vtb_split.split_files(test_output_path, paths["CONSTITUENCY_DATA_DIR"], dataset_name, train_size=0, dev_size=0)
    if not args.test_split and not args.n_splits and dataset_name == 'vi_vlsp23':
        print("Procesing test files and keeping ids:\n  {}".format("\n  ".join(vlsp_test_files)))
        with tempfile.TemporaryDirectory() as test_output_path:
            vtb_convert.convert_files(vlsp_test_files, test_output_path, verbose=True, fix_errors=True, convert_brackets=args.convert_brackets, updated_tagset=updated_tagset, write_ids=True)
            vtb_split.split_files(test_output_path, paths["CONSTITUENCY_DATA_DIR"], dataset_name + "-ids", train_size=0, dev_size=0)

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

def process_id_icon(paths, dataset_name, *args):
    lang, source = dataset_name.split("_", 1)
    assert lang == 'id'
    assert source == 'icon'

    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "seacorenlp", "seacorenlp-data", "id", "constituency")
    input_files = [os.path.join(input_dir, x) for x in ("train.txt", "dev.txt", "test.txt")]
    datasets = []
    for input_file in input_files:
        trees = tree_reader.read_tree_file(input_file)
        trees = [Tree("ROOT", tree) for tree in trees]
        datasets.append(trees)

    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    write_dataset(datasets, output_dir, dataset_name)

def process_ctb_51(paths, dataset_name, *args):
    lang, source = dataset_name.split("_", 1)
    assert lang == 'zh-hans'
    assert source == 'ctb-51'

    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "chinese", "LDC2005T01U01_ChineseTreebank5.1", "bracketed")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    convert_ctb.convert_ctb(input_dir, output_dir, dataset_name, convert_ctb.Version.V51)

def process_ctb_51b(paths, dataset_name, *args):
    lang, source = dataset_name.split("_", 1)
    assert lang == 'zh-hans'
    assert source == 'ctb-51b'

    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "chinese", "LDC2005T01U01_ChineseTreebank5.1", "bracketed")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    if not os.path.exists(input_dir):
        raise FileNotFoundError("CTB 5.1 location not found: %s" % input_dir)
    print("Loading trees from %s" % input_dir)
    convert_ctb.convert_ctb(input_dir, output_dir, dataset_name, convert_ctb.Version.V51b)

def process_ctb_90(paths, dataset_name, *args):
    lang, source = dataset_name.split("_", 1)
    assert lang == 'zh-hans'
    assert source == 'ctb-90'

    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "chinese", "LDC2016T13", "ctb9.0", "data", "bracketed")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]
    convert_ctb.convert_ctb(input_dir, output_dir, dataset_name, convert_ctb.Version.V90)


def process_ptb3_revised(paths, dataset_name, *args):
    input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "english", "LDC2015T13_eng_news_txt_tbnk-ptb_revised")
    if not os.path.exists(input_dir):
        backup_input_dir = os.path.join(paths["CONSTITUENCY_BASE"], "english", "LDC2015T13")
        if not os.path.exists(backup_input_dir):
            raise FileNotFoundError("Could not find ptb3-revised in either %s or %s" % (input_dir, backup_input_dir))
        input_dir = backup_input_dir

    bracket_dir = os.path.join(input_dir, "data", "penntree")
    output_dir = paths["CONSTITUENCY_DATA_DIR"]

    # compensate for a weird mislabeling in the original dataset
    label_map = {"ADJ-PRD": "ADJP-PRD"}

    train_trees = []
    for i in tqdm(range(2, 22)):
        new_trees = tree_reader.read_directory(os.path.join(bracket_dir, "%02d" % i))
        new_trees = [t.remap_constituent_labels(label_map) for t in new_trees]
        train_trees.extend(new_trees)

    move_tregex = "_ROOT_ <1 __=home <2 /^[.]$/=move"
    move_tsurgeon = "move move >-1 home"

    print("Moving sentence final punctuation if necessary")
    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        train_trees = [tsurgeon_processor.process(tree, move_tregex, move_tsurgeon)[0] for tree in tqdm(train_trees)]

    dev_trees = tree_reader.read_directory(os.path.join(bracket_dir, "22"))
    dev_trees = [t.remap_constituent_labels(label_map) for t in dev_trees]

    test_trees = tree_reader.read_directory(os.path.join(bracket_dir, "23"))
    test_trees = [t.remap_constituent_labels(label_map) for t in test_trees]
    print("Read %d train trees, %d dev trees, and %d test trees" % (len(train_trees), len(dev_trees), len(test_trees)))
    datasets = [train_trees, dev_trees, test_trees]
    write_dataset(datasets, output_dir, dataset_name)

def process_en_mctb(paths, dataset_name, *args):
    """
    Converts the following blocks:

    dialogue.cleaned.txt  forum.cleaned.txt  law.cleaned.txt  literature.cleaned.txt  review.cleaned.txt
    """
    base_path = os.path.join(paths["CONSTITUENCY_BASE"], "english", "multi-domain-parsing-analysis", "data", "MCTB_en")
    if not os.path.exists(base_path):
        raise FileNotFoundError("Please download multi-domain-parsing-analysis to %s" % base_path)
    def tree_callback(tree):
        if len(tree.children) > 1:
            tree = parse_tree.Tree("S", tree.children)
            return parse_tree.Tree("ROOT", [tree])
        return parse_tree.Tree("ROOT", tree.children)

    filenames = ["dialogue.cleaned.txt", "forum.cleaned.txt", "law.cleaned.txt", "literature.cleaned.txt", "review.cleaned.txt"]
    for filename in filenames:
        trees = tree_reader.read_tree_file(os.path.join(base_path, filename), tree_callback=tree_callback)
        print("%d trees in %s" % (len(trees), filename))
        output_filename = "%s-%s_test.mrg" % (dataset_name, filename.split(".")[0])
        output_filename = os.path.join(paths["CONSTITUENCY_DATA_DIR"], output_filename)
        print("Writing trees to %s" % output_filename)
        parse_tree.Tree.write_treebank(trees, output_filename)

def process_spmrl(paths, dataset_name, *args):
    if dataset_name != 'de_spmrl':
        raise ValueError("SPMRL dataset %s currently not supported" % dataset_name)

    output_directory = paths["CONSTITUENCY_DATA_DIR"]
    input_directory = os.path.join(paths["CONSTITUENCY_BASE"], "spmrl", "SPMRL_SHARED_2014", "GERMAN_SPMRL", "gold", "ptb")

    convert_spmrl(input_directory, output_directory, dataset_name)

DATASET_MAPPING = {
    'da_arboretum': process_arboretum,

    'de_spmrl':     process_spmrl,

    'en_ptb3-revised': process_ptb3_revised,
    'en_mctb':      process_en_mctb,

    'id_icon':      process_id_icon,

    'it_turin':     process_it_turin,
    'it_vit':       process_it_vit,

    'ja_alt':       process_ja_alt,

    'pt_cintil':    process_pt_cintil,

    'tr_starlang':  process_starlang,

    'vi_vlsp09':    process_vlsp09,
    'vi_vlsp21':    process_vlsp21,
    'vi_vlsp22':    process_vlsp22,
    'vi_vlsp23':    process_vlsp22,  # options allow for this

    'zh-hans_ctb-51':   process_ctb_51,
    'zh-hans_ctb-51b':  process_ctb_51b,
    'zh-hans_ctb-90':   process_ctb_90,
}

def main(dataset_name, *args):
    paths = default_paths.get_default_paths()

    random.seed(1234)

    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name, *args)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_con_dataset")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Known datasets:")
        for key in DATASET_MAPPING:
            print("  %s" % key)
    else:
        main(sys.argv[1], sys.argv[2:])


