"""Converts raw data files into json files usable by the training script.

Currently it supports converting wikiner datasets, available here:
  https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500
  - download the language of interest to {Language}-WikiNER
  - then run
    prepare_ner_dataset.py French-WikiNER

Also, Finnish Turku dataset, available here:
  - https://turkunlp.org/fin-ner.html
  - Download and unzip the corpus, putting the .tsv files into
    $NERBASE/fi_turku
  - prepare_ner_dataset.py fi_turku

FBK in Italy produced an Italian dataset.
  The processing here is for a combined .tsv file they sent us.
  - prepare_ner_dataset.py it_fbk

IJCNLP 2008 produced a few Indian language NER datasets.
  description:
    http://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=3
  download:
    http://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=5
  The models produced from these datasets have extremely low recall, unfortunately.
  - prepare_ner_dataset.py hi_ijc

FIRE 2013 also produced NER datasets for Indian languages.
  http://au-kbc.org/nlp/NER-FIRE2013/index.html
  The datasets are password locked.
  For Stanford users, contact Chris Manning for license details.
  For external users, please contact the organizers for more information.
  - prepare_ner_dataset.py hi-fire2013

Ukranian NER is provided by lang-uk, available here:
  https://github.com/lang-uk/ner-uk
  git clone the repo to $NERBASE/lang-uk
  There should be a subdirectory $NERBASE/lang-uk/ner-uk/data at that point
  Conversion script graciously provided by Andrii Garkavyi @gawy
  - prepare_ner_dataset.py uk_languk

There are two Hungarian datasets are available here:
  https://rgai.inf.u-szeged.hu/node/130
  http://www.lrec-conf.org/proceedings/lrec2006/pdf/365_pdf.pdf
  We combined them and give them the label hu_rgai
  You can also build individual pieces with hu_rgai_business or hu_rgai_criminal
  Create a subdirectory of $NERBASE, $NERBASE/hu_rgai, and download both of
    the pieces and unzip them in that directory.
  - prepare_ner_dataset.py hu_rgai

Another Hungarian dataset is here:
  - https://github.com/nytud/NYTK-NerKor
  - git clone the entire thing in your $NERBASE directory to operate on it
  - prepare_ner_dataset.py hu_nytk

The two Hungarian datasets can be combined with hu_combined
  TODO: verify that there is no overlap in text
  - prepare_ner_dataset.py hu_combined

BSNLP publishes NER datasets for Eastern European languages.
  - In 2019 they published BG, CS, PL, RU.
  - http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html
  - In 2021 they added some more data, but the test sets
    were not publicly available as of April 2021.
    Therefore, currently the model is made from 2019.
    In 2021, the link to the 2021 task is here:
    http://bsnlp.cs.helsinki.fi/shared-task.html
  - The below method processes the 2019 version of the corpus.
    It has specific adjustments for the BG section, which has
    quite a few typos or mis-annotations in it.  Other languages
    probably need similar work in order to function optimally.
  - make a directory $NERBASE/bsnlp2019
  - download the "training data are available HERE" and
    "test data are available HERE" to this subdirectory
  - unzip those files in that directory
  - we use the code name "bg_bsnlp19".  Other languages from
    bsnlp 2019 can be supported by adding the appropriate
    functionality in convert_bsnlp.py.
  - prepare_ner_dataset.py bg_bsnlp19

NCHLT produced NER datasets for many African languages.
  Unfortunately, it is difficult to make use of many of these,
  as there is no corresponding UD data from which to build a
  tokenizer or other tools.
  - Afrikaans:  https://repo.sadilar.org/handle/20.500.12185/299
  - isiNdebele: https://repo.sadilar.org/handle/20.500.12185/306
  - isiXhosa:   https://repo.sadilar.org/handle/20.500.12185/312
  - isiZulu:    https://repo.sadilar.org/handle/20.500.12185/319
  - Sepedi:     https://repo.sadilar.org/handle/20.500.12185/328
  - Sesotho:    https://repo.sadilar.org/handle/20.500.12185/334
  - Setswana:   https://repo.sadilar.org/handle/20.500.12185/341
  - Siswati:    https://repo.sadilar.org/handle/20.500.12185/346
  - Tsivenda:   https://repo.sadilar.org/handle/20.500.12185/355
  - Xitsonga:   https://repo.sadilar.org/handle/20.500.12185/362
  Agree to the license, download the zip, and unzip it in
  $NERBASE/NCHLT

UCSY built a Myanmar dataset.  They have not made it publicly
  available, but they did make it available to Stanford for research
  purposes.  Contact Chris Manning or John Bauer for the data files if
  you are Stanford affiliated.
  - https://arxiv.org/abs/1903.04739
  - Syllable-based Neural Named Entity Recognition for Myanmar Language
    by Hsu Myat Mo and Khin Mar Soe

Hanieh Poostchi et al produced a Persian NER dataset:
  - git@github.com:HaniehP/PersianNER.git
  - https://github.com/HaniehP/PersianNER
  - Hanieh Poostchi, Ehsan Zare Borzeshi, Mohammad Abdous, and Massimo Piccardi,
    "PersoNER: Persian Named-Entity Recognition"
  - Hanieh Poostchi, Ehsan Zare Borzeshi, and Massimo Piccardi,
    "BiLSTM-CRF for Persian Named-Entity Recognition; ArmanPersoNERCorpus: the First Entity-Annotated Persian Dataset"
  - Conveniently, this dataset is already in BIO format.  It does not have a dev split, though.
    git clone the above repo, unzip ArmanPersoNERCorpus.zip, and this script will split the
    first train fold into a dev section.

SUC3 is a Swedish NER dataset provided by Språkbanken
  - https://spraakbanken.gu.se/en/resources/suc3
  - The splitting tool is generously provided by
    Emil Stenstrom
    https://github.com/EmilStenstrom/suc_to_iob
  - Download the .bz2 file at this URL and put it in $NERBASE/sv_suc3shuffle
    It is not necessary to unzip it.
  - Gustafson-Capková, Sophia and Britt Hartmann, 2006, 
    Manual of the Stockholm Umeå Corpus version 2.0.
    Stockholm University.
  - Östling, Robert, 2013, Stagger 
    an Open-Source Part of Speech Tagger for Swedish
    Northern European Journal of Language Technology 3: 1–18
    DOI 10.3384/nejlt.2000-1533.1331
  - The shuffled dataset can be converted with dataset code
    prepare_ner_dataset.py sv_suc3shuffle
  - If you fill out the license form and get the official data,
    you can get the official splits by putting the provided zip file
    in $NERBASE/sv_suc3licensed.  Again, not necessary to unzip it
    prepare_ner_dataset.py sv_suc3licensed

DDT is a reformulation of the Danish Dependency Treebank as an NER dataset
  - https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dane
  - direct download link as of late 2021: https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip
  - https://aclanthology.org/2020.lrec-1.565.pdf
    DaNE: A Named Entity Resource for Danish
    Rasmus Hvingelby, Amalie Brogaard Pauli, Maria Barrett,
    Christina Rosted, Lasse Malm Lidegaard, Anders Søgaard
  - place ddt.zip in $NERBASE/da_ddt/ddt.zip
    prepare_ner_dataset.py da_ddt

NorNE is the Norwegian Dependency Treebank with NER labels
  - LREC 2020
    NorNE: Annotating Named Entities for Norwegian
    Fredrik Jørgensen, Tobias Aasmoe, Anne-Stine Ruud Husevåg,
    Lilja Øvrelid, and Erik Velldal
  - both Bokmål and Nynorsk
  - This dataset is in a git repo:
    https://github.com/ltgoslo/norne
    Clone it into $NERBASE
    git clone git@github.com:ltgoslo/norne.git
    prepare_ner_dataset.py nb_norne
    prepare_ner_dataset.py nn_norne

starlang is a set of constituency trees for Turkish
  The words in this dataset (usually) have NER labels as well

  A dataset in three parts from the Starlang group in Turkey:
  Neslihan Kara, Büşra Marşan, et al
    Creating A Syntactically Felicitous Constituency Treebank For Turkish
    https://ieeexplore.ieee.org/document/9259873
  git clone the following three repos
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank-15
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank2-15
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank2-20
  Put them in
    $CONSTITUENCY_HOME/turkish    (yes, the constituency home)
  prepare_ner_dataset.py tr_starlang

en_sample is the toy dataset included with stanza-train
  https://github.com/stanfordnlp/stanza-train
  this is not meant for any kind of actual NER use
"""

import glob
import os
import random
import shutil
import sys
import tempfile

from stanza.models.common.constant import treebank_to_short_name, lcode2lang
import stanza.utils.default_paths as default_paths

from stanza.utils.datasets.ner.preprocess_wikiner import preprocess_wikiner
from stanza.utils.datasets.ner.split_wikiner import split_wikiner
import stanza.utils.datasets.ner.conll_to_iob as conll_to_iob
import stanza.utils.datasets.ner.convert_bsf_to_beios as convert_bsf_to_beios
import stanza.utils.datasets.ner.convert_bsnlp as convert_bsnlp
import stanza.utils.datasets.ner.convert_fire_2013 as convert_fire_2013
import stanza.utils.datasets.ner.convert_ijc as convert_ijc
import stanza.utils.datasets.ner.convert_my_ucsy as convert_my_ucsy
import stanza.utils.datasets.ner.convert_rgai as convert_rgai
import stanza.utils.datasets.ner.convert_nytk as convert_nytk
import stanza.utils.datasets.ner.convert_starlang_ner as convert_starlang_ner
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file
import stanza.utils.datasets.ner.suc_to_iob as suc_to_iob
import stanza.utils.datasets.ner.suc_conll_to_iob as suc_conll_to_iob

SHARDS = ('train', 'dev', 'test')

class UnknownDatasetError(ValueError):
    def __init__(self, dataset, text):
        super().__init__(text)
        self.dataset = dataset

def convert_bio_to_json(base_input_path, base_output_path, short_name, suffix="bio"):
    """
    Convert BIO files to json

    It can often be convenient to put the intermediate BIO files in
    the same directory as the output files, in which case you can pass
    in same path for both base_input_path and base_output_path.
    """
    for shard in SHARDS:
        input_filename = os.path.join(base_input_path, '%s.%s.%s' % (short_name, shard, suffix))
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def write_dataset(datasets, output_dir, short_name, suffix="bio"):
    for shard, dataset in zip(SHARDS, datasets):
        output_filename = os.path.join(output_dir, "%s.%s.%s" % (short_name, shard, suffix))
        with open(output_filename, "w", encoding="utf-8") as fout:
            for sentence in dataset:
                for word in sentence:
                    fout.write("%s\t%s\n" % word)
                fout.write("\n")

    convert_bio_to_json(output_dir, output_dir, short_name, suffix)

def process_turku(paths):
    short_name = 'fi_turku'
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]
    for shard in SHARDS:
        input_filename = os.path.join(base_input_path, '%s.tsv' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def process_it_fbk(paths):
    short_name = "it_fbk"
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    csv_file = os.path.join(base_input_path, "all-wiki-split.tsv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError("Cannot find the FBK dataset in its expected location: {}".format(csv_file))
    base_output_path = paths["NER_DATA_DIR"]
    split_wikiner(base_output_path, csv_file, prefix=short_name, suffix="io", shuffle=False, train_fraction=0.8, dev_fraction=0.1)
    convert_bio_to_json(base_output_path, base_output_path, short_name, suffix="io")


def process_languk(paths):
    short_name = 'uk_languk'
    base_input_path = os.path.join(paths["NERBASE"], 'lang-uk', 'ner-uk', 'data')
    base_output_path = paths["NER_DATA_DIR"]
    train_test_split_fname = os.path.join(paths["NERBASE"], 'lang-uk', 'ner-uk', 'doc', 'dev-test-split.txt')
    convert_bsf_to_beios.convert_bsf_in_folder(base_input_path, base_output_path, train_test_split_file=train_test_split_fname)
    for shard in SHARDS:
        input_filename = os.path.join(base_output_path, convert_bsf_to_beios.CORPUS_NAME, "%s.bio" % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)


def process_ijc(paths, short_name):
    """
    Splits the ijc Hindi dataset in train, dev, test

    The original data had train & test splits, so we randomly divide
    the files in train to make a dev set.

    The expected location of the IJC data is hi_ijc.  This method
    should be possible to use for other languages, but we have very
    little support for the other languages of IJC at the moment.
    """
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]

    test_files = [os.path.join(base_input_path, "test-data-hindi.txt")]
    test_csv_file = os.path.join(base_output_path, short_name + ".test.csv")
    print("Converting test input %s to space separated file in %s" % (test_files[0], test_csv_file))
    convert_ijc.convert_ijc(test_files, test_csv_file)

    train_input_path = os.path.join(base_input_path, "training-hindi", "*utf8")
    train_files = glob.glob(train_input_path)
    train_csv_file = os.path.join(base_output_path, short_name + ".train.csv")
    dev_csv_file = os.path.join(base_output_path, short_name + ".dev.csv")
    print("Converting training input from %s to space separated files in %s and %s" % (train_input_path, train_csv_file, dev_csv_file))
    convert_ijc.convert_split_ijc(train_files, train_csv_file, dev_csv_file)

    for csv_file, shard in zip((train_csv_file, dev_csv_file, test_csv_file), SHARDS):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)


def process_fire_2013(paths, dataset):
    """
    Splits the FIRE 2013 dataset into train, dev, test

    The provided datasets are all mixed together at this point, so it
    is not possible to recreate the original test conditions used in
    the bakeoff
    """
    short_name = treebank_to_short_name(dataset)
    langcode, _ = short_name.split("_")
    short_name = "%s_fire2013" % langcode
    if not langcode in ("hi", "en", "ta", "bn", "mal"):
        raise UnkonwnDatasetError(dataset, "Language %s not one of the FIRE 2013 languages" % langcode)
    language = lcode2lang[langcode].lower()
    
    # for example, FIRE2013/hindi_train
    base_input_path = os.path.join(paths["NERBASE"], "FIRE2013", "%s_train" % language)
    base_output_path = paths["NER_DATA_DIR"]

    train_csv_file = os.path.join(base_output_path, "%s.train.csv" % short_name)
    dev_csv_file   = os.path.join(base_output_path, "%s.dev.csv" % short_name)
    test_csv_file  = os.path.join(base_output_path, "%s.test.csv" % short_name)

    convert_fire_2013.convert_fire_2013(base_input_path, train_csv_file, dev_csv_file, test_csv_file)

    for csv_file, shard in zip((train_csv_file, dev_csv_file, test_csv_file), SHARDS):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)

def process_wikiner(paths, dataset):
    short_name = treebank_to_short_name(dataset)

    base_input_path = os.path.join(paths["NERBASE"], dataset)
    base_output_path = paths["NER_DATA_DIR"]

    raw_input_path = os.path.join(base_input_path, "raw")
    input_files = glob.glob(os.path.join(raw_input_path, "aij-wikiner*"))
    if len(input_files) == 0:
        raise FileNotFoundError("Could not find any raw wikiner files in %s" % raw_input_path)
    elif len(input_files) > 1:
        raise FileNotFoundError("Found too many raw wikiner files in %s: %s" % (raw_input_path, ", ".join(input_files)))

    csv_file = os.path.join(raw_input_path, "csv_" + short_name)
    print("Converting raw input %s to space separated file in %s" % (input_files[0], csv_file))
    preprocess_wikiner(input_files[0], csv_file)

    # this should create train.bio, dev.bio, and test.bio
    print("Splitting %s to %s" % (csv_file, base_input_path))
    split_wikiner(base_input_path, csv_file)

    for shard in SHARDS:
        input_filename = os.path.join(base_input_path, '%s.bio' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def get_rgai_input_path(paths):
    return os.path.join(paths["NERBASE"], "hu_rgai")

def process_rgai(paths, short_name):
    base_output_path = paths["NER_DATA_DIR"]
    base_input_path = get_rgai_input_path(paths)

    if short_name == 'hu_rgai':
        use_business = True
        use_criminal = True
    elif short_name == 'hu_rgai_business':
        use_business = True
        use_criminal = False
    elif short_name == 'hu_rgai_criminal':
        use_business = False
        use_criminal = True
    else:
        raise UnknownDatasetError(short_name, "Unknown subset of hu_rgai data: %s" % short_name)

    convert_rgai.convert_rgai(base_input_path, base_output_path, short_name, use_business, use_criminal)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def get_nytk_input_path(paths):
    return os.path.join(paths["NERBASE"], "NYTK-NerKor")

def process_nytk(paths):
    """
    Process the NYTK dataset
    """
    base_output_path = paths["NER_DATA_DIR"]
    base_input_path = get_nytk_input_path(paths)
    short_name = "hu_nytk"

    convert_nytk.convert_nytk(base_input_path, base_output_path, short_name)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def concat_files(output_file, *input_files):
    input_lines = []
    for input_file in input_files:
        with open(input_file) as fin:
            lines = fin.readlines()
        if not len(lines):
            raise ValueError("Empty input file: %s" % input_file)
        if not lines[-1]:
            lines[-1] = "\n"
        elif lines[-1].strip():
            lines.append("\n")
        input_lines.append(lines)
    with open(output_file, "w") as fout:
        for lines in input_lines:
            for line in lines:
                fout.write(line)


def process_hu_combined(paths):
    base_output_path = paths["NER_DATA_DIR"]
    rgai_input_path = get_rgai_input_path(paths)
    nytk_input_path = get_nytk_input_path(paths)
    short_name = "hu_combined"

    with tempfile.TemporaryDirectory() as tmp_output_path:
        convert_rgai.convert_rgai(rgai_input_path, tmp_output_path, "hu_rgai", True, True)
        convert_nytk.convert_nytk(nytk_input_path, tmp_output_path, "hu_nytk")

        for shard in SHARDS:
            rgai_input = os.path.join(tmp_output_path, "hu_rgai.%s.bio" % shard)
            nytk_input = os.path.join(tmp_output_path, "hu_nytk.%s.bio" % shard)
            output_file = os.path.join(base_output_path, "hu_combined.%s.bio" % shard)
            concat_files(output_file, rgai_input, nytk_input)

    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_bsnlp(paths, short_name):
    """
    Process files downloaded from http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html

    If you download the training and test data zip files and unzip
    them without rearranging in any way, the layout is somewhat weird.
    Training data goes into a specific subdirectory, but the test data
    goes into the top level directory.
    """
    base_input_path = os.path.join(paths["NERBASE"], "bsnlp2019")
    base_train_path = os.path.join(base_input_path, "training_pl_cs_ru_bg_rc1")
    base_test_path = base_input_path

    base_output_path = paths["NER_DATA_DIR"]

    output_train_filename = os.path.join(base_output_path, "%s.train.csv" % short_name)
    output_dev_filename   = os.path.join(base_output_path, "%s.dev.csv" % short_name)
    output_test_filename  = os.path.join(base_output_path, "%s.test.csv" % short_name)

    language = short_name.split("_")[0]

    convert_bsnlp.convert_bsnlp(language, base_test_path, output_test_filename)
    convert_bsnlp.convert_bsnlp(language, base_train_path, output_train_filename, output_dev_filename)

    for shard, csv_file in zip(SHARDS, (output_train_filename, output_dev_filename, output_test_filename)):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)

NCHLT_LANGUAGE_MAP = {
    "af":  "NCHLT Afrikaans Named Entity Annotated Corpus",
    # none of the following have UD datasets as of 2.8.  Until they
    # exist, we assume the language codes NCHTL are sufficient
    "nr":  "NCHLT isiNdebele Named Entity Annotated Corpus",
    "nso": "NCHLT Sepedi Named Entity Annotated Corpus",
    "ss":  "NCHLT Siswati Named Entity Annotated Corpus",
    "st":  "NCHLT Sesotho Named Entity Annotated Corpus",
    "tn":  "NCHLT Setswana Named Entity Annotated Corpus",
    "ts":  "NCHLT Xitsonga Named Entity Annotated Corpus",
    "ve":  "NCHLT Tshivenda Named Entity Annotated Corpus",
    "xh":  "NCHLT isiXhosa Named Entity Annotated Corpus",
    "zu":  "NCHLT isiZulu Named Entity Annotated Corpus",
}

def process_nchlt(paths, short_name):
    language = short_name.split("_")[0]
    if not language in NCHLT_LANGUAGE_MAP:
        raise UnknownDatasetError(short_name, "Language %s not part of NCHLT" % language)
    short_name = "%s_nchlt" % language

    base_input_path = os.path.join(paths["NERBASE"], "NCHLT", NCHLT_LANGUAGE_MAP[language], "*Full.txt")
    input_files = glob.glob(base_input_path)
    if len(input_files) == 0:
        raise FileNotFoundError("Cannot find NCHLT dataset in '%s'  Did you remember to download the file?" % base_input_path)

    if len(input_files) > 1:
        raise ValueError("Unexpected number of files matched '%s'  There should only be one" % base_input_path)

    base_output_path = paths["NER_DATA_DIR"]
    split_wikiner(base_output_path, input_files[0], prefix=short_name, remap={"OUT": "O"})
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_my_ucsy(paths):
    language = "my"
    short_name = "my_ucsy"

    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]
    convert_my_ucsy.convert_my_ucsy(base_input_path, base_output_path)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_fa_arman(paths, short_name):
    """
    Converts fa_arman dataset

    The conversion is quite simple, actually.
    Just need to split the train file and then convert bio -> json
    """
    assert short_name == "fa_arman"
    language = "fa"
    base_input_path = os.path.join(paths["NERBASE"], "PersianNER")
    train_input_file = os.path.join(base_input_path, "train_fold1.txt")
    test_input_file = os.path.join(base_input_path, "test_fold1.txt")
    if not os.path.exists(train_input_file) or not os.path.exists(test_input_file):
        full_corpus_file = os.path.join(base_input_path, "ArmanPersoNERCorpus.zip")
        if os.path.exists(full_corpus_file):
            raise FileNotFoundError("Please unzip the file {}".format(full_corpus_file))
        raise FileNotFoundError("Cannot find the arman corpus in the expected directory: {}".format(base_input_path))

    base_output_path = paths["NER_DATA_DIR"]
    test_output_file = os.path.join(base_output_path, "%s.test.bio" % short_name)

    split_wikiner(base_output_path, train_input_file, prefix=short_name, train_fraction=0.8, test_section=False)
    shutil.copy2(test_input_file, test_output_file)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_sv_suc3licensed(paths, short_name):
    """
    The .zip provided for SUC3 includes train/dev/test splits already

    This extracts those splits without needing to unzip the original file
    """
    assert short_name == "sv_suc3licensed"
    language = "sv"
    train_input_file = os.path.join(paths["NERBASE"], short_name, "SUC3.0.zip")
    if not os.path.exists(train_input_file):
        raise FileNotFoundError("Cannot find the officially licensed SUC3 dataset in %s" % train_input_file)

    base_output_path = paths["NER_DATA_DIR"]
    suc_conll_to_iob.process_suc3(train_input_file, short_name, base_output_path)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_sv_suc3shuffle(paths, short_name):
    """
    Uses an externally provided script to read the SUC3 XML file, then splits it
    """
    assert short_name == "sv_suc3shuffle"
    language = "sv"
    train_input_file = os.path.join(paths["NERBASE"], short_name, "suc3.xml.bz2")
    if not os.path.exists(train_input_file):
        train_input_file = train_input_file[:-4]
    if not os.path.exists(train_input_file):
        raise FileNotFoundError("Unable to find the SUC3 dataset in {}.bz2".format(train_input_file))

    base_output_path = paths["NER_DATA_DIR"]
    train_output_file = os.path.join(base_output_path, "sv_suc3shuffle.bio")
    suc_to_iob.main([train_input_file, train_output_file])
    split_wikiner(base_output_path, train_output_file, prefix=short_name)
    convert_bio_to_json(base_output_path, base_output_path, short_name)    
    
def process_da_ddt(paths, short_name):
    """
    Processes Danish DDT dataset

    This dataset is in a conll file with the "name" attribute in the
    misc column for the NER tag.  This function uses a script to
    convert such CoNLL files to .bio
    """
    assert short_name == "da_ddt"
    language = "da"
    IN_FILES = ("ddt.train.conllu", "ddt.dev.conllu", "ddt.test.conllu")

    base_output_path = paths["NER_DATA_DIR"]
    OUT_FILES = [os.path.join(base_output_path, "%s.%s.bio" % (short_name, shard)) for shard in SHARDS]

    zip_file = os.path.join(paths["NERBASE"], "da_ddt", "ddt.zip")
    if os.path.exists(zip_file):
        for in_filename, out_filename, shard in zip(IN_FILES, OUT_FILES, SHARDS):
            conll_to_iob.process_conll(in_filename, out_filename, zip_file)
    else:
        for in_filename, out_filename, shard in zip(IN_FILES, OUT_FILES, SHARDS):
            in_filename = os.path.join(paths["NERBASE"], "da_ddt", in_filename)
            if not os.path.exists(in_filename):
                raise FileNotFoundError("Could not find zip in expected location %s and could not file %s file in %s" % (zip_file, shard, in_filename))

            conll_to_iob.process_conll(in_filename, out_filename)
    convert_bio_to_json(base_output_path, base_output_path, short_name)


def process_norne(paths, short_name):
    """
    Processes Norwegian NorNE

    Can handle either Bokmål or Nynorsk

    Converts GPE_LOC and GPE_ORG to GPE
    """
    language, name = short_name.split("_", 1)
    assert language in ('nb', 'nn')
    assert name == 'norne'

    if language == 'nb':
        IN_FILES = ("nob/no_bokmaal-ud-train.conllu", "nob/no_bokmaal-ud-dev.conllu", "nob/no_bokmaal-ud-test.conllu")
    else:
        IN_FILES = ("nno/no_nynorsk-ud-train.conllu", "nno/no_nynorsk-ud-dev.conllu", "nno/no_nynorsk-ud-test.conllu")

    base_output_path = paths["NER_DATA_DIR"]
    OUT_FILES = [os.path.join(base_output_path, "%s.%s.bio" % (short_name, shard)) for shard in SHARDS]

    CONVERSION = { "GPE_LOC": "GPE", "GPE_ORG": "GPE" }

    for in_filename, out_filename, shard in zip(IN_FILES, OUT_FILES, SHARDS):
        in_filename = os.path.join(paths["NERBASE"], "norne", "ud", in_filename)
        if not os.path.exists(in_filename):
            raise FileNotFoundError("Could not find %s file in %s" % (shard, in_filename))

        conll_to_iob.process_conll(in_filename, out_filename, conversion=CONVERSION)

    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_starlang(paths, short_name):
    """
    Process a Turkish dataset from Starlang
    """
    assert short_name == 'tr_starlang'

    PIECES = ["TurkishAnnotatedTreeBank-15",
              "TurkishAnnotatedTreeBank2-15",
              "TurkishAnnotatedTreeBank2-20"]

    chunk_paths = [os.path.join(paths["CONSTITUENCY_BASE"], "turkish", piece) for piece in PIECES]
    datasets = convert_starlang_ner.read_starlang(chunk_paths)

    write_dataset(datasets, paths["NER_DATA_DIR"], short_name)

def process_toy_dataset(paths, short_name):
    convert_bio_to_json(os.path.join(paths["NERBASE"], "English-SAMPLE"), paths["NER_DATA_DIR"], short_name)

def main(dataset_name):
    paths = default_paths.get_default_paths()

    random.seed(1234)

    if dataset_name == 'fi_turku':
        process_turku(paths)
    elif dataset_name == 'it_fbk':
        process_it_fbk(paths)
    elif dataset_name in ('uk_languk', 'Ukranian_languk', 'Ukranian-languk'):
        process_languk(paths)
    elif dataset_name == 'hi_ijc':
        process_ijc(paths, dataset_name)
    elif dataset_name.endswith("FIRE2013") or dataset_name.endswith("fire2013"):
        process_fire_2013(paths, dataset_name)
    elif dataset_name.endswith('WikiNER'):
        process_wikiner(paths, dataset_name)
    elif dataset_name.startswith('hu_rgai'):
        process_rgai(paths, dataset_name)
    elif dataset_name == 'hu_nytk':
        process_nytk(paths)
    elif dataset_name == 'hu_combined':
        process_hu_combined(paths)
    elif dataset_name.endswith("_bsnlp19"):
        process_bsnlp(paths, dataset_name)
    elif dataset_name == 'my_ucsy':
        process_my_ucsy(paths)
    elif dataset_name.endswith("_nchlt"):
        process_nchlt(paths, dataset_name)
    elif dataset_name == "fa_arman":
        process_fa_arman(paths, dataset_name)
    elif dataset_name == "sv_suc3licensed":
        process_sv_suc3licensed(paths, dataset_name)
    elif dataset_name == "sv_suc3shuffle":
        process_sv_suc3shuffle(paths, dataset_name)
    elif dataset_name == "da_ddt":
        process_da_ddt(paths, dataset_name)
    elif dataset_name in ("nb_norne", "nn_norne"):
        process_norne(paths, dataset_name)
    elif dataset_name == 'tr_starlang':
        process_starlang(paths, dataset_name)
    elif dataset_name == 'en_sample':
        process_toy_dataset(paths, dataset_name)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_ner_dataset")

if __name__ == '__main__':
    main(sys.argv[1])
