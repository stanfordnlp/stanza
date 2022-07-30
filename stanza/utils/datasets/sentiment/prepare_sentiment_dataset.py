"""Prepare a single dataset or a combination dataset for the sentiment project

Manipulates various downloads from their original form to a form
usable by the classifier model

Explanations for the existing datasets are below.
After processing the dataset, you can train with
the run_sentiment script

python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset <dataset>
python3 -m stanza.utils.training.run_sentiment <dataset>

English
-------

SST (Stanford Sentiment Treebank)
  https://nlp.stanford.edu/sentiment/
  https://github.com/stanfordnlp/sentiment-treebank
  The git repo includes fixed tokenization and sentence splits, along
    with a partial conversion to updated PTB tokenization standards.

  The first step is to git clone the SST to here:
    $SENTIMENT_BASE/sentiment-treebank
  eg:
    cd $SENTIMENT_BASE
    git clone git@github.com:stanfordnlp/sentiment-treebank.git

  There are a few different usages of SST.

  The scores most commonly reported are for SST-2,
    positive and negative only.
  To get a version of this:

    python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset en_sst2
    python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset en_sst2roots

  The model we distribute is a three class model (+, 0, -)
    with some smaller datasets added for better coverage.
    See "sstplus" below.

MELD
  https://github.com/SenticNet/MELD/tree/master/data/MELD
  https://github.com/SenticNet/MELD
  https://arxiv.org/pdf/1810.02508.pdf

  MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. ACL 2019.
  S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.

  An Emotion Corpus of Multi-Party Conversations.
  Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W.

  Copy the three files in the repo into
    $SENTIMENT_BASE/MELD
  TODO: make it so you git clone the repo instead

  There are train/dev/test splits, so you can build a model
    out of just this corpus.  The first step is to convert
    to the classifier data format:

    python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset en_meld

  However, in general we simply include this in the sstplus model
    rather than releasing a separate model.

Arguana
  http://argumentation.bplaced.net/arguana/data
  http://argumentation.bplaced.net/arguana-data/arguana-tripadvisor-annotated-v2.zip

  http://argumentation.bplaced.net/arguana-publications/papers/wachsmuth14a-cicling.pdf
  A Review Corpus for Argumentation Analysis.  CICLing 2014
  Henning Wachsmuth, Martin Trenkmann, Benno Stein, Gregor Engels, Tsvetomira Palarkarska

  Download the zip file and unzip it in
    $SENTIMENT_BASE/arguana

  This is included in the sstplus model.

airline
  A Kaggle corpus for sentiment detection on airline tweets.
  We include this in sstplus as well.

  https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

  Download Tweets.csv and put it in
    $SENTIMENT_BASE/airline

SLSD
  https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

  From Group to Individual Labels using Deep Features.  KDD 2015
  Kotzias et. al

  Put the contents of the zip file in
    $SENTIMENT_BASE/slsd

  The sstplus model includes this as training data

en_sstplus
  This is a three class model built from SST, along with the additional
    English data sources above for coverage of additional domains.

  python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset en_sstplus

German
------

de_sb10k
  This used to be here:
    https://www.spinningbytes.com/resources/germansentiment/
  Now it appears to have moved here?
    https://github.com/oliverguhr/german-sentiment

  https://dl.acm.org/doi/pdf/10.1145/3038912.3052611
  Leveraging Large Amounts of Weakly Supervised Data for Multi-Language Sentiment Classification
  WWW '17: Proceedings of the 26th International Conference on World Wide Web
  Jan Deriu, Aurelien Lucchi, Valeria De Luca, Aliaksei Severyn,
    Simon Müller, Mark Cieliebak, Thomas Hofmann, Martin Jaggi

  The current prep script works on the old version of the data.
  TODO: update to work on the git repo

  python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset de_sb10k

de_scare
  http://romanklinger.de/scare/

  The Sentiment Corpus of App Reviews with Fine-grained Annotations in German
  LREC 2016
  Mario Sänger, Ulf Leser, Steffen Kemmerer, Peter Adolphs, and Roman Klinger

  Download the data and put it in
    $SENTIMENT_BASE/german/scare
  There should be two subdirectories once you are done:
    scare_v1.0.0
    scare_v1.0.0_text

  We wound up not including this in the default German model.
  It might be worth revisiting in the future.

de_usage
  https://www.romanklinger.de/usagecorpus/

  http://www.lrec-conf.org/proceedings/lrec2014/summaries/85.html
  The USAGE Review Corpus for Fine Grained Multi Lingual Opinion Analysis
  Roman Klinger and Philipp Cimiano

  Again, not included in the default German model

Chinese
-------

zh-hans_ren
  This used to be here:
  http://a1-www.is.tokushima-u.ac.jp/member/ren/Ren-CECps1.0/Ren-CECps1.0.html

  That page doesn't seem to respond as of 2022, and I can't find it elsewhere.

The following will be available starting in 1.4.1:

Spanish
-------

tass2020
  - http://tass.sepln.org/2020/?page_id=74
  - Download the following 5 files:
      task1.2-test-gold.tsv
      Task1-train-dev.zip
      tass2020-test-gold.zip
      Test1.1.zip
      test1.2.zip
    Put them in a directory
      $SENTIMENT_BASE/spanish/tass2020

  python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset es_tass2020


Vietnamese
----------

vi_vsfc
  I found a corpus labeled VSFC here:
  https://drive.google.com/drive/folders/1xclbjHHK58zk2X6iqbvMPS2rcy9y9E0X
  It doesn't seem to have a license or paper associated with it,
  but happy to put those details here if relevant.

  Download the files to
    $SENTIMENT_BASE/vietnamese/_UIT-VSFC

  python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset vi_vsfc

Marathi
-------

mr_l3cube
  https://github.com/l3cube-pune/MarathiNLP

  https://arxiv.org/abs/2103.11408
  L3CubeMahaSent: A Marathi Tweet-based Sentiment Analysis Dataset
  Atharva Kulkarni, Meet Mandhane, Manali Likhitkar, Gayatri Kshirsagar, Raviraj Joshi

  git clone the repo in
    $SENTIMENT_BASE

  cd $SENTIMENT_BASE
  git clone git@github.com:l3cube-pune/MarathiNLP.git

  python3 -m stanza.utils.datasets.sentiment.prepare_sentiment_dataset mr_l3cube
"""

import os
import random
import sys

import stanza.utils.default_paths as default_paths

from stanza.utils.datasets.sentiment import process_airline
from stanza.utils.datasets.sentiment import process_arguana_xml
from stanza.utils.datasets.sentiment import process_es_tass2020
from stanza.utils.datasets.sentiment import process_MELD
from stanza.utils.datasets.sentiment import process_ren_chinese
from stanza.utils.datasets.sentiment import process_sb10k
from stanza.utils.datasets.sentiment import process_scare
from stanza.utils.datasets.sentiment import process_slsd
from stanza.utils.datasets.sentiment import process_sst
from stanza.utils.datasets.sentiment import process_usage_german
from stanza.utils.datasets.sentiment import process_vsfc_vietnamese

from stanza.utils.datasets.sentiment import process_utils

def convert_sst_general(paths, dataset_name, version):
    in_directory = paths['SENTIMENT_BASE']
    sst_dir = os.path.join(in_directory, "sentiment-treebank")
    train_phrases = process_sst.get_phrases(version, "train.txt", sst_dir)
    dev_phrases = process_sst.get_phrases(version, "dev.txt", sst_dir)
    test_phrases = process_sst.get_phrases(version, "test.txt", sst_dir)

    out_directory = paths['SENTIMENT_DATA_DIR']
    dataset = [train_phrases, dev_phrases, test_phrases]
    process_utils.write_dataset(dataset, out_directory, dataset_name)

def convert_sst2(paths, dataset_name):
    """
    Create a 2 class SST dataset (neutral items are dropped)
    """
    convert_sst_general(paths, dataset_name, "binary")

def convert_sst2roots(paths, dataset_name):
    """
    Create a 2 class SST dataset using only the roots
    """
    convert_sst_general(paths, dataset_name, "binaryroot")

def convert_sst3roots(paths, dataset_name):
    """
    Create a 3 class SST dataset using only the roots
    """
    convert_sst_general(paths, dataset_name, "threeclassroot")

def convert_sstplus(paths, dataset_name):
    """
    Create a 3 class SST dataset with a few other small datasets added
    """
    train_phrases = []
    in_directory = paths['SENTIMENT_BASE']
    train_phrases.extend(process_arguana_xml.get_tokenized_phrases(os.path.join(in_directory, "arguana")))
    train_phrases.extend(process_MELD.get_tokenized_phrases("train", os.path.join(in_directory, "MELD")))
    train_phrases.extend(process_slsd.get_tokenized_phrases(os.path.join(in_directory, "slsd")))
    train_phrases.extend(process_airline.get_tokenized_phrases(os.path.join(in_directory, "airline")))

    sst_dir = os.path.join(in_directory, "sentiment-treebank")
    train_phrases.extend(process_sst.get_phrases("threeclass", "train.txt", sst_dir))
    train_phrases.extend(process_sst.get_phrases("threeclass", "extra-train.txt", sst_dir))
    train_phrases.extend(process_sst.get_phrases("threeclass", "checked-extra-train.txt", sst_dir))

    dev_phrases = process_sst.get_phrases("threeclass", "dev.txt", sst_dir)
    test_phrases = process_sst.get_phrases("threeclass", "test.txt", sst_dir)

    out_directory = paths['SENTIMENT_DATA_DIR']
    dataset = [train_phrases, dev_phrases, test_phrases]
    process_utils.write_dataset(dataset, out_directory, dataset_name)

def convert_meld(paths, dataset_name):
    """
    Convert the MELD dataset to train/dev/test files
    """
    in_directory = os.path.join(paths['SENTIMENT_BASE'], "MELD")
    out_directory = paths['SENTIMENT_DATA_DIR']
    process_MELD.main(in_directory, out_directory, dataset_name)

def convert_scare(paths, dataset_name):
    in_directory = os.path.join(paths['SENTIMENT_BASE'], "german", "scare")
    out_directory = paths['SENTIMENT_DATA_DIR']
    process_scare.main(in_directory, out_directory, dataset_name)
    

def convert_de_usage(paths, dataset_name):
    in_directory = os.path.join(paths['SENTIMENT_BASE'], "USAGE")
    out_directory = paths['SENTIMENT_DATA_DIR']
    process_usage_german.main(in_directory, out_directory, dataset_name)

def convert_sb10k(paths, dataset_name):
    """
    Essentially runs the sb10k script twice with different arguments to produce the de_sb10k dataset

    stanza.utils.datasets.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/de_full/de_test.tsv --out_dir $SENTIMENT_DATA_DIR --short_name de_sb10k --split test --sentiment_column 2 --text_column 3
    stanza.utils.datasets.sentiment.process_sb10k --csv_filename extern_data/sentiment/german/sb-10k/de_full/de_train.tsv --out_dir $SENTIMENT_DATA_DIR --short_name de_sb10k --split train_dev --sentiment_column 2 --text_column 3
    """
    column_args = ["--sentiment_column", "2", "--text_column", "3"]

    process_sb10k.main(["--csv_filename", os.path.join(paths['SENTIMENT_BASE'], "german", "sb-10k", "de_full", "de_test.tsv"),
                        "--out_dir", paths['SENTIMENT_DATA_DIR'],
                        "--short_name", dataset_name,
                        "--split", "test",
                        *column_args])
    process_sb10k.main(["--csv_filename", os.path.join(paths['SENTIMENT_BASE'], "german", "sb-10k", "de_full", "de_train.tsv"),
                        "--out_dir", paths['SENTIMENT_DATA_DIR'],
                        "--short_name", dataset_name,
                        "--split", "train_dev",
                        *column_args])

def convert_vi_vsfc(paths, dataset_name):
    in_directory = os.path.join(paths['SENTIMENT_BASE'], "vietnamese", "_UIT-VSFC")
    out_directory = paths['SENTIMENT_DATA_DIR']
    process_vsfc_vietnamese.main(in_directory, out_directory, dataset_name)

def convert_mr_l3cube(paths, dataset_name):
    # csv_filename = 'extern_data/sentiment/MarathiNLP/L3CubeMahaSent Dataset/tweets-train.csv'
    MAPPING = {"-1": "0", "0": "1", "1": "2"}

    out_directory = paths['SENTIMENT_DATA_DIR']
    os.makedirs(out_directory, exist_ok=True)

    in_directory = os.path.join(paths['SENTIMENT_BASE'], "MarathiNLP", "L3CubeMahaSent Dataset")
    input_files = ['tweets-train.csv', 'tweets-valid.csv', 'tweets-test.csv']
    input_files = [os.path.join(in_directory, x) for x in input_files]
    datasets = [process_utils.read_snippets(csv_filename, sentiment_column=1, text_column=0, tokenizer_language="mr", mapping=MAPPING, delimiter=',', quotechar='"', skip_first_line=True)
                for csv_filename in input_files]

    process_utils.write_dataset(datasets, out_directory, dataset_name)

def convert_es_tass2020(paths, dataset_name):
    process_es_tass2020.convert_tass2020(paths['SENTIMENT_BASE'], paths['SENTIMENT_DATA_DIR'], dataset_name)

def convert_ren(paths, dataset_name):
    in_directory = os.path.join(paths['SENTIMENT_BASE'], "chinese", "RenCECps")
    out_directory = paths['SENTIMENT_DATA_DIR']
    process_ren_chinese.main(in_directory, out_directory, dataset_name)

DATASET_MAPPING = {
    "de_sb10k":     convert_sb10k,
    "de_scare":     convert_scare,
    "de_usage":     convert_de_usage,

    "en_sst2":      convert_sst2,
    "en_sst2roots": convert_sst2roots,
    "en_sst3roots": convert_sst3roots,
    "en_sstplus":   convert_sstplus,
    "en_meld":      convert_meld,

    "es_tass2020":  convert_es_tass2020,

    "mr_l3cube":    convert_mr_l3cube,

    "vi_vsfc":      convert_vi_vsfc,

    "zh-hans_ren":  convert_ren,
}

def main(dataset_name):
    paths = default_paths.get_default_paths()

    random.seed(1234)

    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name)
    else:
        raise ValueError(f"dataset {dataset_name} currently not handled")

if __name__ == '__main__':
    main(sys.argv[1])

