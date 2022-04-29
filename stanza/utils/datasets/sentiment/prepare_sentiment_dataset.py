"""
Prepare a single dataset or a combination dataset for the sentiment project

Manipulates various downloads from their original form to a form
usable by the classifier model

Notes on the individual datasets can be found in the relevant
process_dataset script
"""

import os
import random
import sys

import stanza.utils.default_paths as default_paths

from stanza.utils.datasets.sentiment import process_airline
from stanza.utils.datasets.sentiment import process_arguana_xml
from stanza.utils.datasets.sentiment import process_MELD
from stanza.utils.datasets.sentiment import process_ren_chinese
from stanza.utils.datasets.sentiment import process_sb10k
from stanza.utils.datasets.sentiment import process_scare
from stanza.utils.datasets.sentiment import process_slsd
from stanza.utils.datasets.sentiment import process_sst
from stanza.utils.datasets.sentiment import process_usage_german

from stanza.utils.datasets.sentiment import process_utils

SHARDS = ["train", "dev", "test"]

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
    for shard, phrases in zip(SHARDS, dataset):
        output_file = os.path.join(out_directory, "%s.%s.txt" % (dataset_name, shard))
        process_utils.write_list(output_file, phrases)    

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

def convert_ren(paths, dataset_name):
    in_directory = os.path.join(paths['SENTIMENT_BASE'], "chinese", "RenCECps")
    out_directory = paths['SENTIMENT_DATA_DIR']
    process_ren_chinese.main(in_directory, out_directory, dataset_name)

DATASET_MAPPING = {
    "de_sb10k":   convert_sb10k,
    "de_scare":   convert_scare,
    "de_usage":   convert_de_usage,

    "en_sstplus": convert_sstplus,
    "en_meld":    convert_meld,

    "zh_ren":     convert_ren,
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

