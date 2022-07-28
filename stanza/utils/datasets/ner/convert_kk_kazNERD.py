"""
Convert a Kazakh NER dataset to our internal .json format
The dataset is here:

https://github.com/IS2AI/KazNERD/tree/main/KazNERD
"""

import argparse
import os
import shutil
# import random

from stanza.utils.datasets.ner.utils import convert_bio_to_json, SHARDS

def convert_dataset(in_directory, out_directory, short_name):
    """
    Reads in train, validation, and test data and converts them to .json file
    """
    filenames = ("IOB2_train.txt", "IOB2_valid.txt", "IOB2_test.txt")
    for shard, filename in zip(SHARDS, filenames):
        input_filename = os.path.join(in_directory, filename)
        output_filename = os.path.join(out_directory, "%s.%s.bio" % (short_name, shard))
        shutil.copy(input_filename, output_filename)
    convert_bio_to_json(out_directory, out_directory, short_name, "bio")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="/nlp/scr/aaydin/kazNERD/NER", help="Where to find the files")
    parser.add_argument('--output_path', type=str, default="/nlp/scr/aaydin/kazNERD/data/ner", help="Where to output the results")
    args = parser.parse_args()
    # in_path = '/nlp/scr/aaydin/kazNERD/NER'
    # out_path = '/nlp/scr/aaydin/kazNERD/NER/output'
    # convert_dataset(in_path, out_path)
    convert_dataset(args.input_path, args.output_path, "kk_kazNERD")

