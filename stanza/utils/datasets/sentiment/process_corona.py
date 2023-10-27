"""
Processes a kaggle covid-19 text classification dataset

The original description of the dataset is here:

https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification

There are two files in the archive, Corona_NLP_train.csv and Corona_NLP_test.csv
Unzip the files in archive.zip to $SENTIMENT_BASE/english/corona/Corona_NLP_train.csv

There is no dedicated dev set, so we randomly split train/dev
(using a specific seed, so that the split always comes out the same)
"""

import argparse
import os
import random

import stanza

import stanza.utils.datasets.sentiment.process_utils as process_utils
from stanza.utils.default_paths import get_default_paths

# TODO: could give an option to keep the 'extremely'
MAPPING = {'extremely positive': "2",
           'positive': "2",
           'neutral': "1",
           'negative': "0",
           'extremely negative': "0"}

def main(args=None):
    default_paths = get_default_paths()
    sentiment_base_dir = default_paths["SENTIMENT_BASE"]
    default_in_dir = os.path.join(sentiment_base_dir, "english", "corona")
    default_out_dir = default_paths["SENTIMENT_DATA_DIR"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=default_in_dir, help='Where to get the input files')
    parser.add_argument('--out_dir', type=str, default=default_out_dir, help='Where to write the output files')
    parser.add_argument('--short_name', type=str, default="en_corona", help='short name to use when writing files')
    args = parser.parse_args(args=args)

    TEXT_COLUMN = 4
    SENTIMENT_COLUMN = 5

    train_csv = os.path.join(args.in_dir, "Corona_NLP_train.csv")
    test_csv = os.path.join(args.in_dir, "Corona_NLP_test.csv")

    nlp = stanza.Pipeline("en", processors='tokenize')

    train_snippets = process_utils.read_snippets(train_csv, SENTIMENT_COLUMN, TEXT_COLUMN, 'en', MAPPING, delimiter=",", quotechar='"', skip_first_line=True, nlp=nlp, encoding="latin1")
    test_snippets = process_utils.read_snippets(test_csv, SENTIMENT_COLUMN, TEXT_COLUMN, 'en', MAPPING, delimiter=",", quotechar='"', skip_first_line=True, nlp=nlp, encoding="latin1")

    print("Read %d train snippets" % len(train_snippets))
    print("Read %d test snippets" % len(test_snippets))

    random.seed(1234)
    random.shuffle(train_snippets)

    os.makedirs(args.out_dir, exist_ok=True)
    process_utils.write_splits(args.out_dir,
                               train_snippets,
                               (process_utils.Split("%s.train.json" % args.short_name, 0.9),
                                process_utils.Split("%s.dev.json" % args.short_name, 0.1)))
    process_utils.write_list(os.path.join(args.out_dir, "%s.test.json" % args.short_name), test_snippets)

if __name__ == '__main__':
    main()

