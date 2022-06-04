"""
Processes the SB10k dataset

The original description of the dataset and corpus_v1.0.tsv is here:

https://www.spinningbytes.com/resources/germansentiment/

Download script is here:

https://github.com/aritter/twitter_download

The problem with this file is that many of the tweets with labels no
longer exist.  Roughly 1/3 as of June 2020.

You can contact the authors for the complete dataset.

There is a paper describing some experiments run on the dataset here:
https://dl.acm.org/doi/pdf/10.1145/3038912.3052611
"""

import argparse
import csv
import os
import random
import sys

from enum import Enum
from tqdm import tqdm

import stanza

from stanza.utils.datasets.sentiment.process_utils import Fragment
import stanza.utils.datasets.sentiment.process_utils as process_utils

class Split(Enum):
    TRAIN_DEV_TEST = 1
    TRAIN_DEV = 2
    TEST = 3

def read_snippets(csv_filename, sentiment_column, text_column):
    nlp = stanza.Pipeline('de', processors='tokenize')

    with open(csv_filename, newline='') as fin:
        cin = csv.reader(fin, delimiter='\t', quotechar=None)
        lines = list(cin)

    # Read in the data and parse it
    snippets = []
    for line in tqdm(lines):
        sentiment = line[sentiment_column]
        text = line[text_column]
        doc = nlp(text)

        if sentiment.lower() == 'positive':
            sentiment = "2"
        elif sentiment.lower() == 'neutral':
            sentiment = "1"
        elif sentiment.lower() == 'negative':
            sentiment = "0"
        else:
            raise ValueError("Tell John he screwed up and this is why he can't have Mox Opal: {}".format(sentiment))

        text = []
        for sentence in doc.sentences:
            text.extend(token.text for token in sentence.tokens)
        text = process_utils.clean_tokenized_tweet(text)
        snippets.append(Fragment(sentiment, text))
    return snippets

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_filename', type=str, default=None, help='CSV file to read in')
    parser.add_argument('--out_dir', type=str, default=None, help='Where to write the output files')
    parser.add_argument('--sentiment_column', type=int, default=2, help='Column with the sentiment')
    parser.add_argument('--text_column', type=int, default=3, help='Column with the text')
    parser.add_argument('--short_name', type=str, default="sb10k", help='short name to use when writing files')

    parser.add_argument('--split', type=lambda x: Split[x.upper()], default=Split.TRAIN_DEV_TEST,
                        help="How to split the resulting data")

    args = parser.parse_args(args=args)

    snippets = read_snippets(args.csv_filename, args.sentiment_column, args.text_column)

    print(len(snippets))
    random.shuffle(snippets)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.split is Split.TRAIN_DEV_TEST:
        process_utils.write_splits(args.out_dir,
                                   snippets,
                                   (process_utils.Split("%s.train.json" % args.short_name, 0.8),
                                    process_utils.Split("%s.dev.json" % args.short_name, 0.1),
                                    process_utils.Split("%s.test.json" % args.short_name, 0.1)))
    elif args.split is Split.TRAIN_DEV:
        process_utils.write_splits(args.out_dir,
                                   snippets,
                                   (process_utils.Split("%s.train.json" % args.short_name, 0.9),
                                    process_utils.Split("%s.dev.json" % args.short_name, 0.1)))
    elif args.split is Split.TEST:
        process_utils.write_list(os.path.join(args.out_dir, "%s.test.json" % args.short_name), snippets)
    else:
        raise ValueError("Unknown split method {}".format(args.split))

if __name__ == '__main__':
    random.seed(1234)
    main()

