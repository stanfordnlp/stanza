"""
Processes the SB10k dataset

The original description of the dataset and corpus_v1.0.tsv is here:

https://www.spinningbytes.com/resources/germansentiment/

Download script is here:

https://github.com/aritter/twitter_download

The problem with this file is that many of the tweets with labels no
longer exist.  Roughly 1/3 as of June 2020.

You can contact the authors for the complete dataset.
"""

import argparse
import csv
import os
import random
import sys

from enum import Enum

import stanza

import scripts.sentiment.process_utils as process_utils

class Split(Enum):
    TRAIN_DEV_TEST = 1
    TRAIN_DEV = 2
    TEST = 3

parser = argparse.ArgumentParser()
parser.add_argument('--csv_filename', type=str, default=None, help='CSV file to read in')
parser.add_argument('--out_dir', type=str, default=None, help='Where to write the output files')
parser.add_argument('--sentiment_column', type=int, default=1, help='Column with the sentiment')
parser.add_argument('--text_column', type=int, default=4, help='Column with the text')

parser.add_argument('--split', type=lambda x: Split[x.upper()], default=Split.TRAIN_DEV_TEST,
                    help="How to split the resulting data")

args = parser.parse_args()


nlp = stanza.Pipeline('de', processors='tokenize')


with open(args.csv_filename, newline='') as fin:
    cin = csv.reader(fin, delimiter='\t', quotechar=None)
    lines = list(cin)

# Read in the data and parse it
snippets = []
for line in lines:
    sentiment = line[args.sentiment_column]
    text = line[args.text_column]
    doc = nlp(text)

    #if sentiment.lower() == 'unknown':
    #    continue
    #el
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
    snippets.append(sentiment + " " + " ".join(text))

print(len(snippets))
random.seed(1000)
random.shuffle(snippets)

if args.split is Split.TRAIN_DEV_TEST:
    train_limit = int(len(snippets) * 0.8)
    dev_limit = int(len(snippets) * 0.9)
    process_utils.write_list(os.path.join(args.out_dir, "train.txt"), snippets[:train_limit])
    process_utils.write_list(os.path.join(args.out_dir, "dev.txt"), snippets[train_limit:dev_limit])
    process_utils.write_list(os.path.join(args.out_dir, "test.txt"), snippets[dev_limit:])
elif args.split is Split.TRAIN_DEV:
    train_limit = int(len(snippets) * 0.9)
    process_utils.write_list(os.path.join(args.out_dir, "train.txt"), snippets[:train_limit])
    process_utils.write_list(os.path.join(args.out_dir, "dev.txt"), snippets[train_limit:])
elif args.split is Split.TEST:
    process_utils.write_list(os.path.join(args.out_dir, "test.txt"), snippets)
else:
    raise ValueError("Unknown split method {}".format(args.split))
