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

import csv
import os
import random
import sys

import stanza

import scripts.sentiment.process_utils as process_utils

nlp = stanza.Pipeline('de', processors='tokenize')

csv_filename = sys.argv[1]
with open(csv_filename, newline='') as fin:
    cin = csv.reader(fin, delimiter='\t', quotechar=None)
    lines = list(cin)

snippets = []

for line in lines:
    sentiment = line[1]
    text = line[4]
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

out_dir = sys.argv[2]

print(len(snippets))
random.seed(1000)
random.shuffle(snippets)
train_limit = int(len(snippets) * 0.8)
dev_limit = int(len(snippets) * 0.9)
process_utils.write_list(os.path.join(out_dir, "train.txt"), snippets[:train_limit])
process_utils.write_list(os.path.join(out_dir, "dev.txt"), snippets[train_limit:dev_limit])
process_utils.write_list(os.path.join(out_dir, "test.txt"), snippets[dev_limit:])
