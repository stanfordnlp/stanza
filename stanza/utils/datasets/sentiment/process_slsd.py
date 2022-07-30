"""
A small dataset of 1500 positive and 1500 negative sentences.
Supposedly has no neutral sentences by design

https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

https://archive.ics.uci.edu/ml/machine-learning-databases/00331/

See the existing readme for citation requirements etc

Files in the slsd repo were one line per annotation, with labels 0
for negative and 1 for positive.  No neutral labels existed.

Accordingly, we rearrange the text and adjust the label to fit the
0/1/2 paradigm.  Text is retokenized using PTBTokenizer.

<class> <sentence>

process_slsd.py <directory> <outputfile>
"""

import os
import sys

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

def get_phrases(in_directory):
    in_filenames = [os.path.join(in_directory, 'amazon_cells_labelled.txt'),
                    os.path.join(in_directory, 'imdb_labelled.txt'),
                    os.path.join(in_directory, 'yelp_labelled.txt')]

    lines = []
    for filename in in_filenames:
        lines.extend(open(filename, newline=''))

    phrases = []
    for line in lines:
        line = line.strip()
        sentiment = line[-1]
        utterance = line[:-1]
        utterance = utterance.replace("!.", "!")
        utterance = utterance.replace("?.", "?")
        if sentiment == '0':
            sentiment = '0'
        elif sentiment == '1':
            sentiment = '2'
        else:
            raise ValueError("Unknown sentiment: {}".format(sentiment))
        phrases.append(SentimentDatum(sentiment, utterance))

    return phrases

def get_tokenized_phrases(in_directory):
    phrases = get_phrases(in_directory)
    phrases = process_utils.get_ptb_tokenized_phrases(phrases)
    print("Found %d phrases in slsd" % len(phrases))
    return phrases

def main(in_directory, out_directory, short_name):
    phrases = get_tokenized_phrases(in_directory)
    out_filename = os.path.join(out_directory, "%s.train.json" % short_name)
    os.makedirs(out_directory, exist_ok=True)
    process_utils.write_list(out_filename, phrases)


if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]
    main(in_directory, out_directory, short_name)
