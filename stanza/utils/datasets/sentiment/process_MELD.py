"""
MELD is a dataset of Friends (the TV show) utterances.  

The ratings include judgment based on the visuals, so it might be
harder than expected to directly extract from the text.  However, it
should broaden the scope of the model and doesn't seem to hurt
performance.

https://github.com/SenticNet/MELD/tree/master/data/MELD

https://github.com/SenticNet/MELD

https://arxiv.org/pdf/1810.02508.pdf

Files in the MELD repo are csv, with quotes in "..." if they contained commas themselves.

Accordingly, we use the csv module to read the files and output them in the format
<class> <sentence>

Run using 

python3 convert_MELD.py MELD/train_sent_emo.csv train.txt
etc

"""

import csv
import os
import sys

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

def get_phrases(in_filename):
    """
    Get the phrases from a single CSV filename
    """
    with open(in_filename, newline='', encoding='windows-1252') as fin:
        cin = csv.reader(fin, delimiter=',', quotechar='"')
        lines = list(cin)

    phrases = []
    for line in lines[1:]:
        sentiment = line[4]
        if sentiment == 'negative':
            sentiment = '0'
        elif sentiment == 'neutral':
            sentiment = '1'
        elif sentiment == 'positive':
            sentiment = '2'
        else:
            raise ValueError("Unknown sentiment: {}".format(sentiment))
        utterance = line[1].replace("Â", "")
        phrases.append(SentimentDatum(sentiment, utterance))
    return phrases

def get_tokenized_phrases(split, in_directory):
    """
    split in train,dev,test
    """
    in_filename  = os.path.join(in_directory, "%s_sent_emo.csv" % split)
    phrases = get_phrases(in_filename)

    phrases = process_utils.get_ptb_tokenized_phrases(phrases)
    print("Found {} phrases in MELD {}".format(len(phrases), split))
    return phrases

def main(in_directory, out_directory, short_name):
    os.makedirs(out_directory, exist_ok=True)
    for split in ("train", "dev", "test"):
        phrases = get_tokenized_phrases(split, in_directory)
        process_utils.write_list(os.path.join(out_directory, "%s.%s.json" % (short_name, split)), phrases)

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]

    main(in_directory, out_directory, short_name)
