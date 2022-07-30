"""
Airline tweets from Kaggle
from https://www.kaggle.com/crowdflower/twitter-airline-sentiment/data#
Some ratings seem questionable, but it doesn't hurt performance much, if at all

Files in the airline repo are csv, with quotes in "..." if they contained commas themselves.

Accordingly, we use the csv module to read the files and output them in the format
<class> <sentence>

Run using 

python3 convert_airline.py Tweets.csv train.json

If the first word is an @, it is removed, and after that, leading @ or # are removed.
For example:

@AngledLuffa you must hate having Mox Opal #banned
-> 
you must hate having Mox Opal banned
"""

import csv
import os
import sys

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

def get_phrases(in_directory):
    in_filename = os.path.join(in_directory, "Tweets.csv")
    with open(in_filename, newline='') as fin:
        cin = csv.reader(fin, delimiter=',', quotechar='"')
        lines = list(cin)

    phrases = []
    for line in lines[1:]:
        sentiment = line[1]
        if sentiment == 'negative':
            sentiment = '0'
        elif sentiment == 'neutral':
            sentiment = '1'
        elif sentiment == 'positive':
            sentiment = '2'
        else:
            raise ValueError("Unknown sentiment: {}".format(sentiment))
        # some of the tweets have \n in them
        utterance = line[10].replace("\n", " ")
        phrases.append(SentimentDatum(sentiment, utterance))

    return phrases

def get_tokenized_phrases(in_directory):
    phrases = get_phrases(in_directory)
    phrases = process_utils.get_ptb_tokenized_phrases(phrases)
    phrases = [SentimentDatum(x.sentiment, process_utils.clean_tokenized_tweet(x.text)) for x in phrases]
    print("Found {} phrases in the airline corpus".format(len(phrases)))
    return phrases

def main(in_directory, out_directory, short_name):
    phrases = get_tokenized_phrases(in_directory)

    os.makedirs(out_directory, exist_ok=True)
    out_filename = os.path.join(out_directory, "%s.train.json" % short_name)
    # filter leading @United, @American, etc from the tweets
    process_utils.write_list(out_filename, phrases)

    # something like this would count @s if you cared enough to count
    # would need to update for SentimentDatum()
    #ats = Counter()
    #for line in lines:
    #    ats.update([x for x in line.split() if x[0] == '@'])

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]
    main(in_directory, out_directory, short_name)
