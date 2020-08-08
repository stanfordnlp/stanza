"""
Airline tweets from Kaggle
from https://www.kaggle.com/crowdflower/twitter-airline-sentiment/data#
Some ratings seem questionable, but it doesn't hurt performance much, if at all

Files in the airline repo are csv, with quotes in "..." if they contained commas themselves.

Accordingly, we use the csv module to read the files and output them in the format
<class> <sentence>

Run using 

python3 convert_airline.py Tweets.csv train.txt

If the first word is an @, it is removed, and after that, leading @ or # are removed.
For example:

@AngledLuffa you must hate having Mox Opal #banned
-> 
you must hate having Mox Opal banned
"""

import csv
import os
import sys
import tempfile

import scripts.sentiment.process_utils as process_utils

in_filename = sys.argv[1]
out_filename = sys.argv[2]

with open(in_filename, newline='') as fin:
    cin = csv.reader(fin, delimiter=',', quotechar='"')
    lines = list(cin)

tmp_filename = tempfile.NamedTemporaryFile(delete=False).name
with open(tmp_filename, "w") as fout:
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
        fout.write("%s %s\n" % (sentiment, utterance))

tmp2_filename = tempfile.NamedTemporaryFile(delete=False).name
os.system("java edu.stanford.nlp.process.PTBTokenizer -preserveLines %s > %s" % (tmp_filename, tmp2_filename))
os.unlink(tmp_filename)

# filter leading @United, @American, etc from the tweets
lines = open(tmp2_filename).readlines()
lines = [x.strip().split() for x in lines if x.strip()]
lines = [[line[0]] + process_utils.clean_tokenized_tweet(line[1:]) for line in lines]

lines = [' '.join(x) for x in lines]

# this would count @s if you cared enough to count
#ats = Counter()
#for line in lines:
#    ats.update([x for x in line.split() if x[0] == '@'])

with open(out_filename, "w") as fout:
    for line in lines:
        fout.write(line)
        fout.write("\n")

