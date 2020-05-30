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
import tempfile

in_filename = sys.argv[1]
out_filename = sys.argv[2]

with open(in_filename, newline='', encoding='windows-1252') as fin:
    cin = csv.reader(fin, delimiter=',', quotechar='"')
    lines = list(cin)

tmp_filename = tempfile.NamedTemporaryFile(delete=False).name
with open(tmp_filename, "w") as fout:
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
        fout.write("%s %s\n" % (sentiment, utterance))


os.system("java edu.stanford.nlp.process.PTBTokenizer -preserveLines %s > %s" % (tmp_filename, out_filename))
os.unlink(tmp_filename)
