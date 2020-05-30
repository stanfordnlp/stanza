"""
SCARE is a dataset of German text with sentiment annotations.

http://romanklinger.de/scare/

You need to fill out a license agreement to not redistribute the data
in order to get the data, but the process is not onerous.

Although it sounds interesting, there are unfortunately a lot of very
short items.  Not sure the long items will be enough
"""


import csv
import glob
import os
import random
import sys

import stanza

def write_list(out_filename, dataset):
    with open(out_filename, 'w') as fout:
        for line in dataset:
            fout.write(line)
            fout.write("\n")

basedir = sys.argv[1]
nlp = stanza.Pipeline('de', processors='tokenize')

text_files = glob.glob(os.path.join(basedir, "scare_v1.0.0_text/annotations/*txt"))
text_id_map = {}
for filename in text_files:
    with open(filename) as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            if key in text_id_map:
                raise ValueError("Duplicate key {}".format(key))
            text_id_map[key] = value

print(len(text_id_map))

num_short_items = 0

snippets = []
csv_files = glob.glob(os.path.join(basedir, "scare_v1.0.0/annotations/*csv"))
for csv_filename in csv_files:
    with open(csv_filename, newline='') as fin:
        cin = csv.reader(fin, delimiter='\t', quotechar='"')
        lines = list(cin)

        for line in lines:
            ann_id, begin, end, sentiment = [line[i] for i in [1, 2, 3, 6]]
            begin = int(begin)
            end = int(end)
            if sentiment == 'Unknown':
                continue
            elif sentiment == 'Positive':
                sentiment = 2
            elif sentiment == 'Neutral':
                sentiment = 1
            elif sentiment == 'Negative':
                sentiment = 0
            else:
                raise ValueError("Tell John he screwed up and this is why he can't have Mox Opal")
            snippet = text_id_map[ann_id][begin:end]
            doc = nlp(snippet)
            text = " ".join(sentence.text for sentence in doc.sentences)
            num_tokens = sum(len(sentence.tokens) for sentence in doc.sentences)
            if num_tokens < 4:
                num_short_items = num_short_items + 1
            snippets.append("%d %s" % (sentiment, text))

print(len(snippets))
print("Number of short items: {}".format(num_short_items))
random.seed(1000)
random.shuffle(snippets)
train_limit = int(len(snippets) * 0.8)
dev_limit = int(len(snippets) * 0.9)
write_list(os.path.join(basedir, "train.txt"), snippets[:train_limit])
write_list(os.path.join(basedir, "dev.txt"), snippets[train_limit:dev_limit])
write_list(os.path.join(basedir, "test.txt"), snippets[dev_limit:])

