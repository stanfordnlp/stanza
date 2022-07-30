"""
USAGE is produced by the same people as SCARE.  

USAGE has a German and English part.  This script parses the German part.
Run the script as 
  process_usage_german.py path

Here, path should be where USAGE was unpacked.  It will have the
documents, files, etc subdirectories.

https://www.romanklinger.de/usagecorpus/
"""

import csv
import glob
import os
import sys

import stanza

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

def main(in_directory, out_directory, short_name):
    os.makedirs(out_directory, exist_ok=True)
    nlp = stanza.Pipeline('de', processors='tokenize')

    num_short_items = 0
    snippets = []
    csv_files = glob.glob(os.path.join(in_directory, "files/de*csv"))
    for csv_filename in csv_files:
        with open(csv_filename, newline='') as fin:
            cin = csv.reader(fin, delimiter='\t', quotechar=None)
            lines = list(cin)

            for index, line in enumerate(lines):
                begin, end, snippet, sentiment = [line[i] for i in [2, 3, 4, 6]]
                begin = int(begin)
                end = int(end)
                if len(snippet) != end - begin:
                    raise ValueError("Error found in {} line {}.  Expected {} got {}".format(csv_filename, index, (end-begin), len(snippet)))
                if sentiment.lower() == 'unknown':
                    continue
                elif sentiment.lower() == 'positive':
                    sentiment = 2
                elif sentiment.lower() == 'neutral':
                    sentiment = 1
                elif sentiment.lower() == 'negative':
                    sentiment = 0
                else:
                    raise ValueError("Tell John he screwed up and this is why he can't have Mox Opal: {}".format(sentiment))
                doc = nlp(snippet)
                text = [token.text for sentence in doc.sentences for token in sentence.tokens]
                num_tokens = sum(len(sentence.tokens) for sentence in doc.sentences)
                if num_tokens < 4:
                    num_short_items = num_short_items + 1
                snippets.append(SentimentDatum(sentiment, text))

    print("Total snippets found for USAGE: %d" % len(snippets))

    process_utils.write_list(os.path.join(out_directory, "%s.train.json" % short_name), snippets)

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]

    main(in_directory, out_directory, short_name)
