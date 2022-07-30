"""
SCARE is a dataset of German text with sentiment annotations.

http://romanklinger.de/scare/

To run the script, pass in the directory where scare was unpacked.  It
should have subdirectories scare_v1.0.0 and scare_v1.0.0_text

You need to fill out a license agreement to not redistribute the data
in order to get the data, but the process is not onerous.

Although it sounds interesting, there are unfortunately a lot of very
short items.  Not sure the long items will be enough
"""


import csv
import glob
import os
import sys

import stanza

from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

def get_scare_snippets(nlp, csv_dir_path, text_id_map, filename_pattern="*.csv"):
    """
    Read snippets from the given CSV directory
    """
    num_short_items = 0

    snippets = []
    csv_files = glob.glob(os.path.join(csv_dir_path, filename_pattern))
    for csv_filename in csv_files:
        with open(csv_filename, newline='') as fin:
            cin = csv.reader(fin, delimiter='\t', quotechar='"')
            lines = list(cin)

            for line in lines:
                ann_id, begin, end, sentiment = [line[i] for i in [1, 2, 3, 6]]
                begin = int(begin)
                end = int(end)
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
                if ann_id not in text_id_map:
                    print("Found snippet which can't be found: {}-{}".format(csv_filename, ann_id))
                    continue
                snippet = text_id_map[ann_id][begin:end]
                doc = nlp(snippet)
                text = [token.text for sentence in doc.sentences for token in sentence.tokens]
                num_tokens = sum(len(sentence.tokens) for sentence in doc.sentences)
                if num_tokens < 4:
                    num_short_items = num_short_items + 1
                snippets.append(SentimentDatum(sentiment, text))
    print("Number of short items: {}".format(num_short_items))
    return snippets


def main(in_directory, out_directory, short_name):
    os.makedirs(out_directory, exist_ok=True)

    input_path = os.path.join(in_directory, "scare_v1.0.0_text", "annotations", "*txt")
    text_files = glob.glob(input_path)
    if len(text_files) == 0:
        raise FileNotFoundError("Did not find any input files in %s" % input_path)
    else:
        print("Found %d input files in %s" % (len(text_files), input_path))
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

    print("Found %d total sentiment ratings" % len(text_id_map))
    nlp = stanza.Pipeline('de', processors='tokenize')
    snippets = get_scare_snippets(nlp, os.path.join(in_directory, "scare_v1.0.0", "annotations"), text_id_map)

    print(len(snippets))
    process_utils.write_list(os.path.join(out_directory, "%s.train.json" % short_name), snippets)

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]

    main(in_directory, out_directory, short_name)
