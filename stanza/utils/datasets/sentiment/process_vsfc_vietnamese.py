"""
VSFC sentiment dataset is available at
  https://drive.google.com/drive/folders/1xclbjHHK58zk2X6iqbvMPS2rcy9y9E0X

The format is extremely similar to ours - labels are 0,1,2.
Text needs to be tokenized, though.
Also, the files are split into two pieces, labels and text.
"""

import os
import sys

from tqdm import tqdm

import stanza
from stanza.utils.datasets.sentiment.process_utils import SentimentDatum
import stanza.utils.datasets.sentiment.process_utils as process_utils

import stanza.utils.default_paths as default_paths

def combine_columns(in_directory, dataset, nlp):
    directory = os.path.join(in_directory, dataset)

    sentiment_file = os.path.join(directory, "sentiments.txt")
    with open(sentiment_file) as fin:
        sentiment = fin.readlines()

    text_file = os.path.join(directory, "sents.txt")
    with open(text_file) as fin:
        text = fin.readlines()

    text = [[token.text for sentence in nlp(line.strip()).sentences for token in sentence.tokens]
            for line in tqdm(text)]

    phrases = [SentimentDatum(s.strip(), t) for s, t in zip(sentiment, text)]
    return phrases

def main(in_directory, out_directory, short_name):
    nlp = stanza.Pipeline('vi', processors='tokenize')
    for shard in ("train", "dev", "test"):
        phrases = combine_columns(in_directory, shard, nlp)
        output_file = os.path.join(out_directory, "%s.%s.json" % (short_name, shard))
        process_utils.write_list(output_file, phrases)


if __name__ == '__main__':
    paths = default_paths.get_default_paths()

    if len(sys.argv) <= 1:
        in_directory = os.path.join(paths['SENTIMENT_BASE'], "vietnamese", "_UIT-VSFC")
    else:
        in_directory = sys.argv[1]

    if len(sys.argv) <= 2:
        out_directory = paths['SENTIMENT_DATA_DIR']
    else:
        out_directory = sys.argv[2]

    if len(sys.argv) <= 3:
        short_name = 'vi_vsfc'
    else:
        short_name = sys.argv[3]

    main(in_directory, out_directory, short_name)
