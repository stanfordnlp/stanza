"""
VSFC sentiment dataset is available at
  https://drive.google.com/drive/folders/1xclbjHHK58zk2X6iqbvMPS2rcy9y9E0X

The format is extremely similar to ours - labels are 0,1,2 and the
text is pretokenized.  The big difference is that the files are split
into two pieces, labels and text.
"""

import os

BASE_DIR = "extern_data/sentiment/vietnamese/_UIT-VSFC"

def combine_columns(dataset):
    directory = os.path.join(BASE_DIR, dataset)

    sentiment_file = os.path.join(directory, "sentiments.txt")
    with open(sentiment_file) as fin:
        sentiment = fin.readlines()

    text_file = os.path.join(directory, "sents.txt")
    with open(text_file) as fin:
        text = fin.readlines()

    output_file = os.path.join(BASE_DIR, dataset + ".txt")
    with open(output_file, "w") as fout:
        for s, t in zip(sentiment, text):
            fout.write("%s %s" % (s.strip(), t))

combine_columns("train")
combine_columns("dev")
combine_columns("test")
