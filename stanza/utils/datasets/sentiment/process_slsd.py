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
import tempfile

def main(in_directory, out_directory, short_name):
    in_filenames = [os.path.join(in_directory, 'amazon_cells_labelled.txt'),
                    os.path.join(in_directory, 'imdb_labelled.txt'),
                    os.path.join(in_directory, 'yelp_labelled.txt')]
    out_filename = os.path.join(out_directory, "%s.train.txt" % short_name)
    os.makedirs(out_directory, exist_ok=True)

    lines = []
    for filename in in_filenames:
        lines.extend(open(filename, newline=''))

    tmp_filename = tempfile.NamedTemporaryFile(delete=False).name
    with open(tmp_filename, "w") as fout:
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
            fout.write("%s %s\n" % (sentiment, utterance))

    os.system("java edu.stanford.nlp.process.PTBTokenizer -preserveLines %s > %s" % (tmp_filename, out_filename))
    os.unlink(tmp_filename)

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    short_name = sys.argv[3]
    main(in_directory, out_directory, short_name)
