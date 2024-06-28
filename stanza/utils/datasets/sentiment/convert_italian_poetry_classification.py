"""
A short tool to turn a labeled dataset of the format
Prof. Delmonte provided into a stanza input file for the classifier.

Data is expected to be in the sentiment italian subdirectory (see below)

Only writes a test set.  Use it as an eval file for a trained model.
"""

import os

import stanza
from stanza.models.classifiers.data import SentimentDatum
from stanza.utils.datasets.sentiment import process_utils
import stanza.utils.default_paths as default_paths

def main():
    paths = default_paths.get_default_paths()

    dataset_name = "it_vit_sentences_poetry"

    poetry_filename = os.path.join(paths["SENTIMENT_BASE"], "italian", "sentence_classification", "poetry", "testset_300_labeled.txt")
    if not os.path.exists(poetry_filename):
        raise FileNotFoundError("Expected to find the labeled file in %s" % poetry_filename)
    print("Reading the labeled poetry from %s" % poetry_filename)

    tokenizer = stanza.Pipeline("it", processors="tokenize", tokenize_no_ssplit=True)
    dataset = []
    with open(poetry_filename, encoding="utf-8") as fin:
        for line_num, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            line = line.replace(u'\ufeff', '')
            pieces = line.split(maxsplit=1)
            # first column is the label
            # remainder of the text is the raw text
            label = pieces[0].strip()
            if label not in ('0', '1'):
                if label == "viene" and line_num == 257:
                    print("Skipping known missing label at line 257")
                    continue
                assert isinstance(label, str)
                ords = ",".join(str(ord(x)) for x in label)
                raise ValueError("Unexpected label |%s| (%s) for line %d" % (label, ords, line_num))

            # tokenize the text into words
            # we could make this faster by stacking it, but the input file is quite short anyway
            text = pieces[1]
            doc = tokenizer(text)
            words = [x.text for x in doc.sentences[0].words]

            dataset.append(SentimentDatum(label, words))

    print("Read %d lines from %s" % (len(dataset), poetry_filename))
    output_filename = "%s.test.json" % dataset_name
    output_path = os.path.join(paths["SENTIMENT_DATA_DIR"], output_filename)
    print("Writing output to %s" % output_path)
    process_utils.write_list(output_path, dataset)


if __name__ == '__main__':
    main()
