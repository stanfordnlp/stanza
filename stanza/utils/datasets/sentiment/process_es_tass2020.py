"""
Convert the TASS 2020 dataset, available here: http://tass.sepln.org/2020/?page_id=74

There are two parts to the dataset, but only part 1 has the gold
annotations available.

Download:
Task 1 train & dev sets
Task 1.1 test set
Task 1.2 test set
Task 1.1 test set gold standard
Task 1.2 test set gold standard   (.tsv, not .zip)

No need to unzip any of the files.  The extraction script reads the
expected paths directly from the zip files.

There are two subtasks in TASS 2020.  One is split among 5 Spanish
speaking countries, and the other is combined across all of the
countries.  Here we combine all of the data into one output file.

Also, each of the subparts are output into their own files, such as
p2.json, p1.mx.json, etc
"""

import os
import zipfile

import stanza

import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.sentiment.process_utils import SentimentDatum, write_dataset, write_list

def convert_label(label):
    """
    N/NEU/P or error
    """
    if label == "N":
        return 0
    if label == "NEU":
        return 1
    if label == "P":
        return 2
    raise ValueError("Unexpected label %s" % label)

def read_test_labels(fin):
    """
    Read a tab (or space) separated list of id/label pairs
    """
    label_map = {}
    for line_idx, line in enumerate(fin):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        pieces = line.split()
        if len(pieces) < 2:
            continue
        if len(pieces) > 2:
            raise ValueError("Unexpected format at line %d: all label lines should be len==2\n%s" % (line_idx, line))

        datum_id, label = pieces
        try:
            label = convert_label(label)
        except ValueError:
            raise ValueError("Unexpected test label %s at line %d\n%s" % (label, line_idx, line))

        label_map[datum_id] = label
    return label_map

def open_read_test_labels(filename, zip_filename=None):
    """
    Open either a text or zip file, then read the labels
    """
    if zip_filename is None:
        with open(filename, encoding="utf-8") as fin:
            test_labels = read_test_labels(fin)
            print("Read %d lines from %s" % (len(test_labels), filename))
            return test_labels

    with zipfile.ZipFile(zip_filename) as zin:
        with zin.open(filename) as fin:
            test_labels = read_test_labels(fin)
            print("Read %d lines from %s - %s" % (len(test_labels), zip_filename, filename))
            return test_labels


def read_sentences(fin):
    """
    Read ids and text from the given file
    """
    lines = []
    for line_idx, line in enumerate(fin):
        line = line.decode("utf-8")
        pieces = line.split(maxsplit=1)
        if len(pieces) < 2:
            continue
        lines.append(pieces)
    return lines

def open_read_sentences(filename, zip_filename):
    """
    Opens a file and then reads the sentences

    Only applies to files inside zips, as all of the sentence files in
    this dataset are inside a zip
    """
    with zipfile.ZipFile(zip_filename) as zin:
        with zin.open(filename) as fin:
            test_sentences = read_sentences(fin)
            print("Read %d texts from %s - %s" % (len(test_sentences), zip_filename, filename))

    return test_sentences

def combine_test_set(sentences, labels):
    """
    Combines the labels and sentences from two pieces of the test set

    Matches the ID from the label files and the text files
    """
    combined = []
    if len(sentences) != len(labels):
        raise ValueError("Lengths of sentences and labels should match!")
    for sent_id, text in sentences:
        label = labels.get(sent_id, None)
        if label is None:
            raise KeyError("Cannot find a test label from the ID: %s" % sent_id)
        # not tokenized yet - we can do tokenization in batches
        combined.append(SentimentDatum(label, text))
    return combined

DATASET_PIECES = ("cr", "es", "mx", "pe", "uy")

def tokenize(sentiment_data, pipe):
    """
    Takes a list of (label, text) and returns a list of SentimentDatum with tokenized text

    Only the first 'sentence' is used - ideally the pipe has ssplit turned off
    """
    docs = [x.text for x in sentiment_data]
    in_docs = [stanza.Document([], text=d) for d in docs]
    out_docs = pipe(in_docs)

    sentiment_data = [SentimentDatum(datum.sentiment,
                                     [y.text for y in doc.sentences[0].tokens]) # list of text tokens for each doc
                      for datum, doc in zip(sentiment_data, out_docs)]

    return sentiment_data

def read_test_set(label_zip_filename, label_filename, sentence_zip_filename, sentence_filename, pipe):
    """
    Read and tokenize an entire test set given the label and sentence filenames
    """
    test_labels = open_read_test_labels(label_filename, label_zip_filename)
    test_sentences = open_read_sentences(sentence_filename, sentence_zip_filename)
    sentiment_data = combine_test_set(test_sentences, test_labels)
    return tokenize(sentiment_data, pipe)

    return sentiment_data

def read_train_file(zip_filename, filename, pipe):
    """
    Read and tokenize a train set

    All of the train data is inside one zip.  We read it one piece at a time
    """
    sentiment_data = []
    with zipfile.ZipFile(zip_filename) as zin:
        with zin.open(filename) as fin:
            for line_idx, line in enumerate(fin):
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                pieces = line.split(maxsplit=1)
                if len(pieces) < 2:
                    continue
                pieces = pieces[1].rsplit(maxsplit=1)
                if len(pieces) < 2:
                    continue
                text, label = pieces
                try:
                    label = convert_label(label)
                except ValueError:
                    raise ValueError("Unexpected train label %s at line %d\n%s" % (label, line_idx, line))
                sentiment_data.append(SentimentDatum(label, text))

    print("Read %d texts from %s - %s" % (len(sentiment_data), zip_filename, filename))
    sentiment_data = tokenize(sentiment_data, pipe)
    return sentiment_data

def convert_tass2020(in_directory, out_directory, dataset_name):
    """
    Read all of the data from in_directory/spanish/tass2020, write it to out_directory/dataset_name...
    """
    in_directory = os.path.join(in_directory, "spanish", "tass2020")

    pipe = stanza.Pipeline(lang="es", processors="tokenize", tokenize_no_ssplit=True)

    test_11 = {}
    test_11_labels_zip = os.path.join(in_directory, "tass2020-test-gold.zip")
    test_11_sentences_zip = os.path.join(in_directory, "Test1.1.zip")
    for piece in DATASET_PIECES:
        inner_label_filename = piece + ".tsv"
        inner_sentence_filename = os.path.join("Test1.1", piece.upper() + ".tsv")
        test_11[piece] = read_test_set(test_11_labels_zip, inner_label_filename,
                                       test_11_sentences_zip, inner_sentence_filename, pipe)

    test_12_label_filename = os.path.join(in_directory, "task1.2-test-gold.tsv")
    test_12_sentences_zip = os.path.join(in_directory, "test1.2.zip")
    test_12_sentences_filename = "test1.2/task1.2.tsv"
    test_12 = read_test_set(None, test_12_label_filename,
                            test_12_sentences_zip, test_12_sentences_filename, pipe)

    train_dev_zip = os.path.join(in_directory, "Task1-train-dev.zip")
    dev = {}
    train = {}
    for piece in DATASET_PIECES:
        dev_filename = os.path.join("dev", piece + ".tsv")
        dev[piece] = read_train_file(train_dev_zip, dev_filename, pipe)

    for piece in DATASET_PIECES:
        train_filename = os.path.join("train", piece + ".tsv")
        train[piece] = read_train_file(train_dev_zip, train_filename, pipe)

    all_test = test_12 + [item for piece in test_11.values() for item in piece]
    all_dev = [item for piece in dev.values() for item in piece]
    all_train = [item for piece in train.values() for item in piece]

    print("Total train items: %8d" % len(all_train))
    print("Total dev items:   %8d" % len(all_dev))
    print("Total test items:  %8d" % len(all_test))

    write_dataset((all_train, all_dev, all_test), out_directory, dataset_name)

    output_file = os.path.join(out_directory, "%s.test.p2.json" % dataset_name)
    write_list(output_file, test_12)

    for piece in DATASET_PIECES:
        output_file = os.path.join(out_directory, "%s.test.p1.%s.json" % (dataset_name, piece))
        write_list(output_file, test_11[piece])

def main(paths):
    in_directory = paths['SENTIMENT_BASE']
    out_directory = paths['SENTIMENT_DATA_DIR']

    convert_tass2020(in_directory, out_directory, "es_tass2020")


if __name__ == '__main__':
    paths = default_paths.get_default_paths()
    main(paths)
