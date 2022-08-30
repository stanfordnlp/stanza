"""Parses the BEST Thai dataset.

That is to say, the dataset named BEST.  We have not yet figured out
which segmentation standard we prefer.

Note that the version of BEST we used actually had some strange
sentence splits according to a native Thai speaker.  Not sure how to
fix that.  Options include doing it automatically or finding some
knowledgable annotators to resplit it for us (or just not using BEST)

This outputs the tokenization results in a conll format similar to
that of the UD treebanks, so we pretend to be a UD treebank for ease
of compatibility with the stanza tools.

BEST can be downloaded from here:

https://aiforthai.in.th/corpus.php

python3 -m stanza.utils.datasets.tokenization.process_best extern_data/thai/best data/tokenize
./scripts/run_tokenize.sh UD_Thai-best --dropout 0.05 --unit_dropout 0.05 --steps 50000
"""
import glob
import os
import random
import re
import sys

try:
    from pythainlp import sent_tokenize
except ImportError:
    pass

from stanza.utils.datasets.tokenization.process_thai_tokenization import reprocess_lines, write_dataset, convert_processed_lines, write_dataset_best, write_dataset

def clean_line(line):
    line = line.replace("html>", "html|>")
    # news_00089.txt
    line = line.replace("<NER>", "<NE>")
    line = line.replace("</NER>", "</NE>")
    # specific error that occurs in encyclopedia_00095.txt
    line = line.replace("</AB>Penn", "</AB>|Penn>")
    # news_00058.txt
    line = line.replace("<AB>จม.</AB>เปิดผนึก", "<AB>จม.</AB>|เปิดผนึก")
    # news_00015.txt
    line = re.sub("<NE><AB>([^|<>]+)</AB>([^|<>]+)</NE>", "\\1|\\2", line)
    # news_00024.txt
    line = re.sub("<NE><AB>([^|<>]+)</AB></NE>", "\\1", line)
    # news_00055.txt
    line = re.sub("<NE>([^|<>]+)<AB>([^|<>]+)</AB></NE>", "\\1|\\2", line)
    line = re.sub("<NE><AB>([^|<>]+)</AB><AB>([^|<>]+)</AB></NE>", "\\1|\\2", line)
    line = re.sub("<NE>([^|<>]+)<AB>([^|<>]+)</AB> <AB>([^|<>]+)</AB></NE>", "\\1|\\2|\\3", line)
    # news_00008.txt and other news articles
    line = re.sub("</AB>([0-9])", "</AB>|\\1", line)
    line = line.replace("</AB> ", "</AB>|")
    line = line.replace("<EM>", "<POEM>")
    line = line.replace("</EM>", "</POEM>")
    line = line.strip()
    return line


def clean_word(word):
    # novel_00078.txt
    if word == '<NEพี่มน</NE>':
        return 'พี่มน'
    if word.startswith("<NE>") and word.endswith("</NE>"):
        return word[4:-5]
    if word.startswith("<AB>") and word.endswith("</AB>"):
        return word[4:-5]
    if word.startswith("<POEM>") and word.endswith("</POEM>"):
        return word[6:-7]
    """
    if word.startswith("<EM>"):
        return word[4:]
    if word.endswith("</EM>"):
        return word[:-5]
    """
    if word.startswith("<NE>"):
        return word[4:]
    if word.endswith("</NE>"):
        return word[:-5]
    if word.startswith("<POEM>"):
        return word[6:]
    if word.endswith("</POEM>"):
        return word[:-7]
    if word == '<':
        return word
    return word

def read_data(input_dir):
    # data for test sets
    test_files = [os.path.join(input_dir, 'TEST_100K_ANS.txt')]
    print(test_files)

    # data for train and dev sets
    subdirs = [os.path.join(input_dir, 'article'),
               os.path.join(input_dir, 'encyclopedia'),
               os.path.join(input_dir, 'news'),
               os.path.join(input_dir, 'novel')]
    files = []
    for subdir in subdirs:
        if not os.path.exists(subdir):
            raise FileNotFoundError("Expected a directory that did not exist: {}".format(subdir))
        files.extend(glob.glob(os.path.join(subdir, '*.txt')))

    test_documents = []
    for filename in test_files:
        print("File name:", filename)
        with open(filename) as fin:
            processed_lines = []
            for line in fin.readlines():
                line = clean_line(line)
                words = line.split("|")
                words = [clean_word(x) for x in words]
                for word in words:
                    if len(word) > 1 and word[0] == '<':
                        raise ValueError("Unexpected word '{}' in document {}".format(word, filename))
                words = [x for x in words if x]
                processed_lines.append(words)

            processed_lines = reprocess_lines(processed_lines)
            paragraphs = convert_processed_lines(processed_lines)

            test_documents.extend(paragraphs)
    print("Test document finished.")

    documents = []

    for filename in files:
        with open(filename) as fin:
            print("File:", filename)
            processed_lines = []
            for line in fin.readlines():
                line = clean_line(line)
                words = line.split("|")
                words = [clean_word(x) for x in words]
                for word in words:
                    if len(word) > 1 and word[0] == '<':
                        raise ValueError("Unexpected word '{}' in document {}".format(word, filename))
                words = [x for x in words if x]
                processed_lines.append(words)

            processed_lines = reprocess_lines(processed_lines)
            paragraphs = convert_processed_lines(processed_lines)

            documents.extend(paragraphs)

    print("All documents finished.")

    return documents, test_documents


def main(*args):
    random.seed(1000)
    if not args:
        args = sys.argv[1:]

    input_dir = args[0]
    full_input_dir = os.path.join(input_dir, "thai", "best")
    if os.path.exists(full_input_dir):
        # otherwise hopefully the user gave us the full path?
        input_dir = full_input_dir

    output_dir = args[1]
    documents, test_documents = read_data(input_dir)
    print("Finished reading data.")
    write_dataset_best(documents, test_documents, output_dir, "best")


if __name__ == '__main__':
    main()

