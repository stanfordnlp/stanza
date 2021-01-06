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

python3 -m stanza.utils.datasets.process_best extern_data/thai/best data/tokenize
./scripts/run_tokenize.sh UD_Thai-best --dropout 0.05 --unit_dropout 0.05 --steps 50000
"""

import glob
import os
import random
import re
import sys

from pythainlp import sent_tokenize

from stanza.utils.datasets.process_thai_tokenization import write_dataset

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

def reprocess_lines(processed_lines):
    reprocessed_lines = []
    for line in processed_lines:
        text = "".join(line)
        chunks = sent_tokenize(text)
        if sum(len(x) for x in chunks) != len(text):
            raise ValueError("Got unexpected text length: \n{}\nvs\n{}".format(text, chunks))

        chunk_lengths = [len(x) for x in chunks]

        current_length = 0
        new_line = []
        for word in line:
            if len(word) + current_length < chunk_lengths[0]:
                new_line.append(word)
                current_length = current_length + len(word)
            elif len(word) + current_length == chunk_lengths[0]:
                new_line.append(word)
                reprocessed_lines.append(new_line)
                new_line = []
                chunk_lengths = chunk_lengths[1:]
                current_length = 0
            else:
                remaining_len = chunk_lengths[0] - current_length
                new_line.append(word[:remaining_len])
                reprocessed_lines.append(new_line)
                word = word[remaining_len:]
                chunk_lengths = chunk_lengths[1:]
                while len(word) > chunk_lengths[0]:
                    new_line = [word[:chunk_lengths[0]]]
                    reprocessed_lines.append(new_line)
                    word = word[chunk_lengths[0]:]
                    chunk_lengths = chunk_lengths[1:]
                new_line = [word]
                current_length = len(word)
        reprocessed_lines.append(new_line)
    return reprocessed_lines

def read_data(input_dir):
    subdirs = [os.path.join(input_dir, 'article'),
               os.path.join(input_dir, 'encyclopedia'),
               os.path.join(input_dir, 'news'),
               os.path.join(input_dir, 'novel')]
    files = []
    for subdir in subdirs:
        if not os.path.exists(subdir):
            raise FileNotFoundError("Expected a directory that did not exist: {}".format(subdir))
        files.extend(glob.glob(os.path.join(subdir, '*.txt')))

    documents = []
    for filename in files:
        with open(filename) as fin:
            sentences = []
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

            for words in processed_lines:
                # turn the words into a sentence
                sentence = []
                for word in words:
                    word = word.strip()
                    if not word:
                        if len(sentence) == 0:
                            raise ValueError("Unexpected space at start of sentence in document {}".format(filename))
                        sentence[-1] = (sentence[-1][0], True)
                    else:
                        sentence.append((word, False))
                # blank lines are very rare in best, but why not treat them as a paragraph break
                if len(sentence) == 0:
                    paragraphs = [sentences]
                    documents.append(paragraphs)
                    sentences = []
                    continue
                sentence[-1] = (sentence[-1][0], True)
                sentences.append(sentence)
            paragraphs = [sentences]
            documents.append(paragraphs)

    return documents

def main():
    random.seed(1000)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    documents = read_data(input_dir)
    write_dataset(documents, output_dir, "best")


if __name__ == '__main__':
    main()
