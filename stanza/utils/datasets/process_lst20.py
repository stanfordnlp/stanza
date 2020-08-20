"""Processes the tokenization section of the LST20 Thai dataset

The dataset is available here:

https://aiforthai.in.th/corpus.php


python3 -m stanza.utils.datasets.process_lst20 extern_data/thai/LST20_Corpus data/tokenize

Unlike Orchid and BEST, LST20 has train/eval/test splits, which we relabel train/dev/test.

./scripts/run_tokenize.sh UD_Thai-lst20 --dropout 0.05 --unit_dropout 0.05
"""


import glob
import os
import sys

from stanza.utils.datasets.process_thai_tokenization import write_section

def read_data(input_dir, section):
    input_dir = os.path.join(input_dir, section)
    filenames = glob.glob(os.path.join(input_dir, "*.txt"))
    documents = []
    for filename in filenames:
        document = []
        lines = open(filename).readlines()
        sentence = []
        for line in lines:
            line = line.strip()
            if not line:
                if sentence:
                    #sentence[-1] = (sentence[-1][0], True)
                    document.append(sentence)
                    sentence = []
            else:
                pieces = line.split("\t")
                if pieces[0] == '_':
                    sentence[-1] = (sentence[-1][0], True)
                else:
                    sentence.append((pieces[0], False))

        if sentence:
            #sentence[-1] = (sentence[-1][0], True)
            document.append(sentence)
            sentence = []
        # TODO: is there any way to divide up a single document into paragraphs?
        documents.append([document])
    return documents

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    for (in_section, out_section) in (("train", "train"),
                                      ("eval", "dev"),
                                      ("test", "test")):
        documents = read_data(input_dir, in_section)
        write_section(output_dir, "lst20", out_section, documents)


if __name__ == '__main__':
    main()
