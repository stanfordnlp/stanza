import argparse
import random
import sys

"""
Converts IJC data to a TSV format.

So far, tested on Hindi.  Not checked on any of the other languages.
"""

def convert_tag(tag):
    """
    Project the classes IJC used to 4 classes with more human-readable names

    The trained result is a pile, as I inadvertently taught my
    daughter to call horrible things, but leaving them with the
    original classes is also a pile
    """
    if not tag:
        return "O"
    if tag == "NEP":
        return "PER"
    if tag == "NEO":
        return "ORG"
    if tag == "NEL":
        return "LOC"
    return "MISC"

def read_single_file(input_file, bio_format=True):
    """
    Reads an IJC NER file and returns a list of list of lines
    """
    sentences = []
    lineno = 0
    with open(input_file) as fin:
        current_sentence = []
        in_ner = False
        in_sentence = False
        printed_first = False
        nesting = 0
        for line in fin:
            lineno = lineno + 1
            line = line.strip()
            if not line:
                continue
            if line.startswith("<Story") or line.startswith("</Story>"):
                assert not current_sentence, "File %s had an unexpected <Story> tag" % input_file
                continue

            if line.startswith("<Sentence"):
                assert not current_sentence, "File %s has a nested sentence" % input_file
                continue

            if line.startswith("</Sentence>"):
                # Would like to assert that empty sentences don't exist, but alas, they do
                # assert current_sentence, "File %s has an empty sentence at %d" % (input_file, lineno)
                # AssertionError: File .../hi_ijc/training-hindi/193.naval.utf8 has an empty sentence at 74
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = []
                continue

            if line == "))":
                assert in_sentence, "File %s closed a sentence when there was no open sentence at %d" % (input_file, lineno)
                nesting = nesting - 1
                if nesting < 0:
                    in_sentence = False
                    nesting = 0
                elif nesting == 0:
                    in_ner = False
                continue

            pieces = line.split("\t")
            if pieces[0] == '0':
                assert pieces[1] == '((', "File %s has an unexpected first line at %d" % (input_file, lineno)
                in_sentence = True
                continue

            if pieces[1] == '((':
                nesting = nesting + 1
                if nesting == 1:
                    if len(pieces) < 4:
                        tag = None
                    else:
                        assert pieces[3][0] == '<' and pieces[3][-1] == '>', "File %s has an unexpected tag format at %d: %s" % (input_file, lineno, pieces[3])
                        ne, tag = pieces[3][1:-1].split('=', 1)
                        assert pieces[3] == "<%s=%s>" % (ne, tag), "File %s has an unexpected tag format at %d: %s" % (input_file, lineno, pieces[3])
                        in_ner = True
                        printed_first = False
                        tag = convert_tag(tag)
            elif in_ner and tag:
                if bio_format:
                    if printed_first:
                        current_sentence.append((pieces[1], "I-" + tag))
                    else:
                        current_sentence.append((pieces[1], "B-" + tag))
                        printed_first = True
                else:
                    current_sentence.append((pieces[1], tag))
            else:
                current_sentence.append((pieces[1], "O"))
    assert not current_sentence, "File %s is unclosed!" % input_file
    return sentences

def read_ijc_files(input_files, bio_format=True):
    sentences = []
    for input_file in input_files:
        sentences.extend(read_single_file(input_file, bio_format))
    return sentences

def convert_ijc(input_files, csv_file, bio_format=True):
    sentences = read_ijc_files(input_files, bio_format)
    with open(csv_file, "w") as fout:
        for sentence in sentences:
            for word in sentence:
                fout.write("%s\t%s\n" % word)
            fout.write("\n")

def convert_split_ijc(input_files, train_csv, dev_csv):
    """
    Randomly splits the given list of input files into a train/dev with 85/15 split

    The original datasets only have train & test
    """
    random.seed(1234)
    train_files = []
    dev_files = []
    for filename in input_files:
        if random.random() < 0.85:
            train_files.append(filename)
        else:
            dev_files.append(filename)

    if len(train_files) == 0 or len(dev_files) == 0:
        raise RuntimeError("Not enough files to split into train & dev")

    convert_ijc(train_files, train_csv)
    convert_ijc(dev_files, dev_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="/home/john/stanza/data/ner/hi_ijc.test.csv", help="Where to output the results")
    parser.add_argument('input_files', metavar='N', nargs='+', help='input files to process')
    args = parser.parse_args()

    convert_ijc(args.input_files, args.output_path, False)
