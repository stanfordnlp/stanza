import csv
import glob
import json
import os
import tempfile

from collections import namedtuple

Fragment = namedtuple('Fragment', ['sentiment', 'text'])
Split = namedtuple('Split', ['filename', 'weight'])

def write_list(out_filename, dataset):
    """
    Write a list of items to the given output file

    Expected: list(Fragment)
    """
    formatted_dataset = [{'sentiment': line.sentiment, 'text': line.text} for line in dataset]
    # Rather than write the dataset at once, we write one line at a time
    # Using `indent` puts each word on a separate line, which is rather noisy,
    # but not formatting at all makes one long line out of an entire dataset,
    # which is impossible to read
    #json.dump(formatted_dataset, fout, indent=2, ensure_ascii=False)

    with open(out_filename, 'w') as fout:
        fout.write("[\n")
        for idx, line in enumerate(formatted_dataset):
            fout.write("  ")
            json.dump(line, fout, ensure_ascii=False)
            if idx < len(formatted_dataset) - 1:
                fout.write(",")
            fout.write("\n")
        fout.write("]\n")

def write_splits(out_directory, snippets, splits):
    """
    Write the given list of items to the split files in the specified output directory
    """
    total_weight = sum(split.weight for split in splits)
    divs = []
    subtotal = 0.0
    for split in splits:
        divs.append(int(len(snippets) * subtotal / total_weight))
        subtotal = subtotal + split.weight
    # the last div will be guaranteed to be the full thing - no math used
    divs.append(len(snippets))

    for i, split in enumerate(splits):
        filename = os.path.join(out_directory, split.filename)
        print("Writing {}:{} to {}".format(divs[i], divs[i+1], filename))
        write_list(filename, snippets[divs[i]:divs[i+1]])

def clean_tokenized_tweet(line):
    line = list(line)
    if len(line) > 3 and line[0] == 'RT' and line[1][0] == '@' and line[2] == ':':
        line = line[3:]
    elif len(line) > 4 and line[0] == 'RT' and line[1] == '@' and line[3] == ':':
        line = line[4:]
    elif line[0][0] == '@':
        line = line[1:]
    for i in range(len(line)):
        if line[i][0] == '@' or line[i][0] == '#':
            line[i] = line[i][1:]
        if line[i].startswith("http:") or line[i].startswith("https:"):
            line[i] = ' '
    return line

def get_ptb_tokenized_phrases(fragments):
    """
    Use the PTB tokenizer to retokenize the phrases

    Not clear which is better, "Nov." or "Nov ."
    strictAcronym=true makes it do the latter
    tokenizePerLine=true should make it only pay attention to one line at a time

    Phrases will be returned as lists of words rather than one string
    """
    with tempfile.TemporaryDirectory() as tempdir:
        phrase_filename = os.path.join(tempdir, "phrases.txt")
        #phrase_filename = "asdf.txt"
        with open(phrase_filename, "w", encoding="utf-8") as fout:
            for fragment in fragments:
                # extra newlines are so the tokenizer treats the lines
                # as separate sentences
                fout.write("%s\n\n\n" % (fragment.text))
        tok_filename = os.path.join(tempdir, "tokenized.txt")
        os.system('java edu.stanford.nlp.process.PTBTokenizer -options "strictAcronym=true,tokenizePerLine=true" -preserveLines %s > %s' % (phrase_filename, tok_filename))
        with open(tok_filename, encoding="utf-8") as fin:
            tokenized = fin.readlines()

    tokenized = [x.strip() for x in tokenized]
    tokenized = [x for x in tokenized if x]
    phrases = [Fragment(x.sentiment, y.split()) for x, y in zip(fragments, tokenized)]
    return phrases

