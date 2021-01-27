"""
Turns a directory of conllu files from the conll 2017 shared task to a text file

Part of the process for building a charlm dataset

python conll17_to_text.py <directory>

Extension of this script:
  https://github.com/stanfordnlp/stanza-scripts/blob/master/charlm/conll17/conll2txt.py
"""

import lzma
import sys
import os

def process_file(input_filename):
    if not input_filename.endswith('.conllu') and not input_filename.endswith(".conllu.xz"):
        print("Skipping {}".format(input_filename))
        return

    if input_filename.endswith(".xz"):
        open_fn = lambda x: lzma.open(x, mode='rt')
        output_filename = input_filename[:-3].replace(".conllu", ".txt")
    else:
        open_fn = lambda x: open(x)
        output_filename = input_filename.replace('.conllu', '.txt')
    if os.path.exists(output_filename):
        print("Cowardly refusing to overwrite %s" % output_filename)
        return

    print("Converting %s to %s" % (input_filename, output_filename))
    with open_fn(input_filename) as fin:
        sentences = []
        sentence = []
        for line in fin:
            line = line.strip()
            if len(line) == 0: # new sentence
                sentences.append(sentence)
                sentence = []
                continue
            if line[0] == '#': # comment
                continue
            splitline = line.split('\t')
            assert(len(splitline) == 10) # correct conllu
            id, word = splitline[0], splitline[1]
            if '-' not in id: # not mwt token
                sentence.append(word)

    if sentence:
        sentences.append(sentence)

    print(len(sentences))
    with open(output_filename, 'w') as fout:
        fout.write('\n'.join([' '.join(sentence) for sentence in sentences]))

if __name__ == '__main__':
    directory = sys.argv[1]
    filenames = sorted(os.listdir(directory))
    print("Files to process in {}: {}".format(directory, filenames))

    for filename in filenames:
        process_file(os.path.join(directory, filename))

