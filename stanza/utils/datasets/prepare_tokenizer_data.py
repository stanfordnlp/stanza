import argparse
import json
import os
import re
import sys

from collections import Counter

"""
Data is output in 4 files:

a file containing the mwt information
a file containing the words and sentences in conllu format
a file containing the raw text of each paragraph
a file of 0,1,2 indicating word break or sentence break on a character level for the raw text
  1: end of word
  2: end of sentence
"""

PARAGRAPH_BREAK = re.compile(r'\n\s*\n')

def is_para_break(index, text):
    """ Detect if a paragraph break can be found, and return the length of the paragraph break sequence. """
    if text[index] == '\n':
        para_break = PARAGRAPH_BREAK.match(text[index:])
        if para_break:
            break_len = len(para_break.group(0))
            return True, break_len
    return False, 0

def find_next_word(index, text, word, output):
    """
    Locate the next word in the text. In case a paragraph break is found, also write paragraph break to labels.
    """
    idx = 0
    word_sofar = ''
    while index < len(text) and idx < len(word):
        para_break, break_len = is_para_break(index, text)
        if para_break:
            # multiple newlines found, paragraph break
            if len(word_sofar) > 0:
                assert re.match(r'^\s+$', word_sofar), 'Found non-empty string at the end of a paragraph that doesn\'t match any token: |{}|'.format(word_sofar)
                word_sofar = ''

            output.write('\n\n')
            index += break_len - 1
        elif re.match(r'^\s$', text[index]) and not re.match(r'^\s$', word[idx]):
            # whitespace found, and whitespace is not part of a word
            word_sofar += text[index]
        else:
            # non-whitespace char, or a whitespace char that's part of a word
            word_sofar += text[index]
            assert text[index].replace('\n', ' ') == word[idx], "Character mismatch: raw text contains |%s| but the next word is |%s|." % (word_sofar, word)
            idx += 1
        index += 1
    return index, word_sofar

def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('plaintext_file', type=str, help="Plaintext file containing the raw input")
    parser.add_argument('conllu_file', type=str, help="CoNLL-U file containing tokens and sentence breaks")
    parser.add_argument('-o', '--output', default=None, type=str, help="Output file name; output to the console if not specified (the default)")
    parser.add_argument('-m', '--mwt_output', default=None, type=str, help="Output file name for MWT expansions; output to the console if not specified (the default)")

    args = parser.parse_args(args=args)

    with open(args.plaintext_file, 'r') as f:
        text = ''.join(f.readlines())
    textlen = len(text)

    if args.output is None:
        output = sys.stdout
    else:
        outdir = os.path.split(args.output)[0]
        os.makedirs(outdir, exist_ok=True)
        output = open(args.output, 'w')

    index = 0 # character offset in rawtext

    mwt_expansions = []
    with open(args.conllu_file, 'r') as f:
        buf = ''
        mwtbegin = 0
        mwtend = -1
        expanded = []
        last_comments = ""
        for line in f:
            line = line.strip()
            if len(line):
                if line[0] == "#":
                    # comment, don't do anything
                    if len(last_comments) == 0:
                        last_comments = line
                    continue

                line = line.split('\t')
                if '.' in line[0]:
                    # the tokenizer doesn't deal with ellipsis
                    continue

                word = line[1]
                if '-' in line[0]:
                    # multiword token
                    mwtbegin, mwtend = [int(x) for x in line[0].split('-')]
                    lastmwt = word
                    expanded = []
                elif mwtbegin <= int(line[0]) < mwtend:
                    expanded += [word]
                    continue
                elif int(line[0]) == mwtend:
                    expanded += [word]
                    expanded = [x.lower() for x in expanded] # evaluation doesn't care about case
                    mwt_expansions += [(lastmwt, tuple(expanded))]
                    if lastmwt[0].islower() and not expanded[0][0].islower():
                        print('Sentence ID with potential wrong MWT expansion: ', last_comments, file=sys.stderr)
                    mwtbegin = 0
                    mwtend = -1
                    lastmwt = None
                    continue

                if len(buf):
                    output.write(buf)
                index, word_found = find_next_word(index, text, word, output)
                buf = '0' * (len(word_found)-1) + ('1' if '-' not in line[0] else '3')
            else:
                # sentence break found
                if len(buf):
                    assert int(buf[-1]) >= 1
                    output.write(buf[:-1] + '{}'.format(int(buf[-1]) + 1))
                    buf = ''

                last_comments = ''

    output.close()

    mwts = Counter(mwt_expansions)
    if args.mwt_output is None:
        print('MWTs:', mwts)
    else:
        with open(args.mwt_output, 'w') as f:
            json.dump(list(mwts.items()), f)

        print('{} unique MWTs found in data'.format(len(mwts)))

if __name__ == '__main__':
    main(sys.argv[1:])
