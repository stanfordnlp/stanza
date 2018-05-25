import argparse
import re
import sys

parser = argparse.ArgumentParser()

parser.add_argument('plaintext_file', type=str, help="Plaintext file containing the raw input")
parser.add_argument('conllu_file', type=str, help="CoNLL-U file containing tokens and sentence breaks")
parser.add_argument('-o', '--output', default=None, type=str, help="Output file name; output to the console if not specified (the default)")

args = parser.parse_args()

with open(args.plaintext_file, 'r') as f:
    text = ''.join(f.readlines())
textlen = len(text)

output = sys.stdout if args.output is None else open(args.output, 'w')

index = 0 # character offset in rawtext

def find_next_word(index, text, word, output):
    while index < len(text) and text[index] != word[0]:
        if text[index] == '\n' and index+1 < len(text) and text[index+1] == '\n':
            # paragraph break
            output.write('\n\n')
            index += 1
        elif re.match(r'^\s$', text[index]):
            output.write('0') # not word boundary
        else:
            assert False, "character mismatch: raw text contains |%s| but the next word is |%s|." % (text[index:(index+20)], word)
        index += 1
    return index

mwt_expansions = []
with open(args.conllu_file, 'r') as f:
    buf = ''
    mwtbegin = 0
    mwtend = -1
    expanded = []
    for line in f:
        line = line.strip()
        if len(line):
            if line[0] == "#":
                # comment, don't do anything
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
                mwt_expansions += [(lastmwt, tuple(expanded))]
                mwtbegin = 0
                mwtend = -1
                lastmwt = None
                continue

            if len(buf):
                output.write(buf)
            index = find_next_word(index, text, word, output)
            assert text[index:(index+len(word))].replace('\n', ' ') == word, "word mismatch: raw text contains |%s| but the next word is |%s|." % (text[index:(index+len(word))], word)
            buf = '0' * (len(word)-1) + ('1' if '-' not in line[0] else '3')
            index += len(word)
        else:
            # sentence break found
            if len(buf):
                assert buf[-1] == '1'
                output.write(buf[:-1] + '2')
                buf = ''

from collections import Counter
print('MWTs: ', Counter(mwt_expansions))

output.close()
