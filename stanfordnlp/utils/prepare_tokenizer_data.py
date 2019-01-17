import argparse
import re
import sys

parser = argparse.ArgumentParser()

parser.add_argument('plaintext_file', type=str, help="Plaintext file containing the raw input")
parser.add_argument('conllu_file', type=str, help="CoNLL-U file containing tokens and sentence breaks")
parser.add_argument('-o', '--output', default=None, type=str, help="Output file name; output to the console if not specified (the default)")
parser.add_argument('-m', '--mwt_output', default=None, type=str, help="Output file name for MWT expansions; output to the console if not specified (the default)")

args = parser.parse_args()

with open(args.plaintext_file, 'r') as f:
    text = ''.join(f.readlines())
textlen = len(text)

output = sys.stdout if args.output is None else open(args.output, 'w')

index = 0 # character offset in rawtext

def find_next_word(index, text, word, output):
    idx = 0
    word_sofar = ''
    yeah=False
    while index < len(text) and idx < len(word):
        if text[index] == '\n' and index+1 < len(text) and text[index+1] == '\n':
            # paragraph break
            if len(word_sofar) > 0:
                assert re.match(r'^\s+$', word_sofar), 'Found non-empty string at the end of a paragraph that doesn\'t match any token: |{}|'.format(word_sofar)
                word_sofar = ''

            output.write('\n\n')
            index += 1
        elif re.match(r'^\s$', text[index]) and not re.match(r'^\s$', word[idx]):
            word_sofar += text[index]
        else:
            word_sofar += text[index]
            assert text[index].replace('\n', ' ') == word[idx], "character mismatch: raw text contains |%s| but the next word is |%s|." % (word_sofar, word)
            idx += 1
        index += 1
    return index, word_sofar

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

from collections import Counter
mwts = Counter(mwt_expansions)
if args.mwt_output is None:
    print('MWTs:', mwts)
else:
    import json
    with open(args.mwt_output, 'w') as f:
        json.dump(list(mwts.items()), f)

    print('{} unique MWTs found in data'.format(len(mwts)))
