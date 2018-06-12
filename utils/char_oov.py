import sys
from collections import Counter
from glob import glob

train_file = None
dev_file = None

for f in glob("/u/nlp/data/dependency_treebanks/CoNLL18/{}/*.conllu".format(sys.argv[1])):
    if f.endswith('-train.conllu'):
        train_file = f
    elif f.endswith('-dev.conllu'):
        dev_file = f

if train_file is None or dev_file is None:
    print('{} N/A'.format(sys.argv[1]))
    exit()

def read_words_lemmas(filename):
    words = ''
    lemmas = ''
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) <= 0:
                continue

            words += line[1]
            lemmas += line[2]

    return words, lemmas

words, lemmas = read_words_lemmas(train_file)
wvocab = Counter(words+lemmas)

words, lemmas = read_words_lemmas(dev_file)
dwvocab = Counter(words+lemmas)

woov = sum([dwvocab[k] for k in dwvocab if k not in wvocab])

print("{} {:.3f}".format(sys.argv[1], 100 * woov / sum(dwvocab.values())))
