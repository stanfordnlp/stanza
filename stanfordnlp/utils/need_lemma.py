import sys
from collections import Counter
from glob import glob

train_file = None

for f in glob("/u/nlp/data/dependency_treebanks/CoNLL18/{}/*.conllu".format(sys.argv[1])):
    if f.endswith('-train.conllu'):
        train_file = f

if train_file is None:
    print('{} N/A'.format(sys.argv[1]))
    exit()

N = 0
p = 0
with open(train_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or len(line) <= 0:
            continue

        line = line.split('\t')

        N += 1
        word = line[1]
        lemma = line[2]
        if word != lemma:
            p += 1

print("{} {:.3f}".format(sys.argv[1], 100 * p / N))
