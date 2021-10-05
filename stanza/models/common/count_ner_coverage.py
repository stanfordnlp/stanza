from stanza.models.common import pretrain
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ners', type=str, nargs='*', help='Which treebanks to run on')
    parser.add_argument('--pretrain', type=str, default="/home/john/stanza_resources/hi/pretrain/hdtb.pt", help='Which pretrain to use')
    parser.set_defaults(ners=["/home/john/stanza/data/ner/hi_fire2013.train.csv",
                              "/home/john/stanza/data/ner/hi_fire2013.dev.csv"])
    args = parser.parse_args()
    return args


def read_ner(filename):
    words = []
    for line in open(filename).readlines():
        line = line.strip()
        if not line:
            continue
        if line.split("\t")[1] == 'O':
            continue
        words.append(line.split("\t")[0])
    return words

def count_coverage(pretrain, words):
    count = 0
    for w in words:
        if w in pretrain.vocab:
            count = count + 1
    return count / len(words)

args = parse_args()
pt = pretrain.Pretrain(args.pretrain)
for dataset in args.ners:
    words = read_ner(dataset)
    print(dataset)
    print(count_coverage(pt, words))
    print()
