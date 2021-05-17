"""A simple script to count the fraction of words in a UD dataset which are in a particular pretrain.

For example, this script shows that the word2vec Armenian vectors,
truncated at 250K words, have 75% coverage of the Western Armenian
dataset, whereas the vectors available here have 88% coverage:

https://github.com/ispras-texterra/word-embeddings-eval-hy
"""

from stanza.models.common import pretrain
from stanza.utils.conll import CoNLL

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('treebanks', type=str, nargs='*', help='Which treebanks to run on')
    parser.add_argument('--pretrain', type=str, default="/home/john/extern_data/wordvec/glove/armenian.pt", help='Which pretrain to use')
    parser.set_defaults(treebanks=["/home/john/extern_data/ud2/ud-treebanks-v2.7/UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-train.conllu",
                                   "/home/john/extern_data/ud2/ud-treebanks-v2.7/UD_Armenian-ArmTDP/hy_armtdp-ud-train.conllu"])
    args = parser.parse_args()
    return args


args = parse_args()
pt = pretrain.Pretrain(args.pretrain)
pt.load()
print("Pretrain stats: {} vectors, {} dim".format(len(pt.vocab), pt.emb[0].shape[0]))

for treebank in args.treebanks:
    print(treebank)
    found = 0
    total = 0
    doc = CoNLL.conll2doc(treebank)
    for sentence in doc.sentences:
        for word in sentence.words:
            total = total + 1
            if word.text in pt.vocab:
                found = found + 1

    print (found / total)
