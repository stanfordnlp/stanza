"""
Randomly split a file into train, dev, and test sections

Specifically used in the case of building a tagger from the initial
POS tagging provided by Isra, but obviously can be used to split any
conllu file
"""

import argparse
import os
import random

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
from stanza.utils.default_paths import get_default_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='extern_data/sindhi/upos/sindhi_upos.conllu', help='Which file to split')
    parser.add_argument('--train', type=float, default=0.8, help='Fraction of the data to use for train')
    parser.add_argument('--dev', type=float, default=0.1, help='Fraction of the data to use for dev')
    parser.add_argument('--test', type=float, default=0.1, help='Fraction of the data to use for test')
    parser.add_argument('--seed', default='1234', help='Random seed to use')
    parser.add_argument('--short_name', default='sd_isra', help='Dataset name to use when writing output files')
    parser.add_argument('--no_remove_xpos', default=True, action='store_false', dest='remove_xpos', help='By default, we remove the xpos from the dataset')
    parser.add_argument('--no_remove_feats', default=True, action='store_false', dest='remove_feats', help='By default, we remove the feats from the dataset')
    parser.add_argument('--output_directory', default=get_default_paths()["POS_DATA_DIR"], help="Where to put the split conllu")
    args = parser.parse_args()

    weights = (args.train, args.dev, args.test)

    doc = CoNLL.conll2doc(args.filename)
    random.seed(args.seed)

    train_doc = ([], [])
    dev_doc = ([], [])
    test_doc = ([], [])
    splits = [train_doc, dev_doc, test_doc]
    for sentence in doc.sentences:
        sentence_dict = sentence.to_dict()
        if args.remove_xpos:
            for x in sentence_dict:
                x.pop('xpos', None)
        if args.remove_feats:
            for x in sentence_dict:
                x.pop('feats', None)
        split = random.choices(splits, weights)[0]
        split[0].append(sentence_dict)
        split[1].append(sentence.comments)

    splits = [Document(split[0], comments=split[1]) for split in splits]
    for split_doc, split_name in zip(splits, ("train", "dev", "test")):
        filename = os.path.join(args.output_directory, "%s.%s.in.conllu" % (args.short_name, split_name))
        print("Outputting %d sentences to %s" % (len(split_doc.sentences), filename))
        CoNLL.write_doc2conll(split_doc, filename)

if __name__ == '__main__':
    main()

