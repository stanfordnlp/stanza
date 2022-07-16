import argparse
from collections import defaultdict
import os
import re
import sys
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.common.constant import treebank_to_short_name
from stanza.models.pos.vocab import XPOSVocab, WordVocab
from stanza.models.common.doc import *
from stanza.utils.conll import CoNLL
from stanza.utils import default_paths

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")
DATA_DIR = default_paths.get_default_paths()['POS_DATA_DIR']
DEFAULT_KEY = 'WordVocab(data, shorthand, idx=2, ignore=["_"])'

def filter_data(data, idx):
    data_filtered = []
    for sentence in data:
        flag = True
        for token in sentence:
            if token[idx] is None:
                flag = False
        if flag: data_filtered.append(sentence)
    return data_filtered

def get_factory(sh, fn):
    print('Resolving vocab option for {}...'.format(sh))
    train_file = os.path.join(DATA_DIR, '{}.train.in.conllu'.format(sh))
    if not os.path.exists(train_file):
        raise UserWarning('Training data for {} not found in the data directory, falling back to using WordVocab. To generate the '
                          'XPOS vocabulary for this treebank properly, please run the following command first:\n'
                          '\tstanza/utils/datasets/prepare_pos_treebank.py {}'.format(fn, fn))
        # without the training file, there's not much we can do
        key = DEFAULT_KEY
        return key

    doc = CoNLL.conll2doc(input_file=train_file)
    data = doc.get([TEXT, UPOS, XPOS, FEATS], as_sentences=True)
    print(f'Original length = {len(data)}')
    data = filter_data(data, idx=2)
    print(f'Filtered length = {len(data)}')
    vocab = WordVocab(data, sh, idx=2, ignore=["_"])
    key = DEFAULT_KEY
    best_size = len(vocab) - len(VOCAB_PREFIX)
    if best_size > 20:
        for sep in ['', '-', '+', '|', ',', ':']: # separators
            vocab = XPOSVocab(data, sh, idx=2, sep=sep)
            length = sum(len(x) - len(VOCAB_PREFIX) for x in vocab._id2unit.values())
            if length < best_size:
                key = 'XPOSVocab(data, shorthand, idx=2, sep="{}")'.format(sep)
                best_size = length
    return key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebanks', type=str, default=DATA_DIR, help="Treebanks to process - directory with processed datasets or a file with a list")
    parser.add_argument('--output_file', type=str, default="stanza/models/pos/xpos_vocab_factory.py", help="Where to write the results")
    args = parser.parse_args()

    output_file = args.output_file
    if os.path.isdir(args.treebanks):
        # if the path is a directory of datasets (which is the default if  is set)
        # we use those datasets to prepare the xpos factories
        treebanks = os.listdir(args.treebanks)
        treebanks = [x.split(".", maxsplit=1)[0] for x in treebanks]
        treebanks = sorted(set(treebanks))
    elif os.path.exists(args.treebanks):
        # maybe it's a file with a list of names
        with open(args.treebanks) as fin:
            treebanks = sorted(set([x.strip() for x in fin.readlines() if x.strip()]))
    else:
        raise ValueError("Cannot figure out which treebanks to use.   Please set the --treebanks parameter")

    print("Processing the following treebanks: %s" % " ".join(treebanks))

    shorthands = []
    fullnames = []
    for treebank in treebanks:
        fullnames.append(treebank)
        if SHORTNAME_RE.match(treebank):
            shorthands.append(treebank)
        else:
            shorthands.append(treebank_to_short_name(treebank))

    # For each treebank, we would like to find the XPOS Vocab configuration that minimizes
    # the number of total classes needed to predict by all tagger classifiers. This is
    # achieved by enumerating different options of separators that different treebanks might
    # use, and comparing that to treating the XPOS tags as separate categories (using a
    # WordVocab).
    mapping = defaultdict(list)
    for sh, fn in zip(shorthands, fullnames):
        factory = get_factory(sh, fn)
        mapping[factory].append(sh)
        if sh == 'zh-hans_gsdsimp':
            mapping[factory].append('zh_gsdsimp')
        elif sh == 'no_bokmaal':
            mapping[factory].append('nb_bokmaal')

    mapping[DEFAULT_KEY].append('en_test')

    # Generate code. This takes the XPOS vocabulary classes selected above, and generates the
    # actual factory class as seen in models.pos.xpos_vocab_factory.
    first = True
    with open(output_file, 'w') as f:
        print('''# This is the XPOS factory method generated automatically from stanza.models.pos.build_xpos_vocab_factory.
# Please don't edit it!

from stanza.models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(data, shorthand):''', file=f)

        for key in mapping:
            print("    {} shorthand in [{}]:".format('if' if first else 'elif', ', '.join(['"{}"'.format(x) for x in sorted(mapping[key])])), file=f)
            print("        return {}".format(key), file=f)

            first = False
        print('''    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))''', file=f)

    print('Done!')

if __name__ == "__main__":
    main()
