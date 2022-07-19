import argparse
from collections import defaultdict
import logging
import os
import re
import sys
from stanza.models.common.constant import treebank_to_short_name
from stanza.models.pos.xpos_vocab_utils import DEFAULT_KEY, choose_simplest_factory, XPOSType
from stanza.models.common.doc import *
from stanza.utils.conll import CoNLL
from stanza.utils import default_paths

SHORTNAME_RE = re.compile("[a-z-]+_[a-z0-9]+")
DATA_DIR = default_paths.get_default_paths()['POS_DATA_DIR']

logger = logging.getLogger('stanza')

def get_xpos_factory(shorthand, fn):
    logger.info('Resolving vocab option for {}...'.format(shorthand))
    train_file = os.path.join(DATA_DIR, '{}.train.in.conllu'.format(shorthand))
    if not os.path.exists(train_file):
        raise UserWarning('Training data for {} not found in the data directory, falling back to using WordVocab. To generate the '
                          'XPOS vocabulary for this treebank properly, please run the following command first:\n'
                          '\tstanza/utils/datasets/prepare_pos_treebank.py {}'.format(fn, fn))
        # without the training file, there's not much we can do
        key = DEFAULT_KEY
        return key

    doc = CoNLL.conll2doc(input_file=train_file)
    data = doc.get([TEXT, UPOS, XPOS, FEATS], as_sentences=True)
    return choose_simplest_factory(data, shorthand)

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

    logger.info("Processing the following treebanks: %s" % " ".join(treebanks))

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
        factory = get_xpos_factory(sh, fn)
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
        max_len = max(max(len(x) for x in mapping[key]) for key in mapping)
        print('''# This is the XPOS factory method generated automatically from stanza.models.pos.build_xpos_vocab_factory.
# Please don't edit it!

import logging

from stanza.models.pos.vocab import WordVocab, XPOSVocab
from stanza.models.pos.xpos_vocab_utils import XPOSDescription, XPOSType, build_xpos_vocab, choose_simplest_factory

# using a sublogger makes it easier to test in the unittests
logger = logging.getLogger('stanza.models.pos.xpos_vocab_factory')

XPOS_DESCRIPTIONS = {''', file=f)

        for key_idx, key in enumerate(mapping):
            if key_idx > 0:
                print(file=f)
            for shorthand in sorted(mapping[key]):
                # +2 to max_len for the ''
                # this format string is left justified (either would be okay, probably)
                if key.sep is None:
                    sep = 'None'
                else:
                    sep = "'%s'" % key.sep
                print(("    {:%ds}: XPOSDescription({}, {})," % (max_len+2)).format("'%s'" % shorthand, key.xpos_type, sep), file=f)

        print('''}

def xpos_vocab_factory(data, shorthand):
    if shorthand not in XPOS_DESCRIPTIONS:
        logger.warning("%s is not a known dataset.  Examining the data to choose which xpos vocab to use", shorthand)
    desc = choose_simplest_factory(data, shorthand)
    if shorthand in XPOS_DESCRIPTIONS:
        if XPOS_DESCRIPTIONS[shorthand] != desc:
            # log instead of throw
            # otherwise, updating datasets would be unpleasant
            logger.error("XPOS tagset in %s has apparently changed!  Was %s, is now %s", shorthand, XPOS_DESCRIPTIONS[shorthand], desc)
    return build_xpos_vocab(desc, data, shorthand)
''', file=f)

    logger.info('Done!')

if __name__ == "__main__":
    main()
