from collections import defaultdict
import os
import sys
from stanfordnlp.models.common.vocab import VOCAB_PREFIX
from stanfordnlp.models.pos.vocab import XPOSVocab, WordVocab
from stanfordnlp.models.common.conll import CoNLLFile

if len(sys.argv) != 3:
    print('Usage: {} short_to_tb_file output_factory_file'.format(sys.argv[0]))
    sys.exit(0)

# Read list of all treebanks of concern
short_to_tb_file, output_file = sys.argv[1:]

shorthands = []
fullnames = []
with open(short_to_tb_file) as f:
    for line in f:
        line = line.strip().split()
        shorthands.append(line[0])
        fullnames.append(line[1])

# For each treebank, we would like to find the XPOS Vocab configuration that minimizes
# the number of total classes needed to predict by all tagger classifiers. This is
# achieved by enumerating different options of separators that different treebanks might
# use, and comparing that to treating the XPOS tags as separate categories (using a
# WordVocab).
mapping = defaultdict(list)
for sh, fn in zip(shorthands, fullnames):
    print('Resolving vocab option for {}...'.format(sh))
    if not os.path.exists('data/pos/{}.train.in.conllu'.format(sh)):
        raise UserWarning('Training data for {} not found in the data directory, falling back to using WordVocab. To generate the '
            'XPOS vocabulary for this treebank properly, please run the following command first:\n'
            '\tbash scripts/prep_pos_data.sh {}'.format(fn, fn))
        # without the training file, there's not much we can do
        key = 'WordVocab(data, shorthand, idx=2)'
        mapping[key].append(sh)
        continue

    conll_file = CoNLLFile('data/pos/{}.train.in.conllu'.format(sh))
    data = conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
    vocab = WordVocab(data, sh, idx=2, ignore=["_"])
    key = 'WordVocab(data, shorthand, idx=2, ignore=["_"])'
    best_size = len(vocab) - len(VOCAB_PREFIX)
    if best_size > 20:
        for sep in ['', '-', '+', '|', ',', ':']: # separators
            vocab = XPOSVocab(data, sh, idx=2, sep=sep)
            length = sum(len(x) - len(VOCAB_PREFIX) for x in vocab._id2unit.values())
            if length < best_size:
                key = 'XPOSVocab(data, shorthand, idx=2, sep="{}")'.format(sep)
                best_size = length
    mapping[key].append(sh)

# Generate code. This takes the XPOS vocabulary classes selected above, and generates the
# actual factory class as seen in models.pos.xpos_vocab_factory.
first = True
with open(output_file, 'w') as f:
    print('''# This is the XPOS factory method generated automatically from models.pos.build_xpos_factory.
# Please don't edit it!

from models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(data, shorthand):''', file=f)

    for key in mapping:
        print("    {} shorthand in [{}]:".format('if' if first else 'elif', ', '.join(['"{}"'.format(x) for x in mapping[key]])), file=f)
        print("        return {}".format(key), file=f)

        first = False
    print('''    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))''', file=f)

print('Done!')
