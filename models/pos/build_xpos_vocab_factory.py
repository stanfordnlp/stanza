from collections import defaultdict
import os
import sys
from models.common.vocab import VOCAB_PREFIX
from models.pos.vocab import XPOSVocab, WordVocab
from models.common.conll import CoNLLFile

if len(sys.argv) != 3:
    print('Usage: {} short_to_tb_file output_factory_file'.format(sys.argv[0]))
    sys.exit(0)

short_to_tb_file, output_file = sys.argv[1:]

shorthands = []
with open(short_to_tb_file) as f:
    for line in f:
        line = line.strip().split()
        shorthands.append(line[0])

tempfile = 'temp.xpos.vocab'
mapping = defaultdict(list)
for sh in shorthands:
    print('Resolving vocab option for {}...'.format(sh))
    if not os.path.exists('data/pos/{}.train.in.conllu'.format(sh)):
        # without the training file, there's not much we can do
        key = 'WordVocab(vocabfile, data, shorthand, idx=2)'
        mapping[key].append(sh)
        continue

    conll_file = CoNLLFile('data/pos/{}.train.in.conllu'.format(sh))
    data = conll_file.get(['word', 'upos', 'xpos', 'feats'], as_sentences=True)
    vocab = WordVocab(tempfile, data, sh, idx=2)
    os.remove(tempfile)
    key = 'WordVocab(vocabfile, data, shorthand, idx=2)'
    best_size = len(vocab) - len(VOCAB_PREFIX)
    if best_size > 20:
        for sep in ['', '-', '+', '|', ',', ':']: # separators
            vocab = XPOSVocab(tempfile, data, sh, idx=2, sep=sep)
            os.remove(tempfile)
            length = sum(len(x) - len(VOCAB_PREFIX) for x in vocab._id2unit.values())
            if length < best_size:
                key = 'XPOSVocab(vocabfile, data, shorthand, idx=2, sep="{}")'.format(sep)
                best_size = length
    mapping[key].append(sh)

# generate code
first = True
with open(output_file, 'w') as f:
    print('''# This is the XPOS factory method generated automatically from models.pos.build_xpos_factory.
# Please don't edit it!

from models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(vocabfile, data, shorthand):''', file=f)

    for key in mapping:
        print("    {} shorthand in [{}]:".format('if' if first else 'elif', ', '.join(['"{}"'.format(x) for x in mapping[key]])), file=f)
        print("        return {}".format(key), file=f)

        first = False
    print('''    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))''', file=f)

print('Done!')
