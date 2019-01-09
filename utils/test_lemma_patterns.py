"""
Test whether a regex pattern on lemma can recognize identity mappings on UD train/dev conllu data.
"""
import re
import pickle as pkl
import glob
import random
from models.common import conll

UDDIR='/u/nlp/data/dependency_treebanks/CoNLL18'
LONG_NUMBER_PATTERN = re.compile('^\d{5,}$')
DECIMAL_PATTERN = re.compile('^\d*(\.|,)?\d+$')
URL_PATTERN = re.compile('^http|^www')
EMAIL_PATTERN = re.compile('@')
CAPITAL_PATTERN = re.compile('^[A-Z]+$')

def prepare_data():
    pairs = []
    for fn in glob.glob(UDDIR + "/UD_*/*-ud-dev.conllu"):
        conllfile = conll.CoNLLFile(fn)
        ps = conllfile.get(["word", "lemma"])
        pairs += ps
    print("{} total word-lemma pairs loaded.".format(len(pairs)))

    with open('dev-wl-pairs.pkl', 'wb') as outfile:
        pkl.dump(pairs, outfile)

def load_data():
    with open('dev-wl-pairs.pkl', 'rb') as infile:
        pairs = pkl.load(infile)
    print("{} total word-lemma pairs loaded.".format(len(pairs)))
    return pairs

def test(pairs, pattern):
    matched = 0
    equal = 0
    caseless_equal = 0
    exceptions = []
    for w, l in pairs:
        if pattern.match(w):
            matched += 1
            #if w.lower() == l or w == l:
            #    caseless_equal += 1
            if w == l:
                equal += 1
                continue
            exceptions += [(w,l)]
    ratio = equal / matched * 100 if matched > 0 else 0
    print("{}/{} ({:.2f}%) matched words have identical lemma.".format(equal, matched, ratio))
    #ratio = caseless_equal / matched * 100 if matched > 0 else 0
    #print("{}/{} ({:.2f}%) matched words have identical lemma when caseless.".format(caseless_equal, matched, ratio))

    print("Sampled exceptions:")
    sample_num = 20 if len(exceptions) > 20 else len(exceptions)
    ex = random.sample(exceptions, sample_num)
    for w, l in ex:
        print("\t{}\t{}".format(w,l))

if __name__ == '__main__':
    #prepare_data()
    pairs = load_data()
    #test(pairs, LONG_NUMBER_PATTERN)
    #test(pairs, URL_PATTERN)
    #test(pairs, DECIMAL_PATTERN)
    test(pairs, CAPITAL_PATTERN)

