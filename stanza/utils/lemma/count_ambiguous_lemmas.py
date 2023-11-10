"""
Read in a UD file, report any word/verb pairs which get lemmatized to different lemmas
"""

from collections import Counter, defaultdict
import sys

from stanza.utils.conll import CoNLL

filename = sys.argv[1]
print(filename)

lemma_counters = defaultdict(Counter)

doc = CoNLL.conll2doc(input_file=filename)
for sentence in doc.sentences:
    for word in sentence.words:
        text = word.text
        upos = word.upos
        lemma = word.lemma

        lemma_counters[(text, upos)][lemma] += 1

keys = lemma_counters.keys()
keys = sorted(keys, reverse=True, key=lambda x: sum(lemma_counters[x][y] for y in lemma_counters[x]))
for text, upos in keys:
    if len(lemma_counters[(text, upos)]) > 1:
        print(text, upos, lemma_counters[(text, upos)])

