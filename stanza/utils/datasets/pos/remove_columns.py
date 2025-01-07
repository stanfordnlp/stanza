"""
Remove xpos and feats from each file given at the command line.

Useful to strip unwanted tags when combining files of two different
types (or two different stages in the annotation process).

Super rudimentary right now.  Will be upgraded if needed
"""

import sys

from stanza.utils.conll import CoNLL

def remove_columns(filename):
    doc = CoNLL.conll2doc(filename)

    for sentence in doc.sentences:
        for word in sentence.words:
            word.xpos = None
            word.feats = None

    CoNLL.write_doc2conll(doc, filename)

if __name__ == '__main__':
    for filename in sys.argv[1:]:
        remove_columns(filename)
