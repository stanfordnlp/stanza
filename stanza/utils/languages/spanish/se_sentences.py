"""
A simple, almost trivial script to read sentences from UD and mark any words of 'se' with a delimiter

Should be a useful input format for MLtwist to annotate the words
"""

from stanza.utils.conll import CoNLL

import sys

filename = sys.argv[1]
doc = CoNLL.conll2doc(filename, reconstruct_text=True)

def process_se(doc):
    for sentence in doc.sentences:
        for idx, token in enumerate(sentence.tokens):
            if any(word.text == 'se' for word in token.words):
                yield sentence, idx

for sentence, idx in process_se(doc):
    token = sentence.tokens[idx]
    # note that sentences have a start index somewhere in the middle of the doc,
    # not necessarily at 0
    start_char = token.start_char - sentence.tokens[0].start_char
    end_char = token.end_char - sentence.tokens[0].start_char
    text = sentence.text[:start_char] + "---> " + token.text + " <---" + sentence.text[end_char:]
    print(text)

