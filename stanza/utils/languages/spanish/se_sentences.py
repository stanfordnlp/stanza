"""
Read sentences from a UD treebank and mark each token containing the word 'se' with a delimiter.

Writes a CSV with one row per marked token:
    sent_id, word_id, text
where word_id is the id of the 'se' word within the sentence (inside the MWT
for enclitics such as "deberse").  The id columns let the returned labels be
joined back to the treebank by key rather than by row order.

Every field is quoted, so double quotes in the sentence text survive
the round trip through a spreadsheet.

usage: python3 mark_se.py es_ancora-ud-dev.conllu > se_marked.csv
"""

import csv
import sys

from stanza.utils.conll import CoNLL

# a cell starting with one of these is read as a formula by Excel
FORMULA_START = ("=", "+", "-", "@")

def process_se(doc):
    for sentence in doc.sentences:
        for idx, token in enumerate(sentence.tokens):
            for word in token.words:
                if word.text.lower() == 'se' and word.upos == 'PRON':
                    yield sentence, idx, word
                    break

def mark_sentence(sentence, idx):
    token = sentence.tokens[idx]
    # sentences have a start index somewhere in the middle of the doc,
    # not necessarily at 0
    start_char = token.start_char - sentence.tokens[0].start_char
    end_char = token.end_char - sentence.tokens[0].start_char
    text = sentence.text[:start_char] + "---> " + token.text + " <---" + sentence.text[end_char:]
    if text.startswith(FORMULA_START):
        text = " " + text
    return text

def main():
    filename = sys.argv[1]
    doc = CoNLL.conll2doc(filename, reconstruct_text=True)

    writer = csv.writer(sys.stdout, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["sent_id", "word_id", "text"])
    for sentence, idx, word in process_se(doc):
        writer.writerow([sentence.sent_id, word.id, mark_sentence(sentence, idx)])

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding="utf-8")
    main()
