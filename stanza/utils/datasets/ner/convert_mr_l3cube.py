"""
Reads one piece of the MR L3Cube dataset

The dataset is structured as a long list of words already in IOB format
The sentences have an ID which changes when a new sentence starts
The tags are labeled BNEM instead of B-NEM, so we update that.
(Could theoretically remap the tags to names more typical of other datasets as well)
"""

def convert(input_file):
    """
    Converts one file of the dataset

    Return: a list of list of pairs, (text, tag)
    """
    with open(input_file, encoding="utf-8") as fin:
        lines = fin.readlines()

    sentences = []
    current_sentence = []
    prev_sent_id = None
    for idx, line in enumerate(lines):
        # first line of each of the segments is the header
        if idx == 0:
            continue

        line = line.strip()
        if not line:
            continue
        pieces = line.split("\t")
        if len(pieces) != 3:
            raise ValueError("Unexpected number of pieces at line %d of %s" % (idx, input_file))

        text, ner, sent_id = pieces
        if ner != 'O':
            # ner symbols are written as BNEM, BNED, etc in this dataset
            ner = ner[0] + "-" + ner[1:]

        if not prev_sent_id:
            prev_sent_id = sent_id
        if sent_id != prev_sent_id:
            prev_sent_id = sent_id
            if len(current_sentence) == 0:
                raise ValueError("This should not happen!")
            sentences.append(current_sentence)
            current_sentence = []

        current_sentence.append((text, ner))

    if current_sentence:
        sentences.append(current_sentence)

    print("Read %d sentences in %d lines from %s" % (len(sentences), len(lines), input_file))
    return sentences
