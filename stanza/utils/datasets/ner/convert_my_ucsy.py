"""
Processes the three pieces of the NER dataset we received from UCSY.

Requires the Myanmar tokenizer to exist, since the text is not already tokenized.

There are three files sent to us from UCSY, one each for train, dev, test
This script expects them to be in the ner directory with the names
    $NERBASE/my_ucsy/Myanmar_NER_train.txt
    $NERBASE/my_ucsy/Myanmar_NER_dev.txt
    $NERBASE/my_ucsy/Myanmar_NER_test.txt

The files are in the following format:
  unsegmentedtext@LABEL|unsegmentedtext@LABEL|...
with one sentence per line

Solution:
  - break the text up into fragments by splitting on |
  - extract the labels
  - segment each block of text using the MY tokenizer

We could take two approaches to breaking up the blocks.  One would be
to combine all chunks, then segment an entire sentence at once.  This
would require some logic to re-chunk the resulting pieces.  Instead,
we resegment each individual chunk by itself.  This loses the
information from the neighboring chunks, but guarantees there are no
screwups where segmentation crosses segment boundaries and is simpler
to code.

Of course, experimenting with the alternate approach might be better.

There is one stray label of SB in the training data, so we throw out
that entire sentence.
"""


import os

from tqdm import tqdm
import stanza
from stanza.utils.datasets.ner.check_for_duplicates import check_for_duplicates

SPLITS = ("train", "dev", "test")

def convert_file(input_filename, output_filename, pipe):
    with open(input_filename) as fin:
        lines = fin.readlines()

    all_labels = set()

    with open(output_filename, "w") as fout:
        for line in tqdm(lines):
            pieces = line.split("|")
            texts = []
            labels = []
            skip_sentence = False
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                text, label = piece.rsplit("@", maxsplit=1)
                text = text.strip()
                if not text:
                    continue
                if label == 'SB':
                    skip_sentence = True
                    break

                texts.append(text)
                labels.append(label)

            if skip_sentence:
                continue

            text = "\n\n".join(texts)
            doc = pipe(text)
            assert len(doc.sentences) == len(texts)
            for sentence, label in zip(doc.sentences, labels):
                all_labels.add(label)
                for word_idx, word in enumerate(sentence.words):
                    if label == "O":
                        output_label = "O"
                    elif word_idx == 0:
                        output_label = "B-" + label
                    else:
                        output_label = "I-" + label

                    fout.write("%s\t%s\n" % (word.text, output_label))
            fout.write("\n\n")

    print("Finished processing {}  Labels found: {}".format(input_filename, sorted(all_labels)))

def convert_my_ucsy(base_input_path, base_output_path):
    os.makedirs(base_output_path, exist_ok=True)
    pipe = stanza.Pipeline("my", processors="tokenize", tokenize_no_ssplit=True)
    output_filenames = [os.path.join(base_output_path, "my_ucsy.%s.bio" % split) for split in SPLITS]

    for split, output_filename in zip(SPLITS, output_filenames):
        input_filename = os.path.join(base_input_path, "Myanmar_NER_%s.txt" % split)
        if not os.path.exists(input_filename):
            raise FileNotFoundError("Necessary file for my_ucsy does not exist: %s" % input_filename)

        convert_file(input_filename, output_filename, pipe)
