from collections import defaultdict
import json
import os
import re
import glob

import stanza 

from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import find_cconj_head
from stanza.utils.datasets.coref.utils import process_document

from stanza.utils.conll import CoNLL

from random import Random
from pathlib import Path

from glob import glob

import argparse

split_random = Random(8)

tqdm = get_tqdm()

SPANS = re.compile(r"(\(\w+|[%\w]+\))")
IS_UDCOREF_FORMAT = False
UDCOREF_ADDN = 0 if not IS_UDCOREF_FORMAT else 1

def extract_spans(sentence):
    words = [i.text for i in sentence.words]
    misc = [i.misc for i in sentence.words]
    refs = [SPANS.findall(i) if i else [] for i in misc]

    refdict = defaultdict(list)
    final_refs = defaultdict(list)
    last_ref = None
    for indx, i in enumerate(refs):
        for j in i:
            # this is the beginning of a reference
            if j[0] == "(":
                refdict[j[1+UDCOREF_ADDN:]].append(indx)
                last_ref = j[1+UDCOREF_ADDN:]
            # at the end of a reference, if we got exxxxx, that ends
            # a particular refereenc; otherwise, it ends the last reference
            elif j[-1] == ")" and j[UDCOREF_ADDN:-1].isnumeric():
                if (not UDCOREF_ADDN) or j[0] == "e":
                    try:
                        final_refs[j[UDCOREF_ADDN:-1]].append((refdict[j[UDCOREF_ADDN:-1]].pop(-1), indx))
                    except IndexError:
                        # this is probably zero anaphora
                        continue
            elif j[-1] == ")":
                final_refs[last_ref].append((refdict[last_ref].pop(-1), indx))
                last_ref = None
    final_refs = dict(final_refs)
    # convert it to the right format (specifically, in (ref, start, end) tuples)
    coref_spans = []
    for k, v in final_refs.items():
        for i in v:
            coref_spans.append([int(k), i[0], i[1]])

    return words, coref_spans

def process_split(dataset, pipe):
    """Process a split of the whole dataset

    Arguments
    ----------
        dataset : List[str]
            A list of filenames to process as a part of that split, must be in
            Litbank CoNLL format which is a single document per file, with a
            variable number of columns between 10 and 11 (where the 11th column
            is the coref annotation).
        pipe : stanza.Pipeline
            A stanza pipeline to process the data with.

    Returns
    -------
    List[Dict]
        Prepared data for the dataset
    """
    prepared = []
    for doc in tqdm(dataset):
        # extract document-level information; the ignore_first="litbank" is
        # a special flag to deal with variable 10-11 column litbank data which
        # isn't technicially ignoring the first columns and is instead ignoring
        # some of the middle
        shortname = Path(doc).stem
        doc = CoNLL.conll2doc(doc, ignore_first="litbank")
        words = []
        spans = []
        speakers = []
        for i in doc.sentences:
            a, b = extract_spans(i)
            words.append(a)
            spans.append(b)
            speakers.append(["_" for _ in a])
        prepared.append(process_document(pipe, shortname, 0, words, spans, speakers))

    return prepared

def process_dataset(src, coref_output_path, train_split = 0.7, dev_split = 0.2):
    pipe = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                        package="default_accurate", tokenize_pretokenized=True)
    dataset = glob(str(Path(src) / "*.conll"))
    length = len(dataset)

    train_data = split_random.sample(dataset, int(train_split*length))
    dataset = [i for i in dataset if i not in train_data]
    dev_data = split_random.sample(dataset, int(dev_split*length))
    test_data = [i for i in dataset if i not in dev_data]

    for i,j in zip(["train", "dev", "test"], [train_data, dev_data, test_data]):
        print("Processing %s split of length %d..." % (i, len(j)))
        results = process_split(j, pipe)

        os.makedirs(coref_output_path, exist_ok=True)
        output_filename = os.path.join(coref_output_path, "litbank.%s.json" % (i))
        with open(output_filename, "w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2)

def main():
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        prog='Convert Litbank Data',
    )
    parser.add_argument('--split_train', default=0.7, type=float, help='How much of the data to randomly split from train to make a test set')
    parser.add_argument('--split_dev', default=0.3, type=float, help='How much of the data to randomly split from train to make a test set')

    args = parser.parse_args()
    coref_input_path = paths['COREF_BASE']
    coref_output_path = paths['COREF_DATA_DIR']

    src = Path(coref_input_path)/"litbank"
    process_dataset(str(src), coref_output_path, args.split_train, args.split_dev)


if __name__ == '__main__':
    main()

