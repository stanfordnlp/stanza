"""
Convert the AU-KBC coreference dataset from Prof. Sobha

https://aclanthology.org/2020.wildre-1.4/

Located in /u/nlp/data/coref/tamil on the Stanford cluster
"""

import argparse
import glob
import json
from operator import itemgetter
import os
import random
import re

import stanza

from stanza.utils.datasets.coref.utils import process_document
from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

begin_re = re.compile(r"B-([0-9]+)")
in_re =  re.compile(r"I-([0-9]+)")

def write_json_file(output_filename, converted_section):
    with open(output_filename, "w", encoding="utf-8") as fout:
        json.dump(converted_section, fout, indent=2)

def read_doc(filename):
    """
    Returns the sentences and the coref markings from this filename

    sentences: a list of list of words
    corefs: a list of list of clusters, which were tagged B-num and I-num in the dataset
    """
    with open(filename, encoding="utf-8") as fin:
        lines = fin.readlines()

    all_words = []
    all_coref = []
    current_words = []
    current_coref = []
    for line in lines:
        line = line.strip()
        if not line:
            all_words.append(current_words)
            all_coref.append(current_coref)
            current_words = []
            current_coref = []
            continue
        pieces = line.split("\t")
        current_words.append(pieces[3])
        current_coref.append(pieces[-1])

    if current_words:
        all_words.append(current_words)
        all_coref.append(current_coref)

    return all_words, all_coref

def convert_clusters(filename, corefs):
    sentence_clusters = []
    # current_clusters will be a list of (cluster id, start idx)
    for sent_idx, sentence_coref in enumerate(corefs):
        current_clusters = []
        processed = []
        for word_idx, word_coref in enumerate(sentence_coref):
            new_clusters = []
            if word_coref == '-':
                pieces = []
            else:
                pieces = word_coref.split(";")
            for piece in pieces:
                if not piece.startswith("I-") and not piece.startswith("B-"):
                    raise ValueError("Unexpected coref format %s in document %s" % (word_coref, filename))
                if piece.startswith("B-"):
                    new_clusters.append((int(piece[2:]), word_idx))
                else:
                    assert piece.startswith("I-")
                    cluster_id = int(piece[2:])
                    # this will keep the first cluster found
                    # the effect of this is that when two clusters overlap,
                    # and they happen to be the same cluster id,
                    # they will be nested instead of overlapping past each other
                    for idx, previous_cluster in enumerate(current_clusters):
                        if previous_cluster[0] == cluster_id:
                            break
                    else:
                        raise ValueError("Cluster %s does not continue an existing cluster in %s" % (piece, filename))
                    new_clusters.append(previous_cluster)
                    del current_clusters[idx]

            for cluster, start_idx in current_clusters:
                processed.append((cluster, start_idx, word_idx-1))
            current_clusters = new_clusters
        for cluster, start_idx in current_clusters:
            processed.append((cluster, start_idx, len(sentence_coref)-1))
        # sort by the first word index
        processed = sorted(processed, key=itemgetter(1))
        # TODO: cluster IDs are starting at 1, not 0.
        # that may or may not be relevant
        sentence_clusters.append(processed)
    return sentence_clusters

def main():
    parser = argparse.ArgumentParser(
        prog='Convert Tamil Coref Data',
    )
    parser.add_argument('--no_use_cconj_heads', dest='use_cconj_heads', action='store_false', help="Don't use the conjunction-aware transformation")
    args = parser.parse_args()

    random.seed(1234)

    paths = get_default_paths()
    coref_input_path = paths["COREF_BASE"]
    tamil_base_path = os.path.join(coref_input_path, "tamil", "coref_ta_corrected")
    tamil_glob = os.path.join(tamil_base_path, "*txt")

    filenames = sorted(glob.glob(tamil_glob))
    docs = [read_doc(x) for x in filenames]
    raw_sentences = [doc[0] for doc in docs]
    sentence_clusters = [convert_clusters(filename, doc[1]) for filename, doc in zip(filenames, docs)]

    pipe = stanza.Pipeline("ta", processors="tokenize,pos,lemma,depparse", package="default_accurate", tokenize_pretokenized=True)

    train, dev, test = [], [], []
    for filename, sentences, coref_spans in tqdm(zip(filenames, raw_sentences, sentence_clusters), total=len(filenames)):
        doc_id = filename
        part_id = " "
        sentence_speakers = [[""] * len(sent) for sent in sentences]

        processed = process_document(pipe, doc_id, part_id, sentences, coref_spans, sentence_speakers, use_cconj_heads=args.use_cconj_heads)
        location = random.choices((train, dev, test), weights = (0.8, 0.1, 0.1))[0]
        location.append(processed)

    output_filename = os.path.join(paths["COREF_DATA_DIR"], "ta_kbc.train.json")
    write_json_file(output_filename, train)

    output_filename = os.path.join(paths["COREF_DATA_DIR"], "ta_kbc.dev.json")
    write_json_file(output_filename, dev)

    output_filename = os.path.join(paths["COREF_DATA_DIR"], "ta_kbc.test.json")
    write_json_file(output_filename, test)

if __name__ == '__main__':
    main()
