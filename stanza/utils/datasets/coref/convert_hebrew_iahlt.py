"""
Convert the coref annotation of IAHLT to the Stanza coref format

This dataset is available at

https://github.com/IAHLT/coref

Download it via git clone to $COREF_BASE/hebrew, so for example on the cluster:

cd /u/nlp/data/coref/
mkdir hebrew
cd hebrew
git clone git@github.com:IAHLT/coref.git

Then run

python3 stanza/utils/datasets/coref/convert_hebrew_iahlt.py

TODO: the scores from this model are horrible, only 30 F1.
Need to either verify the usage elsewhere or double check the outputs of the conversion
"""

from collections import defaultdict, namedtuple
import json
import os

import stanza

from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import process_document

tqdm = get_tqdm()

CorefDoc = namedtuple("CorefDoc", ['doc_id', 'sentences', 'coref_spans'])

# TODO: binary search for speed?
def search_mention_start(doc, mention_start):
    for sent_idx, sentence in enumerate(doc.sentences):
        if mention_start < doc.sentences[sent_idx].words[-1].end_char:
            break
    else:
        raise ValueError
    for word_idx, word in enumerate(sentence.words):
        if mention_start < word.end_char:
            break
    else:
        raise ValueError
    return sent_idx, word_idx

def search_mention_end(doc, mention_end):
    for sent_idx, sentence in enumerate(doc.sentences):
        if sent_idx + 1 == len(doc.sentences) or mention_end < doc.sentences[sent_idx+1].words[0].start_char:
            break
    for word_idx, word in enumerate(sentence.words):
        if word_idx + 1 == len(sentence.words) or mention_end < sentence.words[word_idx+1].start_char:
            break
    return sent_idx, word_idx

def extract_doc(tokenizer, lines):
    # 16, 1, 5 for the train, dev, test sets
    broken = 0
    singletons = 0
    one_words = 0
    processed_docs = []
    for line_idx, line in enumerate(tqdm(lines)):
        all_clusters = defaultdict(list)
        doc_id = line['metadata']['doc_id']
        text = line['text']
        clusters = line['clusters']
        doc = tokenizer(text)
        for cluster_idx, cluster in enumerate(clusters):
            found_mentions = []
            for mention_idx, mention in enumerate(cluster['mentions']):
                mention_start = mention[0]
                mention_end = mention[1]
                start_sent, start_word = search_mention_start(doc, mention_start)
                end_sent, end_word = search_mention_end(doc, mention_end)
                assert end_sent >= start_sent
                if start_sent != end_sent:
                    broken += 1
                else:
                    assert end_word >= start_word
                    if end_word == start_word:
                        one_words += 1
                    found_mentions.append((start_sent, start_word, end_word))

                    #if cluster_idx == 0 and line_idx == 0:
                    #    expanded_start = max(0, mention_start - 10)
                    #    expanded_end = min(len(text), mention_end + 10)
                    #    print("EXTRACTING MENTION: %d %d" % (mention[0], mention[1]))
                    #    print(" context: |%s|" % text[expanded_start:expanded_end])
                    #    print(" mention[0]:mention[1]: |%s|" % text[mention[0]:mention[1]])
                    #    print(" search text: |%s|" % text[mention_start:mention_end])
                    #    extracted_words = doc.sentences[start_sent].words[start_word:end_word+1]
                    #    extracted_text = " ".join([x.text for x in extracted_words])
                    #    print(" extracted words: |%s|" % extracted_text)
                    #    print(" endpoints: %d %d" % (mention_start, mention_end))
                    #    print(" number of extracted words: %d" % len(extracted_words))
                    #    print(" first word endpoints: %d %d" % (extracted_words[0].start_char, extracted_words[0].end_char))
                    #    print(" last word endpoints: %d %d" % (extracted_words[-1].start_char, extracted_words[-1].end_char))
            if len(found_mentions) == 0:
                continue
            elif len(found_mentions) == 1:
                # the number of singletons, after discarding mentions that
                # crossed a sentence boundary according to Stanza, is
                # 5, 0, 1
                # so clearly the dataset does not intentionally have
                # (many?) singletons in it
                singletons += 1
                continue
            else:
                all_clusters[cluster_idx] = found_mentions
        # maybe we need to update the interface - there can be MWT in Hebrew
        sentences = [[word.text for word in sent.words] for sent in doc.sentences]
        coref_spans = defaultdict(list)
        for cluster_idx in all_clusters:
            for sent_idx, start_word, end_word in all_clusters[cluster_idx]:
                coref_spans[sent_idx].append((cluster_idx, start_word, end_word))
        processed_docs.append(CorefDoc(doc_id, sentences, coref_spans))
    print("Found %d broken across two sentences, %d singleton mentions, %d one_word mentions" % (broken, singletons, one_words))
    return processed_docs

def read_doc(tokenizer, filename):
    with open(filename, encoding="utf-8") as fin:
        lines = fin.readlines()
    lines = [json.loads(line) for line in lines]
    return extract_doc(tokenizer, lines)

def write_json_file(output_filename, dataset):
    with open(output_filename, "w", encoding="utf-8") as fout:
        json.dump(dataset, fout, indent=2, ensure_ascii=False)

def main():
    paths = get_default_paths()

    coref_input_path = paths["COREF_BASE"]
    hebrew_base_path = os.path.join(coref_input_path, "hebrew", "coref", "train_val_test")

    tokenizer = stanza.Pipeline("he", processors="tokenize", package="default_accurate")
    pipe = stanza.Pipeline("he", processors="tokenize,pos,lemma,depparse", package="default_accurate", tokenize_pretokenized=True)

    input_files = ["coref-5-heb_train.jsonl", "coref-5-heb_val.jsonl", "coref-5-heb_test.jsonl"]
    output_files = ["he_iahlt.train.json", "he_iahlt.dev.json", "he_iahlt.test.json"]
    for input_filename, output_filename in zip(input_files, output_files):
        input_filename = os.path.join(hebrew_base_path, input_filename)
        assert os.path.exists(input_filename)
        docs = read_doc(tokenizer, input_filename)
        dataset = [process_document(pipe, doc.doc_id, "", doc.sentences, doc.coref_spans, None) for doc in tqdm(docs)]

        output_filename = os.path.join(paths["COREF_DATA_DIR"], output_filename)
        write_json_file(output_filename, dataset)

if __name__ == '__main__':
    main()
