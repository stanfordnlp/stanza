import argparse
import json
from operator import itemgetter
import os

import stanza

from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import process_document

tqdm = get_tqdm()

def flatten_spans(coref_spans):
    """
    Put span IDs on each span, then flatten them into a single list sorted by first word
    """
    # put span indices on the spans
    #   [[[38, 39], [42, 43], [41, 41], [180, 180], [300, 300]], [[60, 68],
    #   -->
    #   [[[0, 38, 39], [0, 42, 43], [0, 41, 41], [0, 180, 180], [0, 300, 300]], [[1, 60, 68], ...
    coref_spans = [[[span_idx, x, y] for x, y in span] for span_idx, span in enumerate(coref_spans)]
    # flatten list
    #   -->
    #   [[0, 38, 39], [0, 42, 43], [0, 41, 41], [0, 180, 180], [0, 300, 300], [1, 60, 68], ...
    coref_spans = [y for x in coref_spans for y in x]
    # sort by the first word index
    #   -->
    #   [[0, 38, 39], [0, 42, 43], [0, 41, 41], [1, 60, 68], [0, 180, 180], [0, 300, 300], ...
    coref_spans = sorted(coref_spans, key=itemgetter(1))
    return coref_spans

def remove_nulls(coref_spans, sentences):
    """
    Removes the "" and "NULL" words from the sentences

    Also, reindex the spans by the number of words removed.
    So, we might get something like
      [[0, 2], [31, 33], [134, 136], [161, 162]]
      ->
      [[0, 2], [30, 32], [129, 131], [155, 156]]
    """
    word_map = []
    word_idx = 0
    map_idx = 0
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            word_map.append(map_idx)
            word_idx += 1
            if word != '' and word != 'NULL':
                new_sentence.append(word)
                map_idx += 1
        new_sentences.append(new_sentence)

    new_spans = []
    for mention in coref_spans:
        new_mention = []
        for span in mention:
            span = [word_map[x] for x in span]
            new_mention.append(span)
        new_spans.append(new_mention)
    return new_spans, new_sentences

def arrange_spans_by_sentence(coref_spans, sentences):
    sentence_spans = []

    current_index = 0
    span_idx = 0
    for sentence in sentences:
        current_sentence_spans = []
        end_index = current_index + len(sentence)
        while span_idx < len(coref_spans) and coref_spans[span_idx][1] < end_index:
            new_span = [coref_spans[span_idx][0], coref_spans[span_idx][1] - current_index, coref_spans[span_idx][2] - current_index]
            current_sentence_spans.append(new_span)
            span_idx += 1
        sentence_spans.append(current_sentence_spans)
        current_index = end_index
    return sentence_spans

def convert_dataset_section(pipe, section, use_cconj_heads):
    """
    Reprocess the original data into a format compatible with previous conversion utilities

    - remove blank and NULL words
    - rearrange the spans into spans per sentence instead of a list of indices for each span
    - process the document using a Hindi pipeline
    """
    processed_section = []

    for idx, doc in enumerate(tqdm(section)):
        doc_id = doc['doc_key']
        part_id = ""
        sentences = doc['sentences']
        sentence_speakers = doc['speakers']

        coref_spans = doc['clusters']
        coref_spans, sentences = remove_nulls(coref_spans, sentences)
        coref_spans = flatten_spans(coref_spans)
        coref_spans = arrange_spans_by_sentence(coref_spans, sentences)

        processed = process_document(pipe, doc_id, part_id, sentences, coref_spans, sentence_speakers, use_cconj_heads=use_cconj_heads)
        processed_section.append(processed)
    return processed_section

def remove_nulls_dataset_section(section):
    processed_section = []
    for doc in section:
        sentences = doc['sentences']
        coref_spans = doc['clusters']
        coref_spans, sentences = remove_nulls(coref_spans, sentences)
        doc['sentences'] = sentences
        doc['clusters'] = coref_spans
        processed_section.append(doc)
    return processed_section


def read_json_file(filename):
    with open(filename, encoding="utf-8") as fin:
        dataset = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))
    return dataset

def write_json_file(output_filename, converted_section):
    with open(output_filename, "w", encoding="utf-8") as fout:
        json.dump(converted_section, fout, indent=2)

def main():
    parser = argparse.ArgumentParser(
        prog='Convert Hindi Coref Data',
    )
    parser.add_argument('--no_use_cconj_heads', dest='use_cconj_heads', action='store_false', help="Don't use the conjunction-aware transformation")
    parser.add_argument('--remove_nulls', action='store_true', help="The only action is to remove the NULLs and blank tokens")
    args = parser.parse_args()

    paths = get_default_paths()
    coref_input_path = paths["COREF_BASE"]
    hindi_base_path = os.path.join(coref_input_path, "hindi", "dataset")

    sections = ("train", "dev", "test")
    if args.remove_nulls:
        for section in sections:
            input_filename = os.path.join(hindi_base_path, "%s.hindi.jsonlines" % section)
            dataset = read_json_file(input_filename)
            dataset = remove_nulls_dataset_section(dataset)
            output_filename = os.path.join(hindi_base_path, "hi_deeph.%s.nonulls.json" % section)
            with open(output_filename, "w", encoding="utf-8") as fout:
                for doc in dataset:
                    json.dump(doc, fout, ensure_ascii=False)
                    fout.write("\n")
    else:
        pipe = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse", package="default_accurate", tokenize_pretokenized=True)

        os.makedirs(paths["COREF_DATA_DIR"], exist_ok=True)

        for section in sections:
            input_filename = os.path.join(hindi_base_path, "%s.hindi.jsonlines" % section)
            dataset = read_json_file(input_filename)

            output_filename = os.path.join(paths["COREF_DATA_DIR"], "hi_deeph.%s.json" % section)
            converted_section = convert_dataset_section(pipe, dataset, use_cconj_heads=args.use_cconj_heads)
            write_json_file(output_filename, converted_section)

if __name__ == '__main__':
    main()
