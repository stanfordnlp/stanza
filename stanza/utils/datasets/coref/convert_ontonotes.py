from collections import defaultdict
import json
import os

import stanza

from stanza.models.constituency import tree_reader
from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

def read_paragraphs(section):
    for doc in section:
        part_id = None
        paragraph = []
        for sentence in doc['sentences']:
            if part_id is None:
                part_id = sentence['part_id']
            elif part_id != sentence['part_id']:
                yield doc['document_id'], part_id, paragraph
                paragraph = []
                part_id = sentence['part_id']
            paragraph.append(sentence)
        if paragraph != []:
            yield doc['document_id'], part_id, paragraph

def convert_dataset_section(pipe, section):
    processed_section = []
    section = list(x for x in read_paragraphs(section))
    for idx, (doc_id, part_id, paragraph) in enumerate(tqdm(section)):
        sentences = [x['words'] for x in paragraph]
        sentence_lens = [len(x) for x in sentences]
        cased_words = [y for x in sentences for y in x]
        sent_id = [y for idx, sent_len in enumerate(sentence_lens) for y in [idx] * sent_len]
        speaker = [y for x, sent_len in zip(paragraph, sentence_lens) for y in [x['speaker']] * sent_len]

        # use the trees to get the xpos tags
        # alternatively, could translate the pos_tags field,
        # but those have numbers, which is annoying
        #tree_text = "\n".join(x['parse_tree'] for x in paragraph)
        #trees = tree_reader.read_trees(tree_text)
        #pos = [x.label for tree in trees for x in tree.yield_preterminals()]
        # actually, the downstream code doesn't use pos at all.  maybe we can skip?

        doc = pipe(sentences)
        word_total = 0
        heads = []
        # TODO: does SD vs UD matter?
        deprel = []
        for sentence in doc.sentences:
            for word in sentence.words:
                deprel.append(word.deprel)
                if word.head == 0:
                    heads.append("null")
                else:
                    heads.append(word.head - 1 + word_total)
            word_total += len(sentence.words)

        span_clusters = defaultdict(list)
        word_clusters = defaultdict(list)
        head2span = []
        word_total = 0
        for parsed_sentence, ontonotes_sentence in zip(doc.sentences, paragraph):
            coref_spans = ontonotes_sentence['coref_spans']
            for span in coref_spans:
                # input is expected to be start word, end word + 1
                # counting from 0
                span_start = span[1] + word_total
                span_end = span[2] + word_total + 1
                for candidate_head in range(span[1], span[2] + 1):
                    # stanza uses 0 to mark the head, whereas OntoNotes is counting
                    # words from 0, so we have to subtract 1 from the stanza heads
                    #print(span, candidate_head, parsed_sentence.words[candidate_head].head - 1)
                    # treat the head of the phrase as the first word that has a head outside the phrase
                    if (parsed_sentence.words[candidate_head].head - 1 < span[1] or
                        parsed_sentence.words[candidate_head].head - 1 > span[2]):
                        break
                else:
                    # if none have a head outside the phrase (circular??)
                    # then just take the first word
                    candidate_head = span[1]
                #print("----> %d" % candidate_head)
                candidate_head += word_total
                span_clusters[span[0]].append((span_start, span_end))
                word_clusters[span[0]].append(candidate_head)
                head2span.append((candidate_head, span_start, span_end))
            word_total += len(ontonotes_sentence['words'])
        span_clusters = sorted([sorted(values) for _, values in span_clusters.items()])
        word_clusters = sorted([sorted(values) for _, values in word_clusters.items()])
        head2span = sorted(head2span)

        processed = {
            "document_id": doc_id,
            "cased_words": cased_words,
            "sent_id": sent_id,
            "part_id": part_id,
            "speaker": speaker,
            #"pos": pos,
            "deprel": deprel,
            "head": heads,
            "span_clusters": span_clusters,
            "word_clusters": word_clusters,
            "head2span": head2span,
        }
        processed_section.append(processed)
    return processed_section

SECTION_NAMES = {"train": "train",
                 "dev": "validation",
                 "test": "test"}

def process_dataset(short_name, ontonotes_path, coref_output_path):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Please install the datasets package to process OntoNotes coref with Stanza")

    if short_name == 'en_ontonotes':
        config_name = 'english_v4'
    elif short_name in ('zh_ontonotes', 'zh-hans_ontonotes'):
        config_name = 'chinese_v4'
    elif short_name == 'ar_ontonotes':
        config_name = 'arabic_v4'
    else:
        raise ValueError("Unknown short name for downloading ontonotes: %s" % short_name)

    pipe = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", package="default_accurate", tokenize_pretokenized=True)
    dataset = load_dataset("conll2012_ontonotesv5", config_name, cache_dir=ontonotes_path)
    for section, hf_name in SECTION_NAMES.items():
    #for section, hf_name in [("test", "test")]:
        print("Processing %s" % section)
        converted_section = convert_dataset_section(pipe, dataset[hf_name])
        output_filename = os.path.join(coref_output_path, "%s.%s.json" % (short_name, section))
        with open(output_filename, "w", encoding="utf-8") as fout:
            json.dump(converted_section, fout, indent=2)


def main():
    paths = get_default_paths()
    coref_input_path = paths['COREF_BASE']
    ontonotes_path = os.path.join(coref_input_path, "english", "en_ontonotes")
    coref_output_path = paths['COREF_DATA_DIR']
    process_dataset("en_ontonotes", ontonotes_path, coref_output_path)

if __name__ == '__main__':
    main()

