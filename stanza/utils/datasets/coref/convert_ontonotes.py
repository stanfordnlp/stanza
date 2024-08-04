import json
import os

import stanza

from stanza.models.constituency import tree_reader
from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import process_document

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
        coref_spans = [x['coref_spans'] for x in paragraph]
        sentence_speakers = [x['speaker'] for x in paragraph]

        processed = process_document(pipe, doc_id, part_id, sentences, coref_spans, sentence_speakers)
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

