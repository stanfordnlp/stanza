"""
Downloads (if necessary) conll03 from Huggingface, then converts it to Stanza .json

Some online sources for CoNLL 2003 require multiple pieces, but it is currently hosted on HF:
https://huggingface.co/datasets/conll2003
"""

import os

from stanza.utils.default_paths import get_default_paths
from stanza.utils.datasets.ner.utils import write_dataset

ID_TO_TAG = ["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE",]

def convert_dataset_section(config_name, section):
    sentences = []
    for doc in section:
        # the nt_ sentences (New Testament) in the HF version of OntoNotes
        # have blank named_entities, even though there was no original .name file
        # that corresponded with these annotations
        if config_name.startswith("english") and doc['document_id'].startswith("pt/nt"):
            continue
        for sentence in doc['sentences']:
            words = sentence['words']
            tags = [ID_TO_TAG[x] for x in sentence['named_entities']]
            sentences.append(list(zip(words, tags)))
    return sentences

def process_dataset(short_name, conll_path, ner_output_path):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Please install the datasets package to process CoNLL03 with Stanza")

    if short_name == 'en_ontonotes':
        # there is an english_v12, but it is filled with junk annotations
        # for example, near the end:
        #   And John_O, I realize
        config_name = 'english_v4'
    elif short_name in ('zh_ontonotes', 'zh-hans_ontonotes'):
        config_name = 'chinese_v4'
    elif short_name == 'ar_ontonotes':
        config_name = 'arabic_v4'
    else:
        raise ValueError("Unknown short name for downloading ontonotes: %s" % short_name)
    dataset = load_dataset("conll2012_ontonotesv5", config_name, cache_dir=conll_path)
    datasets = [convert_dataset_section(config_name, x) for x in [dataset['train'], dataset['validation'], dataset['test']]]
    write_dataset(datasets, ner_output_path, short_name)

def main():
    paths = get_default_paths()
    ner_input_path = paths['NERBASE']
    conll_path = os.path.join(ner_input_path, "english", "en_ontonotes")
    ner_output_path = paths['NER_DATA_DIR']
    process_dataset("en_ontonotes", conll_path, ner_output_path)

if __name__ == '__main__':
    main()
