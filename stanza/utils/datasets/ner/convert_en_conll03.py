"""
Downloads (if necessary) conll03 from Huggingface, then converts it to Stanza .json

Some online sources for CoNLL 2003 require multiple pieces, but it is currently hosted on HF:
https://huggingface.co/datasets/conll2003
"""

import os

from stanza.utils.default_paths import get_default_paths
from stanza.utils.datasets.ner.utils import write_dataset

TAG_TO_ID = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ID_TO_TAG = {y: x for x, y in TAG_TO_ID.items()}

def convert_dataset_section(section):
    sentences = []
    for item in section:
        words = item['tokens']
        tags = [ID_TO_TAG[x] for x in item['ner_tags']]
        sentences.append(list(zip(words, tags)))
    return sentences

def process_dataset(short_name, conll_path, ner_output_path):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Please install the datasets package to process CoNLL03 with Stanza")

    dataset = load_dataset('conll2003', cache_dir=conll_path)
    datasets = [convert_dataset_section(x) for x in [dataset['train'], dataset['validation'], dataset['test']]]
    write_dataset(datasets, ner_output_path, short_name)

def main():
    paths = get_default_paths()
    ner_input_path = paths['NERBASE']
    conll_path = os.path.join(ner_input_path, "english", "en_conll03")
    ner_output_path = paths['NER_DATA_DIR']
    process_dataset("en_conll03", conll_path, ner_output_path)

if __name__ == '__main__':
    main()
