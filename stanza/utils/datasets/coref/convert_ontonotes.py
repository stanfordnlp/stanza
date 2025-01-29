"""
convert_ontonotes.py

This script is used to convert the OntoNotes dataset into a format that can be used by Stanza's coreference resolution model. The script uses the datasets package to download the OntoNotes dataset and then processes the dataset using Stanza's coreference resolution pipeline. The processed dataset is then saved in a JSON file.

If you want to simply process the official OntoNotes dataset...
1. install the `datasets` package: `pip install datasets`
2. make folders! (or those adjusted to taste through scripts/config.sh)
   - extern_data/coref/english/en_ontonotes
   - data/coref
2. run this script: python -m stanza.utils.datasets.coref.convert_ontonotes

If you happen to have singleton annotated coref chains...
1. install the `datasets` package: `pip install datasets`
2. make folders! (or those adjusted to taste through scripts/config.sh)
   - extern_data/coref/english/en_ontonotes
   - data/coref
3. get the singletons annotated coref chains in conll format from the Splice repo
    https://github.com/yilunzhu/splice/raw/refs/heads/main/data/ontonotes5_mentions.zip
4. place the singleton annotated coref chains in the folder `extern_data/coref/english/en_ontonotes`
   $ ls ./extern_data/coref/english/en_ontonotes
        dev_sg_pred.english.v4_gold_conll
        test_sg_pred.english.v4_gold_conll
        train_sg.english.v4_gold_conll
5. run this script: python -m stanza.utils.datasets.coref.convert_ontonotes

Your results will appear in ./data/coref/, and you can be off to the races with training!
Note that this script invokes Stanza itself to run some tagging.
"""

import json
import os

from pathlib import Path

import stanza

from stanza.models.constituency import tree_reader
from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import process_document

from stanza.utils.conll import CoNLL
from collections import defaultdict

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


def convert_dataset_section(pipe, section, override_singleton_chains=None):
    processed_section = []
    section = list(x for x in read_paragraphs(section))

    # we need to do this because apparently the singleton annotations
    # don't use the same numbering scheme as the ontonotes annotations
    # so there will be chain id conflicts
    max_chain_id = sorted([
        chain_id 
        for i in section 
        for j in i[2] 
        for chain_id, _, _ in j["coref_spans"]
    ])[-1]
    # this dictionary will map singleton chains' "special" ids
    # to the OntoNotes IDs
    sg_to_ontonotes_cluster_id_map = defaultdict(
        lambda: len(sg_to_ontonotes_cluster_id_map)+max_chain_id+1
    )

    for idx, (doc_id, part_id, paragraph) in enumerate(tqdm(section)):
        sentences = [x['words'] for x in paragraph]
        truly_coref_spans = [x['coref_spans'] for x in paragraph]
        # the problem to solve here is that the singleton chains'
        # IDs don't match the coref chains' ids
        # 
        # and, what the labels calls a "singleton" may not actually
        # be one because the "singleton" seems like it includes all
        # NPs which may or may not be a singleton
        coref_spans = []
        if override_singleton_chains:
            singleton_chains = override_singleton_chains[doc_id][part_id]
            for singleton_pred, coref_pred in zip(singleton_chains, truly_coref_spans):
                sentence_coref_preds = []
                # these are sentence level predictions, which we will
                # disambiguate: if a subspan of "singleton" exists in the 
                # truly coref sets, we realise its not a singleton and
                # then ignore it
                coref_pred_locs = set([tuple(i[1:]) for i in coref_pred])
                for id,start,end in singleton_pred:
                    if (start,end) not in coref_pred_locs:
                        # this is truly a singleton
                        sentence_coref_preds.append([
                            sg_to_ontonotes_cluster_id_map[id],
                            start,
                            end
                        ])
                sentence_coref_preds += coref_pred
                coref_spans.append(sentence_coref_preds)
        else:
            coref_spans = truly_coref_spans

        sentence_speakers = [x['speaker'] for x in paragraph]

        processed = process_document(pipe, doc_id, part_id, sentences, coref_spans, sentence_speakers)
        processed_section.append(processed)
    return processed_section

def extract_chains_from_chunk(chunk):
    """give a chunk of the gold conll, extract the coref chains

    remember, the indicies are front and back *inclusive*, zero indexed
    and a span that takes one word only is annotated [id, n, n] (i.e. we
    don't fencepost by +1)

    Arguments
    ---------
        chunk : List[str]
            list of strings, each string is a line in the conll file

    Returns
    -------
        final_chains : List[Tuple[int, int, int ]]
            list of chains, each chain is a list of [id, open_location, close_location]
    """

    chains = [sentence.split("    ")[-1].strip()
            for sentence in chunk]
    chains = [[] if i == '-' else i.split("|")
            for i in chains]

    opens = defaultdict(list)
    closes = defaultdict(list)

    for indx, elem in enumerate(chains):

        # for each one, check if its an open, close, or both 
        for i in elem:
            id = int(i.strip("(").strip(")"))
            if (i[0]=="("):
                opens[id].append(indx)
            if (i[-1]==")"):
                closes[id].append(indx)

    # and now, we chain the ids' opens and closes together
    # into the shape of [id, open_location, close_location]
    opens = dict(opens)
    closes = dict(closes)

    final_chains = []
    for key, open_indx in opens.items():
        for o,c in zip(sorted(open_indx), sorted(closes[key])):
            final_chains.append([key, o,c])

    return final_chains

def extract_chains_from_conll(gold_coref_conll):
    """extract the coref chains from the gold conll file

    Arguments
    --------
        gold_coref_conll : str
            path to the gold conll file, with coreference chains
    Returns
    -------
        final_chunks : Dict[str, List[List[List[Tuple[int, int, int]]]]]
            dictionary of document_id to list of paragraphs into
            list of coref chains in OntoNotes style, keyed by document ID
    """
    with open(gold_coref_conll, 'r') as df:
        gold_coref_conll = df.readlines()
    # we want to first seperate the document into sentence-level
    # chunks; we assume that the ordering of the sentences are correct in the
    # gold document
    sections = []
    section = []
    chunk = []
    for i in gold_coref_conll:
        if len(i.split("    ")) < 10:
            if len(chunk) > 0:
                section.append(chunk)
            elif i.startswith("#end document"): # this is a new paragraph
                sections.append(section)
                section = []
            chunk = []
        else:
            chunk.append(i)

    # finally, we process each chunk and *index them by ID*
    final_chunks = defaultdict(list)
    for section in sections:
        section_chains = []
        for chunk in section:
            section_chains.append(extract_chains_from_chunk(chunk))
        final_chunks[chunk[0].split("    ")[0]].append(section_chains)
    final_chunks = dict(final_chunks)

    return final_chunks

SECTION_NAMES = {"train": "train",
                 "dev": "validation",
                 "test": "test"}
OVERRIDE_CONLL_PATHS = {"en_ontonotes": {
    "train": "train_sg.english.v4_gold_conll",
    "validation": "dev_sg_pred.english.v4_gold_conll",
    "test": "test_sg_pred.english.v4_gold_conll"
}}

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

    # if the cache directory doesn't yet exist, we make it
    # we store the cache in a seperate subfolder to distinguish from the
    # possible Singleton conlls that maybe in the folder
    (Path(ontonotes_path) / "cache").mkdir(exist_ok=True)

    dataset = load_dataset("conll2012_ontonotesv5", config_name, cache_dir=str(Path(ontonotes_path) / "cache"), trust_remote_code=True)
    for section, hf_name in SECTION_NAMES.items():
    # for section, hf_name in [("test", "test")]:
        print("Processing %s" % section)
        if (Path(ontonotes_path) / OVERRIDE_CONLL_PATHS[short_name][hf_name]).exists():
            # if, for instance, Amir have given us some singleton annotated coref chains in conll files,
            # we will use those instead of the ones that OntoNotes has
            converted_section = convert_dataset_section(pipe, dataset[hf_name], extract_chains_from_conll(
                str((Path(ontonotes_path) / OVERRIDE_CONLL_PATHS[short_name][hf_name]))
            ))
        else:
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

