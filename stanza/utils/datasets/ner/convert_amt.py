"""
Converts a .json file from AMT to a .bio format and then a .json file

To ignore Facility and Product, turn NORP into miscellaneous:

 python3 stanza/utils/datasets/ner/convert_amt.py --input_path /u/nlp/data/ner/stanza/en_amt/output.manifest --ignore Product,Facility --remap NORP=Miscellaneous

To turn all labels into the 4 class used in conll03:

  python3 stanza/utils/datasets/ner/convert_amt.py --input_path /u/nlp/data/ner/stanza/en_amt/output.manifest --ignore Product,Facility --remap NORP=MISC,Miscellaneous=MISC,Location=LOC,Person=PER,Organization=ORG
"""

import argparse
import copy
import json
from operator import itemgetter
import sys

from tqdm import tqdm

import stanza
from stanza.utils.datasets.ner.utils import write_sentences
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file

def read_json(input_filename):
    """
    Read the json file and extract the NER labels

    Will not return lines which are not labeled

    Return format is a list of lines
    where each line is a tuple: (text, labels)
    labels is a list of maps, {'label':..., 'startOffset':..., 'endOffset':...}
    """
    docs = []
    blank = 0
    unlabeled = 0
    broken = 0
    with open(input_filename, encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            doc = json.loads(line)
            if sorted(doc.keys()) == ['source']:
                unlabeled += 1
                continue
            if 'source' not in doc:
                blank += 1
                continue
            source = doc['source']
            entities = None
            for k in doc.keys():
                if k == 'source' or k.endswith('metadata'):
                    continue
                if 'annotations' not in doc[k]:
                    continue
                annotations = doc[k]['annotations']
                if 'entities' not in annotations:
                    continue
                if 'entities' in annotations:
                    if entities is not None:
                        raise ValueError("Found a map with multiple annotations at line %d" % line_idx)
                    entities = annotations['entities']
                # entities is now a map such as
                # [{'label': 'Location', 'startOffset': 0, 'endOffset': 6},
                #  {'label': 'Location', 'startOffset': 11, 'endOffset': 23},
                #  {'label': 'NORP', 'startOffset': 66, 'endOffset': 74},
                #  {'label': 'NORP', 'startOffset': 191, 'endOffset': 214}]
            if entities is None:
                unlabeled += 1
                continue
            is_broken = any(any(x not in entity for x in ('label', 'startOffset', 'endOffset'))
                            for entity in entities)
            if is_broken:
                broken += 1
                if broken == 1:
                    print("Found an entity which was missing either label, startOffset, or endOffset")
                    print(entities)
            docs.append((source, entities))

    print("Found %d labeled lines.  %d lines were blank, %d lines were broken, and %d lines were unlabeled" % (len(docs), blank, broken, unlabeled))
    return docs

def remove_ignored_labels(docs, ignored):
    if not ignored:
        return docs

    ignored = set(ignored.split(","))
    # drop all labels which match something in ignored
    # otherwise leave everything the same
    new_docs = [(doc[0], [x for x in doc[1] if x['label'] not in ignored])
                for doc in docs]
    return new_docs

def remap_labels(docs, remap):
    if not remap:
        return docs

    remappings = {}
    for remapping in remap.split(","):
        pieces = remapping.split("=")
        remappings[pieces[0]] = pieces[1]

    print(remappings)

    new_docs = []
    for doc in docs:
        entities = copy.deepcopy(doc[1])
        for entity in entities:
            entity['label'] = remappings.get(entity['label'], entity['label'])
        new_doc = (doc[0], entities)
        new_docs.append(new_doc)
    return new_docs

def remove_nesting(docs):
    """
    Currently the NER tool does not handle nesting, so we just throw away nested entities

    In the event of entites which exactly overlap, the first one in the list wins
    """
    new_docs = []
    nested = 0
    exact = 0
    total = 0
    for doc in docs:
        source, labels = doc
        # sort by startOffset, -endOffset
        labels = sorted(labels, key=lambda x: (x['startOffset'], -x['endOffset']))
        new_labels = []
        for label in labels:
            total += 1
            # note that this works trivially for an empty list
            for other in reversed(new_labels):
                if label['startOffset'] == other['startOffset'] and label['endOffset'] == other['endOffset']:
                    exact += 1
                    break
                if label['startOffset'] < other['endOffset']:
                    #print("Ignoring nested entity: {} |{}| vs {} |{}|".format(label, source[label['startOffset']:label['endOffset']], other, source[other['startOffset']:other['endOffset']]))
                    nested += 1
                    break
            else: # yes, this is meant to be a for-else
                new_labels.append(label)

        new_docs.append((source, new_labels))
    print("Ignored %d exact and %d nested labels out of %d entries" % (exact, nested, total))
    return new_docs

def process_doc(source, labels, pipe):
    """
    Given a source text and a list of labels, tokenize the text, then assign labels based on the spans defined
    """
    doc = pipe(source)
    sentences = doc.sentences
    for sentence in sentences:
        for token in sentence.tokens:
            token.ner = "O"

    for label in labels:
        ner = label['label']
        start_offset = label['startOffset']
        end_offset = label['endOffset']
        for sentence in sentences:
            if (sentence.tokens[0].start_char <= start_offset and
                sentence.tokens[-1].end_char >= end_offset):
                # found the sentence!
                break
        else: # for-else again!  deal with it
            continue

        start_token = None
        end_token = None
        for token_idx, token in enumerate(sentence.tokens):
            if token.start_char <= start_offset and token.end_char > start_offset:
                # ideally we'd have start_char == start_offset, but maybe our
                # tokenization doesn't match the tokenization of the annotators
                start_token = token
                start_token.ner = "B-" + ner
            elif start_token is not None:
                if token.start_char >= end_offset and token_idx > 0:
                    end_token = sentence.tokens[token_idx-1]
                    break
                if token.end_char == end_offset and token_idx > 0 and token.text in (',', '.'):
                    end_token = sentence.tokens[token_idx-1]
                    break
                token.ner = "I-" + ner
            if token.end_char >= end_offset and end_token is None:
                end_token = token
                break
        if start_token is None or end_token is None:
            raise AssertionError("This should not happen")

    return [[(token.text, token.ner) for token in sentence.tokens] for sentence in sentences]



def main(args):
    """
    Read in a .json file of labeled data from AMT, write out a converted .bio file

    Enforces that there is only one set of labels on a sentence
    (TODO: add an option to skip certain sets of labels)
    """
    docs = read_json(args.input_path)

    if len(docs) == 0:
        print("Error: no documents found in the input file!")
        return

    docs = remove_ignored_labels(docs, args.ignore)
    docs = remap_labels(docs, args.remap)
    docs = remove_nesting(docs)

    pipe = stanza.Pipeline(args.language, processors="tokenize")
    sentences = []
    for doc in tqdm(docs):
        sentences.extend(process_doc(*doc, pipe))
    print("Found %d total sentences (may be more than #docs if a doc has more than one sentence)" % len(sentences))
    bio_filename = args.output_path
    write_sentences(args.output_path, sentences)
    print("Sentences written to %s" % args.output_path)
    if bio_filename.endswith(".bio"):
        json_filename = bio_filename[:-4] + ".json"
    else:
        json_filename = bio_filename + ".json"
    prepare_ner_file.process_dataset(bio_filename, json_filename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default="en", help="Language to process")
    parser.add_argument('--input_path', type=str, default="output.manifest", help="Where to find the files")
    parser.add_argument('--output_path', type=str, default="data/ner/en_amt.test.bio", help="Where to output the results")
    parser.add_argument('--json_output_path', type=str, default=None, help="Where to output .json.  Best guess will be made if there is no .json file")
    parser.add_argument('--ignore', type=str, default=None, help="Ignore these labels: comma separated list without B- or I-")
    parser.add_argument('--remap', type=str, default=None, help="Remap labels: comma separated list of X=Y")
    args = parser.parse_args()

    main(args)
