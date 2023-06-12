"""
Test a spacy model on a 4 class dataset
"""

import argparse
import json

import spacy
from spacy.tokens import Doc

from stanza.models.ner.utils import process_tags
from stanza.models.ner.scorer import score_by_entity, score_by_token

from stanza.utils.confusion import format_confusion

from tqdm import tqdm

"""
Simplified classes are:

Facility
Location
Misc
Money
NORP
Organization
Person
Product

vs OntoNotes classes:

CARDINAL
DATE
EVENT
FAC
GPE
LANGUAGE
LAW
LOC
MONEY
NORP
ORDINAL
ORG
PERCENT
PERSON"
PRODUCT
QUANTITY
TIME
WORK_OF_ART

"""

ENTITY_MAPPING = {
    "CARDINAL":    None,
    "ORDINAL":     None,
    "PERCENT":     None,
    "QUANTITY":    None,
    "TIME":        None,

    "DATE":        "Misc",
    "EVENT":       "Misc",
    "FAC":         "Facility",
    "GPE":         "Location",
    "LANGUAGE":    "NORP",
    "LAW":         "Misc",
    "LOC":         "Location",
    "MONEY":       "Money",
    "NORP":        "NORP",
    "ORG":         "Organization",
    "PERSON":      "Person",
    "PRODUCT":     "Product",
    "WORK_OF_ART": "Misc",

    "Facility":    "Facility",
    "Location":    "Location",
    "Misc":        "Misc",
    "Money":       "Money",
    "Organization":"Organization",
    "Person":      "Person",
    "Product":     "Product",
}

def simplify_entity(ent_iob, ent_type, simplify):
    if not ent_type:
        return "O"

    if not simplify:
        return ent_iob + "-" + ent_type

    if ent_type in ENTITY_MAPPING:
        if not ENTITY_MAPPING[ent_type]:
            return "O"
        return ent_iob + "-" + ENTITY_MAPPING[ent_type]
    raise ValueError("Unhandled entity: %s" % ent_type)

def test_file(eval_file, tagger, simplify):
    with open(eval_file) as fin:
        gold_doc = json.load(fin)
    gold_doc = [[(x['text'], x['ner']) for x in sentence] for sentence in gold_doc]
    gold_doc = process_tags(gold_doc, 'bioes')

    if simplify:
        for doc in gold_doc:
            for idx, word in enumerate(doc):
                if word[1] != "O":
                    pieces = word[1].split("-")
                    word = [word[0], simplify_entity(pieces[0], pieces[1], True)]
                    doc[idx] = word

    original_text = [[x[0] for x in gold_sentence] for gold_sentence in gold_doc]
    pred_doc = []
    for sentence in tqdm(original_text):
        spacy_sentence = Doc(tagger.vocab, sentence)
        spacy_sentence = tagger(spacy_sentence)
        pred_sentence = [[token.text, simplify_entity(token.ent_iob_, token.ent_type_, simplify)]
                         for token in spacy_sentence]
        pred_doc.append(pred_sentence)

    pred_doc = process_tags(pred_doc, 'bioes')
    pred_tags = [[x[1] for x in sentence] for sentence in pred_doc]
    gold_tags = [[x[1] for x in sentence] for sentence in gold_doc]
    print("RESULTS ON: %s" % eval_file)
    score_by_entity(pred_tags, gold_tags)
    _, _, _, confusion = score_by_token(pred_tags, gold_tags)
    print("NER token confusion matrix:\n{}".format(format_confusion(confusion, hide_blank_rows=True)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_model', type=str, default='en_core_web_sm',  help='Which spacy model to test')
    parser.add_argument('filename', type=str, nargs='+', help='which files to test')
    parser.add_argument('--simplify', default=False, action='store_true', help='Simplify classes to the 8 class Worldwide model')
    args = parser.parse_args()

    # load tagger
    # there is also en_core_web_trf
    tagger = spacy.load(args.ner_model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])


    for filename in args.filename:
        test_file(filename, tagger, args.simplify)

if __name__ == '__main__':
    main()
