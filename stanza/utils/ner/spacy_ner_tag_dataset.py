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
from stanza.utils.datasets.ner.simplify_ontonotes_to_worldwide import simplify_ontonotes_to_worldwide

from stanza.utils.get_tqdm import get_tqdm
tqdm = get_tqdm()

"""
Simplified classes used in the Worldwide dataset are:

Date
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
PERSON
PRODUCT
QUANTITY
TIME
WORK_OF_ART
"""

def test_file(eval_file, tagger, simplify):
    with open(eval_file) as fin:
        gold_doc = json.load(fin)
    gold_doc = [[(x['text'], x['ner']) for x in sentence] for sentence in gold_doc]
    gold_doc = process_tags(gold_doc, 'bioes')

    if simplify:
        for doc in gold_doc:
            for idx, word in enumerate(doc):
                if word[1] != "O":
                    word = [word[0], simplify_ontonotes_to_worldwide(word[1])]
                    doc[idx] = word

    ignore_tags = "Date,DATE" if simplify else None

    original_text = [[x[0] for x in gold_sentence] for gold_sentence in gold_doc]
    pred_doc = []
    for sentence in tqdm(original_text):
        spacy_sentence = Doc(tagger.vocab, sentence)
        spacy_sentence = tagger(spacy_sentence)
        entities = ["O" if not token.ent_type_ else "%s-%s" % (token.ent_iob_, token.ent_type_) for token in spacy_sentence]
        if simplify:
            entities = [simplify_ontonotes_to_worldwide(x) for x in entities]
        pred_sentence = [[token.text, entity] for token, entity in zip(spacy_sentence, entities)]
        pred_doc.append(pred_sentence)

    pred_doc = process_tags(pred_doc, 'bioes')
    pred_tags = [[x[1] for x in sentence] for sentence in pred_doc]
    gold_tags = [[x[1] for x in sentence] for sentence in gold_doc]
    print("RESULTS ON: %s" % eval_file)
    _, _, f_micro = score_by_entity(pred_tags, gold_tags, ignore_tags=ignore_tags)
    _, _, _, confusion = score_by_token(pred_tags, gold_tags, ignore_tags=ignore_tags)
    print("NER token confusion matrix:\n{}".format(format_confusion(confusion, hide_blank=True, transpose=True)))
    return f_micro

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_model', type=str, default=None,  help='Which spacy model to test')
    parser.add_argument('filename', type=str, nargs='*', help='which files to test')
    parser.add_argument('--simplify', default=False, action='store_true', help='Simplify classes to the 8 class Worldwide model')
    args = parser.parse_args()

    if args.ner_model is None:
        ner_models = ['en_core_web_sm', 'en_core_web_trf']
    else:
        ner_models = [args.ner_model]

    if not args.filename:
        args.filename = ["data/ner/en_ontonotes-8class.test.json",
                         "data/ner/en_worldwide-8class.test.json",
                         "data/ner/en_worldwide-8class-africa.test.json",
                         "data/ner/en_worldwide-8class-asia.test.json",
                         "data/ner/en_worldwide-8class-indigenous.test.json",
                         "data/ner/en_worldwide-8class-latam.test.json",
                         "data/ner/en_worldwide-8class-middle_east.test.json"]

    print("Processing the files: %s" % ",".join(args.filename))

    results = []
    model_results = {}

    for ner_model in ner_models:
        model_results[ner_model] = []
        # load tagger
        print("-----------------------------")
        print("Running %s" % ner_model)
        print("-----------------------------")
        tagger = spacy.load(ner_model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

        for filename in args.filename:
            f_micro = test_file(filename, tagger, args.simplify)
            f_micro = "%.2f" % (f_micro * 100)
            results.append((ner_model, filename, f_micro))
            model_results[ner_model].append(f_micro)

    for result in results:
        print(result)

    for model in model_results.keys():
        result = [model] + model_results[model]
        print(" & ".join(result))

if __name__ == '__main__':
    main()
