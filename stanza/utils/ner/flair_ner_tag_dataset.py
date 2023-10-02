"""
Test a flair model on a 4 class dataset
"""

import argparse
import json

from flair.data import Sentence
from flair.models import SequenceTagger

from stanza.models.ner.utils import process_tags
from stanza.models.ner.scorer import score_by_entity, score_by_token

def test_file(eval_file, tagger):
    with open(eval_file) as fin:
        gold_doc = json.load(fin)
    gold_doc = [[(x['text'], x['ner']) for x in sentence] for sentence in gold_doc]
    gold_doc = process_tags(gold_doc, 'bioes')

    pred_doc = []
    for gold_sentence in gold_doc:
        pred_sentence = [[x[0], 'O'] for x in gold_sentence]
        flair_sentence = Sentence(" ".join(x[0] for x in pred_sentence), use_tokenizer=False)
        tagger.predict(flair_sentence)

        for entity in flair_sentence.get_spans('ner'):
            tag = entity.tag
            tokens = entity.tokens
            start_idx = tokens[0].idx - 1
            end_idx = tokens[-1].idx
            if len(tokens) == 1:
                pred_sentence[start_idx][1] = "S-" + tag
            else:
                pred_sentence[start_idx][1] = "B-" + tag
                pred_sentence[end_idx - 1][1] = "E-" + tag
                for idx in range(start_idx+1, end_idx - 1):
                    pred_sentence[idx][1] = "I-" + tag

        pred_doc.append(pred_sentence)

    pred_tags = [[x[1] for x in sentence] for sentence in pred_doc]
    gold_tags = [[x[1] for x in sentence] for sentence in gold_doc]
    print("RESULTS ON: %s" % eval_file)
    _, _, f_micro = score_by_entity(pred_tags, gold_tags)
    score_by_token(pred_tags, gold_tags)
    return f_micro

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_model', type=str, default=None,  help='Which NER model to test')
    parser.add_argument('filename', type=str, nargs='*', help='which files to test')
    args = parser.parse_args()

    if args.ner_model is None:
        ner_models = ["ner-fast", "ner", "ner-large"]
    else:
        ner_models = [args.ner_model]

    if not args.filename:
        args.filename = ["data/ner/en_conll03.test.json",
                         "data/ner/en_worldwide-4class.test.json",
                         "data/ner/en_worldwide-4class-africa.test.json",
                         "data/ner/en_worldwide-4class-asia.test.json",
                         "data/ner/en_worldwide-4class-indigenous.test.json",
                         "data/ner/en_worldwide-4class-latam.test.json",
                         "data/ner/en_worldwide-4class-middle_east.test.json"]

    print("Processing the files: %s" % ",".join(args.filename))

    results = []
    model_results = {}

    for ner_model in ner_models:
        model_results[ner_model] = []

        # load tagger
        #tagger = SequenceTagger.load("ner-fast")
        print("-----------------------------")
        print("Running %s" % ner_model)
        print("-----------------------------")
        tagger = SequenceTagger.load(ner_model)

        for filename in args.filename:
            f_micro = test_file(filename, tagger)
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
