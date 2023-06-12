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
    score_by_entity(pred_tags, gold_tags)
    score_by_token(pred_tags, gold_tags)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_model', type=str, default='ner-fast',  help='Which NER model to test')
    parser.add_argument('filename', type=str, nargs='+', help='which files to test')
    args = parser.parse_args()

    # load tagger
    #tagger = SequenceTagger.load("ner-fast")
    tagger = SequenceTagger.load(args.ner_model)

    for filename in args.filename:
        test_file(filename, tagger)

if __name__ == '__main__':
    main()
