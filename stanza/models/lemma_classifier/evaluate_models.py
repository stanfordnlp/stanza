import os 
import sys 

parentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import stanza
from typing import Any, List, Tuple
import utils
from collections import defaultdict


def evaluate_sequences(gold_tag_sequences, pred_tag_sequences, verbose=True):
    """
    Evaluates a model's predicted tags against a set of gold tags. Computes precision, recall, and f1 for all classes.

    Precision = true positives / true positives + false positives
    Recall = true positives / true positives + false negatives
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    assert len(gold_tag_sequences) == len(pred_tag_sequences), \
    f"Length of gold tag sequences is {len(gold_tag_sequences)}, while length of predicted tag sequence is {len(pred_tag_sequences)}"        
    
    confusion = defaultdict(lambda: defaultdict(int))
    
    for gold_tags, pred_tags in zip(gold_tag_sequences, pred_tag_sequences):

        assert len(gold_tags) == len(pred_tags), f"Number of gold tags doesn't match number of predicted tags ({len(gold_tags)}, {len(pred_tags)})"
        for gold, pred in zip(gold_tags, pred_tags):
            confusion[gold][pred] += 1

    multi_class_result = defaultdict(lambda: defaultdict(float))
    # compute precision, recall and f1 for each class and store inside of `multi_class_result`
    for gold_tag in confusion.keys():

        try:
            prec = confusion.get(gold_tag, {}).get(gold_tag, 0) / sum([confusion.get(k, {}).get(gold_tag, 0) for k in confusion.keys()])
        except ZeroDivisionError:
            prec = 0.0 
        
        try:
            recall = confusion.get(gold_tag, {}).get(gold_tag, 0) / sum(confusion.get(gold_tag, {}).values())
        except ZeroDivisionError:
            recall = 0.0

        try:
            f1 = 2 * (prec * recall) / (prec + recall)
        except ZeroDivisionError:
            f1 = 0.0 

        multi_class_result[gold_tag] = {
            "precision": prec,
            "recall": recall,
            "f1": f1
        }
    
    if verbose:
        for lemma in multi_class_result:
            print(f"Lemma '{lemma}' had precision {100 * multi_class_result[lemma]['precision']}, recall {100 * multi_class_result[lemma]['recall']} and F1 score of {100 * multi_class_result[lemma]['f1']}")

    return multi_class_result, confusion   


def main():
    """
    Runs a test on the EN_GUM test set
    """
    coNLL_path = os.path.join(os.path.dirname(__file__), "en_gum-ud-train.conllu")
    print(f"Attempting to find token 's in file {coNLL_path}...")
    doc = utils.load_doc_from_conll_file(coNLL_path)
    count = 0
    be_count, have_count = 0, 0
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text == "'s" and word.upos == "AUX":
                print("---------------------------")
                print(word)
                print("---------------------------")
                if word.lemma == "have":
                    have_count += 1
                if word.lemma == "be":
                    be_count += 1
                count += 1 

    print(f"The number of 's found was {count}.")
    print(f"There were {have_count} occurrences of the lemma being 'have'.")
    print(f"There were {be_count} occurrences of the lemma being 'be'.")


if __name__ == "__main__":
    main()
