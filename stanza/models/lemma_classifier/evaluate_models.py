# TODO: Figure out how to load in the UD files into Stanza objects to get the features from them.
import os 
import sys 

parentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

import stanza
from typing import Any, List, Tuple
from models.lemma_classifier.baseline_model import BaselineModel
import utils


def update_counts(gold_tag: str, pred_tag: str, true_pos: int, false_pos: int, false_neg: int) -> Tuple[int, int, int]:
    """"
    Takes in a prediction along with the counts for true positive, false positive and false negative and updates the counts
    of the measurements according to the prediction.

    We measure positives, where we treat "be" as a positive and "have" as a negative.
    """
    if gold_tag == "be" and pred_tag == "be":
        true_pos += 1
    elif gold_tag == "be" and pred_tag == "have": 
        false_neg += 1
    elif gold_tag == "have" and pred_tag == "be":
        false_pos += 1
    return true_pos, false_pos, false_neg


def evaluate_models(eval_path: str, binary_classifier: Any, baseline_classifier: BaselineModel):
    """
    Evaluates both the binary classifier and baseline classifier on a test file,
    checking the predicted lemmas for each "'s" token against the gold lemma.

    TODO: Measure precision, recall, and F1.

    Precision = true positives / true positives + false positives
    Recall = true positives / true positives + false negatives
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    gold_doc = utils.load_doc_from_conll_file(eval_path)

    bin_tp, bin_fp, bin_fn = 0, 0, 0
    bl_tp, bl_fp, bl_fn = 0, 0, 0  # baseline counts

    for sentence in gold_doc.sentences:
        for word in sentence.words:
            if word.text == "'s" and word.upos == "AUX":   # only evaluate when the UPOS tag is AUX
                gold_tag = word.lemma
                # predict binary classifier
                bin_predict = None  # TODO
                # predict baseline classifier
                baseline_predict = baseline_classifier.predict(word.text)  # TODO
                
                # score binary classifier
                bin_tp, bin_fp, bin_fn = update_counts(gold_tag, bin_predict, bin_tp, bin_fp, bin_fn)
                bl_tp, bl_fp, bl_fn = update_counts(gold_tag, baseline_predict, bl_tp, bl_fp, bl_fn)

    # compute precision, recall, f1
    bin_precision, bin_recall =  bin_tp / (bin_tp + bin_fp), bin_tp / (bin_tp + bin_fn)
    bin_results = {"precision":  bin_precision,
                    "recall": bin_recall,
                    "f1": 2 * (bin_precision * bin_recall) / (bin_precision + bin_recall)
                 }

    bl_precision, bl_recall =  bl_tp / (bl_tp + bl_fp), bl_tp / (bl_tp + bl_fn)
    bl_results = {"precision":  bl_precision,
                    "recall": bl_recall,
                    "f1": 2 * (bl_precision * bl_recall) / (bl_precision + bl_recall)
                  }
    
    return bin_results, bl_results


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

    # bl_model = BaselineModel("'s", "be")
    # bin_results, bl_results = evaluate_models(coNLL_path, None, bl_model)
    # print(bin_results, bl_results)


if __name__ == "__main__":
    main()
