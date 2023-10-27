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


def load_doc_from_conll_file(path: str):
    return stanza.utils.conll.CoNLL.conll2doc(path)


def evaluate_models(eval_path: str, binary_classifier: Any, baseline_classifier: BaselineModel):
    """
    Evaluates both the binary classifier and baseline classifier on a test file,
    checking the predicted lemmas for each "'s" token against the gold lemma.
    """

    gold_doc = load_doc_from_conll_file(eval_path)
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text == "'s" and word.upos == "VERB":
                gold_tag = word.lemma
                # predict binary classifier
                bin_predict = None  # TODO
                # predict baseline classifier
                baseline_predict = baseline_classifier.predict(word.text)  # TODO
                # score
                if gold_tag == bin_predict:
                    pass 
                if gold_tag == baseline_predict:
                    pass

    return  # TODO write some kind of evaluation  


def main():
    """
    Runs a test on the EN_GUM test set
    """
    coNLL_path = os.path.join(os.path.dirname(__file__), "en_gum-ud-test.conllu")
    doc = load_doc_from_conll_file(coNLL_path)
    count = 0
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text == "'s" and word.upos == "VERB":
                print("Found")
                print(word)
                count += 1 

    print(f"Count was {count}.")


if __name__ == "__main__":
    main()
