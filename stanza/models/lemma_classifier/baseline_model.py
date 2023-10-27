"""
Baseline model for the existing lemmatizer which always predicts "be" and never "have" on the "'s" token.

The BaselineModel class can be updated to any arbitrary token and predicton lemma, not just "be" on the "s" token.
"""

import stanza
import os
from stanza.models.lemma_classifier.evaluate_models import evaluate_sequences
from stanza.models.lemma_classifier.prepare_dataset import load_doc_from_conll_file

class BaselineModel:

    def __init__(self, token_to_lemmatize, prediction_lemma, prediction_upos):
        self.token_to_lemmatize = token_to_lemmatize
        self.prediction_lemma = prediction_lemma
        self.prediction_upos = prediction_upos

    def predict(self, token):
        if token == self.token_to_lemmatize:
            return self.prediction_lemma

    def evaluate(self, conll_path):
        """
        Evaluates the baseline model against the test set defined in conll_path.

        Returns a map where the keys are each class and the values are another map including the precision, recall and f1 scores
        for that class.

        Also returns confusion matrix. Keys are gold tags and inner keys are predicted tags
        """
        doc = load_doc_from_conll_file(conll_path)
        gold_tag_sequences, pred_tag_sequences = [], []
        for sentence in doc.sentences:
            gold_tags, pred_tags = [], []
            for word in sentence.words:
                if word.upos in self.prediction_upos and word.text == self.token_to_lemmatize:
                    pred = self.prediction_lemma
                    gold = word.lemma
                    gold_tags.append(gold)
                    pred_tags.append(pred)
            gold_tag_sequences.append(gold_tags)
            pred_tag_sequences.append(pred_tags)

        multiclass_result, confusion_mtx, weighted_f1 = evaluate_sequences(gold_tag_sequences, pred_tag_sequences)
        return multiclass_result, confusion_mtx


if __name__ == "__main__":

    bl_model = BaselineModel("'s", "be", ["AUX"])
    coNLL_path = os.path.join(os.path.dirname(__file__), "en_gum-ud-train.conllu")
    bl_model.evaluate(coNLL_path)

