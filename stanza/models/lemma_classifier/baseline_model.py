"""
Baseline model for the existing lemmatizer which always predicts "be" and never "have" on the "'s" token.
"""

import stanza

class BaselineModel:

    def __init__(self, token_to_lemmatize, prediction_lemma):
        self.token_to_lemmatize = token_to_lemmatize
        self.prediction_lemma = prediction_lemma
        

    def predict(self, token):
        if token == self.token_to_lemmatize:
            return self.prediction_lemma
        


