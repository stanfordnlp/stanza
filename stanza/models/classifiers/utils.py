from enum import Enum
import torch

"""
Defines some args which are common between the classifier model(s) and tools which use them
"""
# NLP machines:
# word2vec are in
# /u/nlp/data/stanfordnlp/model_production/stanfordnlp/extern_data/word2vec
# google vectors are in
# /scr/nlp/data/wordvectors/en/google/GoogleNews-vectors-negative300.txt

class WVType(Enum):
    WORD2VEC = 1
    GOOGLE = 2
    FASTTEXT = 3
    OTHER = 4

class ExtraVectors(Enum):
    NONE = 1
    CONCAT = 2
    SUM = 3
