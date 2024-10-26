from enum import Enum

from torch import nn

"""
Defines some methods which may occur in multiple model types
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

class ModelType(Enum):
    CNN = 1
    CONSTITUENCY = 2

def build_output_layers(fc_input_size, fc_shapes, num_classes):
    """
    Build a sequence of fully connected layers to go from the final conv layer to num_classes

    Returns an nn.ModuleList
    """
    fc_layers = []
    previous_layer_size = fc_input_size
    for shape in fc_shapes:
        fc_layers.append(nn.Linear(previous_layer_size, shape))
        previous_layer_size = shape
    fc_layers.append(nn.Linear(previous_layer_size, num_classes))
    return nn.ModuleList(fc_layers)
