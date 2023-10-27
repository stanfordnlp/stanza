from enum import Enum

UNKNOWN_TOKEN = "unk"  # token name for unknown tokens
UNKNOWN_TOKEN_IDX = -1   # custom index we apply to unknown tokens

# TODO: ModelType could just be LSTM and TRANSFORMER
# and then the transformer baseline would have the transformer as another argument
class ModelType(Enum):
    LSTM               = 1
    TRANSFORMER        = 2
    BERT               = 3
    ROBERTA            = 4

DEFAULT_BATCH_SIZE = 16