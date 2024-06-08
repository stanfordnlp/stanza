"""
Constant values for model building and execution
"""

import os 

# for model.py
DEFAULT_ENCODER_HIDDEN_DIM = 128
DEFAULT_ENCODER_NUM_LAYERS = 1
DEFAULT_DECODER_HIDDEN_DIM = 128
DEFAULT_DECODER_NUM_LAYERS = 1
UNK_ID = 1
UNK = "<UNK>"
START_TOKEN = "<s>"
STOP_TOKEN = "</s>"

DEFAULT_BATCH_SIZE = 16

DEFAULT_EVAL_ROOT = os.path.join(os.path.dirname(__file__), "data", "validation")
DEFAULT_TRAIN_ROOT = os.path.join(os.path.dirname(__file__), "data", "train")
DEFAULT_SAVE_NAME = os.path.join(os.path.dirname(__file__), "saved_models", "summarization_model.pt")
DEFAULT_WORDVEC_PRETRAIN_FILE = os.path.join(os.path.dirname(__file__), "pretrain", "en", "glove.pt")
