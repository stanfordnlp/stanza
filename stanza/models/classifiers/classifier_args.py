from enum import Enum
import torch

"""
Defines some args which are common between the classifier model(s) and tools which use them
"""

class WVType(Enum):
    WORD2VEC = 1
    GOOGLE = 2
    FASTTEXT = 3
    OTHER = 4

class ExtraVectors(Enum):
    NONE = 1
    CONCAT = 2
    SUM = 3

# NLP machines:
# word2vec are in
# /u/nlp/data/stanfordnlp/model_production/stanfordnlp/extern_data/word2vec
# google vectors are in
# /scr/nlp/data/wordvectors/en/google/GoogleNews-vectors-negative300.txt

def add_common_args(parser):
    parser.add_argument('--seed', default=None, type=int, help='Random seed for model')
    add_pretrain_args(parser)
    add_device_args(parser)

def add_pretrain_args(parser):
    parser.add_argument('--save_dir', type=str, default='saved_models/classifier', help='Root dir for saving models.')
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--wordvec_raw_file', type=str, default=None, help='Exact name of the raw wordvec file to read')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors')
    parser.add_argument('--wordvec_type', type=lambda x: WVType[x.upper()], default='word2vec', help='Different vector types have different options, such as google 300d replacing numbers with #')
    parser.add_argument('--shorthand', type=str, default='en_ewt', help="Treebank shorthand, eg 'en' for English")
    parser.add_argument('--extra_wordvec_dim', type=int, default=0, help="Extra dim of word vectors - will be trained")
    parser.add_argument('--extra_wordvec_method', type=lambda x: ExtraVectors[x.upper()], default='none', help='How to train extra dimensions of word vectors, if at all')
    parser.add_argument('--extra_wordvec_max_norm', type=float, default=None, help="Max norm for initializing the extra vectors")

def add_device_args(parser):
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training/testing', default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_false', help='Ignore CUDA.', dest='cuda')

