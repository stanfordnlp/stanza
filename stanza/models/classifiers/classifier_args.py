from enum import Enum
import torch

"""
Defines some args which are common between the classifier model(s) and tools which use them
"""

class WVType(Enum):
    WORD2VEC = 1
    GOOGLE = 2
    FASTTEXT = 3

# NLP machines:
# word2vec are in
# /u/nlp/data/stanfordnlp/model_production/stanfordnlp/extern_data/word2vec
# google vectors are in
# /scr/nlp/data/wordvectors/en/google/GoogleNews-vectors-negative300.txt

def add_pretrain_args(parser):
    parser.add_argument('--save_dir', type=str, default='saved_models/classifier', help='Root dir for saving models.')
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--wordvec_dir', type=str, default='extern_data', help='Directory of word vectors')
    parser.add_argument('--wordvec_type', type=lambda x: WVType[x.upper()], default='word2vec', help='Different vector types have different options, such as google 300d replacing numbers with #')
    # TODO: the second particle should reflect the actual treebank, eg SST for sentiment treebank
    parser.add_argument('--shorthand', type=str, default='en_ewt', help="Treebank shorthand, eg 'en' for English")

def add_device_args(parser):
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training/testing', default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_false', help='Ignore CUDA.', dest='cuda')

