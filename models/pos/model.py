import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.biaffine import BiaffineScorer
from models.common.hlstm import HighwayLSTM

class Tagger(nn.Module):
    def __init__(self, args):
        super().__init__()
