"""
Based on
https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
"""

import math

import torch
from torch import nn

class SinusoidalEncoding(nn.Module):
    """
    Uses sine & cosine to represent position
    """
    def __init__(self, model_dim, max_len):
        super().__init__()
        self.register_buffer('pe', self.build_position(model_dim, max_len))

    @staticmethod
    def build_position(model_dim, max_len, device=None):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if device is not None:
            pe = pe.to(device=device)
        return pe

    def forward(self, x):
        if max(x) >= self.pe.shape[0]:
            # try to drop the reference first before creating a new encoding
            # the goal being to save memory if we are close to the memory limit
            device = self.pe.device
            shape = self.pe.shape[1]
            self.register_buffer('pe', None)
            # TODO: this may result in very poor performance
            # in the event of a model that increases size one at a time
            self.register_buffer('pe', self.build_position(shape, max(x)+1, device=device))
        return self.pe[x]

    def max_len(self):
        return self.pe.shape[0]


class AddSinusoidalEncoding(nn.Module):
    """
    Uses sine & cosine to represent position.  Adds the position to the given matrix

    Default behavior is batch_first
    """
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.encoding = SinusoidalEncoding(d_model, max_len)

    def forward(self, x, scale=1.0):
        """
        Adds the positional encoding to the input tensor

        The tensor is expected to be of the shape B, N, D
        Properly masking the output tensor is up to the caller
        """
        if len(x.shape) == 3:
            timing = self.encoding(torch.arange(x.shape[1], device=x.device))
            timing = timing.expand(x.shape[0], -1, -1)
        elif len(x.shape) == 2:
            timing = self.encoding(torch.arange(x.shape[0], device=x.device))
        return x + timing * scale


class ConcatSinusoidalEncoding(nn.Module):
    """
    Uses sine & cosine to represent position.  Concats the position and returns a larger object

    Default behavior is batch_first
    """
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.encoding = SinusoidalEncoding(d_model, max_len)

    def forward(self, x):
        if len(x.shape) == 3:
            timing = self.encoding(torch.arange(x.shape[1], device=x.device))
            timing = timing.expand(x.shape[0], -1, -1)
        else:
            timing = self.encoding(torch.arange(x.shape[0], device=x.device))

        out = torch.cat((x, timing), dim=-1)
        return out
