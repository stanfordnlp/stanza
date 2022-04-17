"""
Based on
https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
"""

import math

import torch
from torch import nn

class SinusoidalEncoding(nn.Module):
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
            self.register_buffer('pe', self.build_position(self.pe.shape[1], max(x)+1, device=self.pe.device))
        return self.pe[x]

    def max_len(self):
        return self.pe.shape[0]


class ConcatSinusoidalEncoding(nn.Module):
    """
    Uses sine & cosine to represent position
    """
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.encoding = SinusoidalEncoding(d_model // 2, max_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        timing = self.encoding(torch.arange(x.shape[1], device=x.device))
        x, timing = torch.broadcast_tensors(x, timing)
        out = torch.cat([x, timing], dim=-1)
        out = self.norm(out)
        return out
