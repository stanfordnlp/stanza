import torch.nn as nn

class BroadcastDropout(nn.Module):
    def __init__(self, dropout=0, dims=[1]):
        super().__init__()

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.dims = dims

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x

        masksize = [s for s in x.size()]
        for d in self.dims:
            masksize[d] = 1
        dropmask = self.drop(x.new_ones(*masksize))
        return x * dropmask
