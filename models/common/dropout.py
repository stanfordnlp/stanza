import torch
import torch.nn as nn

class WordDropout(nn.Module):
    def __init__(self, dropprob):
        super().__init__()
        self.dropprob = dropprob

    def forward(self, x, replacement=None):
        if not self.training or self.dropprob == 0:
            return x

        masksize = [y for y in x.size()]
        masksize[-1] = 1
        dropmask = torch.rand(*masksize, device=x.device) < self.dropprob

        res = x.masked_fill(dropmask, 0)
        if replacement is not None:
            res = res + dropmask.float() * replacement

        return res

