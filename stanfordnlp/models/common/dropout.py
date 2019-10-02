import torch
import torch.nn as nn

class WordDropout(nn.Module):
    """ A word dropout layer that's designed for embedded inputs (e.g., any inputs to an LSTM layer).
    Given a batch of embedded inputs, this layer randomly set some of them to be a replacement state.
    Note that this layer assumes the last dimension of the input to be the hidden dimension of a unit.
    """
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

class LockedDropout(nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """
    def __init__(self, dropprob, batch_first=True, inplace=False):
        super().__init__()
        self.dropprob = dropprob
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or self.dropprob == 0:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropprob)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropprob)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropprob)
        mask = mask.expand_as(x)
        return mask * x

class SequenceUnitDropout(nn.Module):
    """ A unit dropout layer that's designed for input of sequence units (e.g., word sequence, char sequence, etc.).
    Given a sequence of unit indices, this layer randomly set some of them to be a replacement id (usually set to be <UNK>).
    """
    def __init__(self, dropprob, replacement_id):
        super().__init__()
        self.dropprob = dropprob
        self.replacement_id = replacement_id

    def forward(self, x):
        """ :param: x must be a LongTensor of unit indices. """
        if not self.training or self.dropprob == 0:
            return x
        masksize = [y for y in x.size()]
        dropmask = torch.rand(*masksize, device=x.device) < self.dropprob
        res = x.masked_fill(dropmask, self.replacement_id)
        return res

