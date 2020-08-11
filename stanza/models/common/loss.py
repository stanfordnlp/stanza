"""
Different loss functions.
"""

import logging
import numpy as np
import torch
import torch.nn as nn

import stanza.models.common.seq2seq_constant as constant

logger = logging.getLogger('stanza')

def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constant.PAD_ID] = 0
    crit = nn.NLLLoss(weight)
    return crit

def weighted_cross_entropy_loss(labels, log_dampened=False):
    """
    Either return a loss function which reweights all examples so the
    classes have the same effective weight, or dampened reweighting
    using log() so that the biggest class has some priority
    """
    if isinstance(labels, list):
        all_labels = np.array(labels)
    _, weights = np.unique(labels, return_counts=True)
    weights = weights / float(np.sum(weights))
    weights = np.sum(weights) / weights
    if log_dampened:
        weights = 1 + np.log(weights)
    logger.debug("Reweighting cross entropy by {}".format(weights))
    loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).type('torch.FloatTensor')
    )
    return loss

class MixLoss(nn.Module):
    """
    A mixture of SequenceLoss and CrossEntropyLoss.
    Loss = SequenceLoss + alpha * CELoss
    """
    def __init__(self, vocab_size, alpha):
        super().__init__()
        self.seq_loss = SequenceLoss(vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        assert alpha >= 0
        self.alpha = alpha

    def forward(self, seq_inputs, seq_targets, class_inputs, class_targets):
        sl = self.seq_loss(seq_inputs, seq_targets)
        cel = self.ce_loss(class_inputs, class_targets)
        loss = sl + self.alpha * cel
        return loss

class MaxEntropySequenceLoss(nn.Module):
    """
    A max entropy loss that encourage the model to have large entropy,
    therefore giving more diverse outputs.

    Loss = NLLLoss + alpha * EntropyLoss
    """
    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[constant.PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N]
        """
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        # entropy loss
        mask = targets.eq(constant.PAD_ID).unsqueeze(1).expand_as(inputs)
        masked_inputs = inputs.clone().masked_fill_(mask, 0.0)
        p = torch.exp(masked_inputs)
        ent_loss = p.mul(masked_inputs).sum() / inputs.size(0) # average over minibatch
        loss = nll_loss + self.alpha * ent_loss
        return loss

