"""
LargeMarginInSoftmax, from the article

@inproceedings{kobayashi2019bmvc,
  title={Large Margin In Softmax Cross-Entropy Loss},
  author={Takumi Kobayashi},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2019}
}

implementation from

https://github.com/tk1980/LargeMarginInSoftmax

There is no license specifically chosen; they just ask people to cite the paper if the work is useful.
"""


import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    r"""
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    regularization proposed in
       T. Kobayashi, "Large-Margin In Softmax Cross-Entropy Loss." In BMVC2019.

    This loss function inherits the parameters from nn.CrossEntropyLoss except for `reg_lambda` and `deg_logit`.
    Args:
         reg_lambda (float, optional): a regularization parameter. (default: 0.3)
         deg_logit (bool, optional): underestimate (degrade) the target logit by -1 or not. (default: False)
                                     If True, it realizes the method that incorporates the modified loss into ours
                                     as described in the above paper (Table 4).
    """
    def __init__(self, reg_lambda=0.3, deg_logit=None,
                weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginInSoftmaxLoss, self).__init__(weight=weight, size_average=size_average,
                                ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0) # number of samples
        C = input.size(1) # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N),target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0/(C-1)) * F.log_softmax(X, dim=1) * (1.0-Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg
