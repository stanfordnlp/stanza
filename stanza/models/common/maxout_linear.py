"""
A layer which implements maxout from the "Maxout Networks" paper

https://arxiv.org/pdf/1302.4389v4.pdf
Goodfellow, Warde-Farley, Mirza, Courville, Bengio

or a simpler explanation here:

https://stats.stackexchange.com/questions/129698/what-is-maxout-in-neural-network/298705#298705

The implementation here:
for k layers of maxout, in -> out channels, we make a single linear
  map of size in -> out*k
then we reshape the end to be (..., k, out)
and return the max over the k layers
"""


import torch
import torch.nn as nn

class MaxoutLinear(nn.Module):
    def __init__(self, in_channels, out_channels, maxout_k):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxout_k = maxout_k

        self.linear = nn.Linear(in_channels, out_channels * maxout_k)

    def forward(self, inputs):
        """
        Use the oversized linear as the repeated linear, then take the max

        One large linear map makes the implementation simpler and easier for pytorch to make parallel
        """
        outputs = self.linear(inputs)
        outputs = outputs.view(*outputs.shape[:-1], self.maxout_k, self.out_channels)
        outputs = torch.max(outputs, dim=-2)[0]
        return outputs

