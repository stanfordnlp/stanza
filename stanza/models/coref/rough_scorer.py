""" Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import Tuple

import torch

from stanza.models.coref.config import Config


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """
    def __init__(self, features: int, config: Config):
        super().__init__()
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.bilinear = torch.nn.Linear(features, features)

        self.k = config.rough_k

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                mentions: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        pair_mask = torch.arange(mentions.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(mentions.device)

        bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)

        rough_scores = pair_mask + bilinear_scores

        return self._prune(rough_scores)

    def _prune(self,
               rough_scores: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects top-k rough antecedent scores for each mention.

        The AnaphoricityScorer's start_map is a linear layer over the k
        antecedent positions, so the layout of each row must be
        deterministic.  This reproduces the layout of CUDA's
        topk(sorted=False), which is what the existing models were
        trained with: the selected entries strictly greater than the
        kth value in ascending index order, followed by the entries
        equal to the kth value (the kth value itself, plus any ties) in
        ascending index order.  When there are more ties than slots,
        the lowest indices are chosen.

        In practice, for a row with k real antecedents, that is the top
        k-1 in ascending order with the lowest scoring antecedent last.
        The first k rows have fewer than k real antecedents, the rest
        being -inf padding, so they always come out as 0..k-1.

        CPU topk(sorted=False) returns an arbitrary order and an
        arbitrary choice among the -inf padding, hence the rearranging.

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        n_mentions = len(rough_scores)
        k = min(self.k, n_mentions)
        _, indices = torch.topk(rough_scores, k=k, dim=1, sorted=False)
        # the only ties expected are the -inf padding in the first k rows,
        # where the lowest indices are 0..k-1
        indices[:k] = torch.arange(k, device=indices.device)
        top_scores = rough_scores.gather(1, indices)

        # entries equal to the kth value go after everything else,
        # and each group is in ascending index order
        kth_scores = top_scores.min(dim=1, keepdim=True).values
        order = (indices + (top_scores == kth_scores) * n_mentions).argsort(dim=1)
        indices = indices.gather(1, order)
        top_scores = top_scores.gather(1, order)
        return top_scores, indices, rough_scores
