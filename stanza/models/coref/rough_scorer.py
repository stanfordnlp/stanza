""" Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import Tuple

import torch

from coref.config import Config


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

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        top_scores, indices = torch.topk(rough_scores,
                                         k=min(self.k, len(rough_scores)),
                                         dim=1, sorted=False)
        return top_scores, indices
