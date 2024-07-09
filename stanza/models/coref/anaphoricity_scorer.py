""" Describes AnaphicityScorer, a torch module that for a matrix of
mentions produces their anaphoricity scores.
"""
import torch

from stanza.models.coref import utils
from stanza.models.coref.config import Config


class AnaphoricityScorer(torch.nn.Module):
    """ Calculates anaphoricity scores by passing the inputs into a FFNN """
    def __init__(self,
                 in_features: int,
                 config: Config):
        super().__init__()
        hidden_size = config.hidden_size
        if not config.n_hidden_layers:
            hidden_size = in_features
        layers = []
        for i in range(config.n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_size if i else in_features,
                                           hidden_size),
                           torch.nn.LeakyReLU(),
                           torch.nn.Dropout(config.dropout_rate)])
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_size, out_features=1)

        # are we going to predict singletons
        self.predict_singletons = config.singletons

        if self.predict_singletons:
            # map to whether or not this is a start of a coref given all the
            # antecedents; not used when config.singletons = False because
            # we only need to know this for predicting singletons
            self.start_map = torch.nn.Linear(config.rough_k, out_features=1, bias=False)


    def forward(self, *,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                top_mentions: torch.Tensor,
                mentions_batch: torch.Tensor,
                pw_batch: torch.Tensor,
                top_rough_scores_batch: torch.Tensor,
                ) -> torch.Tensor:
        """ Builds a pairwise matrix, scores the pairs and returns the scores.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb]
            mentions_batch (torch.Tensor): [batch_size, mention_emb]
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb]
            top_indices_batch (torch.Tensor): [batch_size, n_ants]
            top_rough_scores_batch (torch.Tensor): [batch_size, n_ants]

        Returns:
            torch.Tensor [batch_size, n_ants + 1]
                anaphoricity scores for the pairs + a dummy column
        """
        # [batch_size, n_ants, pair_emb]
        pair_matrix = self._get_pair_matrix(mentions_batch, pw_batch, top_mentions)

        # [batch_size, n_ants] vs [batch_size, 1]
        # first is coref scores, the second is whether its the start of a coref
        if self.predict_singletons:
            scores, start = self._ffnn(pair_matrix)
            scores = utils.add_dummy(scores+top_rough_scores_batch, eps=True)

            return torch.cat([start, scores], dim=1)
        else:
            scores = self._ffnn(pair_matrix)
            return utils.add_dummy(scores+top_rough_scores_batch, eps=True)

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.out(self.hidden(x))
        x = x.squeeze(2)

        if not self.predict_singletons:
            return x

        # because sometimes we only have the first 49 anaphoricities
        start = x @ self.start_map.weight[:,:x.shape[1]].T
        return x, start

    @staticmethod
    def _get_pair_matrix(mentions_batch: torch.Tensor,
                         pw_batch: torch.Tensor,
                         top_mentions: torch.Tensor) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb],
                all the valid mentions of the document,
                can be on a different device
            mentions_batch (torch.Tensor): [batch_size, mention_emb],
                the mentions of the current batch,
                is expected to be on the current device
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb],
                pairwise features of the current batch,
                is expected to be on the current device
            top_indices_batch (torch.Tensor): [batch_size, n_ants],
                indices of antecedents of each mention

        Returns:
            torch.Tensor: [batch_size, n_ants, pair_emb]
        """
        emb_size = mentions_batch.shape[1]
        n_ants = pw_batch.shape[1]

        a_mentions = mentions_batch.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = top_mentions
        similarity = a_mentions * b_mentions

        out = torch.cat((a_mentions, b_mentions, similarity, pw_batch), dim=2)
        return out
