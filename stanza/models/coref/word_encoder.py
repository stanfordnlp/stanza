""" Describes WordEncoder. Extracts mention vectors from bert-encoded text.
"""

from typing import Tuple

import torch

from coref.config import Config
from coref.const import Doc


class WordEncoder(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """ Receives bert contextual embeddings of a text, extracts all the
    possible mentions in that text. """

    def __init__(self, features: int, config: Config):
        """
        Args:
            features (int): the number of featues in the input embeddings
            config (Config): the configuration of the current session
        """
        super().__init__()
        self.attn = torch.nn.Linear(in_features=features, out_features=1)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.attn.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                doc: Doc,
                x: torch.Tensor,
                ) -> Tuple[torch.Tensor, ...]:
        """
        Extracts word representations from text.

        Args:
            doc: the document data
            x: a tensor containing bert output, shape (n_subtokens, bert_dim)

        Returns:
            words: a Tensor of shape [n_words, mention_emb];
                mention representations
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-coreferent words have cluster id of zero.
        """
        word_boundaries = torch.tensor(doc["word2subword"], device=self.device)
        starts = word_boundaries[:, 0]
        ends = word_boundaries[:, 1]

        # [n_mentions, features]
        words = self._attn_scores(x, starts, ends).mm(x)

        words = self.dropout(words)

        return (words, self._cluster_ids(doc))

    def _attn_scores(self,
                     bert_out: torch.Tensor,
                     word_starts: torch.Tensor,
                     word_ends: torch.Tensor) -> torch.Tensor:
        """ Calculates attention scores for each of the mentions.

        Args:
            bert_out (torch.Tensor): [n_subwords, bert_emb], bert embeddings
                for each of the subwords in the document
            word_starts (torch.Tensor): [n_words], start indices of words
            word_ends (torch.Tensor): [n_words], end indices of words

        Returns:
            torch.Tensor: [description]
        """
        n_subtokens = len(bert_out)
        n_words = len(word_starts)

        # [n_mentions, n_subtokens]
        # with 0 at positions belonging to the words and -inf elsewhere
        attn_mask = torch.arange(0, n_subtokens, device=self.device).expand((n_words, n_subtokens))
        attn_mask = ((attn_mask >= word_starts.unsqueeze(1))
                     * (attn_mask < word_ends.unsqueeze(1)))
        attn_mask = torch.log(attn_mask.to(torch.float))

        attn_scores = self.attn(bert_out).T  # [1, n_subtokens]
        attn_scores = attn_scores.expand((n_words, n_subtokens))
        attn_scores = attn_mask + attn_scores
        del attn_mask
        return torch.softmax(attn_scores, dim=1)  # [n_words, n_subtokens]

    def _cluster_ids(self, doc: Doc) -> torch.Tensor:
        """
        Args:
            doc: document information

        Returns:
            torch.Tensor of shape [n_word], containing cluster indices for
                each word. Non-coreferent words have cluster id of zero.
        """
        word2cluster = {word_i: i
                        for i, cluster in enumerate(doc["word_clusters"], start=1)
                        for word_i in cluster}

        return torch.tensor(
            [word2cluster.get(word_i, 0)
             for word_i in range(len(doc["cased_words"]))],
            device=self.device
        )
