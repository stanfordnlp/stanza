""" Describes PairwiseEncodes, that transforms pairwise features, such as
distance between the mentions, same/different speaker into feature embeddings
"""
from typing import List

import torch

from coref.config import Config
from coref.const import Doc


class PairwiseEncoder(torch.nn.Module):
    """ A Pytorch module to obtain feature embeddings for pairwise features

    Usage:
        encoder = PairwiseEncoder(config)
        pairwise_features = encoder(pair_indices, doc)
    """
    def __init__(self, config: Config):
        super().__init__()
        emb_size = config.embedding_size

        self.genre2int = {g: gi for gi, g in enumerate(["bc", "bn", "mz", "nw",
                                                        "pt", "tc", "wb"])}
        self.genre_emb = torch.nn.Embedding(len(self.genre2int), emb_size)

        # each position corresponds to a bucket:
        #   [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8),
        #    (8, 16), (16, 32), (32, 64), (64, float("inf"))]
        self.distance_emb = torch.nn.Embedding(9, emb_size)

        # two possibilities: same vs different speaker
        self.speaker_emb = torch.nn.Embedding(2, emb_size)

        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.shape = emb_size * 3  # genre, distance, speaker\

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.genre_emb.parameters()).device

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                top_indices: torch.Tensor,
                doc: Doc) -> torch.Tensor:
        word_ids = torch.arange(0, len(doc["cased_words"]), device=self.device)
        speaker_map = torch.tensor(self._speaker_map(doc), device=self.device)

        same_speaker = (speaker_map[top_indices] == speaker_map.unsqueeze(1))
        same_speaker = self.speaker_emb(same_speaker.to(torch.long))

        # bucketing the distance (see __init__())
        distance = (word_ids.unsqueeze(1) - word_ids[top_indices]
                    ).clamp_min_(min=1)
        log_distance = distance.to(torch.float).log2().floor_()
        log_distance = log_distance.clamp_max_(max=6).to(torch.long)
        distance = torch.where(distance < 5, distance - 1, log_distance + 2)
        distance = self.distance_emb(distance)

        genre = torch.tensor(self.genre2int[doc["document_id"][:2]],
                             device=self.device).expand_as(top_indices)
        genre = self.genre_emb(genre)

        return self.dropout(torch.cat((same_speaker, distance, genre), dim=2))

    @staticmethod
    def _speaker_map(doc: Doc) -> List[int]:
        """
        Returns a tensor where i-th element is the speaker id of i-th word.
        """
        # speaker string -> speaker id
        str2int = {s: i for i, s in enumerate(set(doc["speaker"]))}

        # word id -> speaker id
        return [str2int[s] for s in doc["speaker"]]
