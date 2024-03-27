""" Contains type aliases for coref module """

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


EPSILON = 1e-7
LARGE_VALUE = 1000  # used instead of inf due to bug #16762 in pytorch

Doc = Dict[str, Any]
Span = Tuple[int, int]


@dataclass
class CorefResult:
    coref_scores: torch.Tensor = None                  # [n_words, k + 1]
    coref_y: torch.Tensor = None                       # [n_words, k + 1]
    rough_y: torch.Tensor = None                       # [n_words, n_words]

    word_clusters: List[List[int]] = None
    span_clusters: List[List[Span]] = None

    rough_scores: torch.Tensor = None                  # [n_words, n_words]
    span_scores: torch.Tensor = None                   # [n_heads, n_words, 2]
    span_y: Tuple[torch.Tensor, torch.Tensor] = None   # [n_heads] x2
