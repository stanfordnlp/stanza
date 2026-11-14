"""
Test that RoughScorer._prune produces a deterministic antecedent layout

The AnaphoricityScorer's start_map is a linear layer over the k
antecedent positions, so the order of the pruned antecedents affects
singleton predictions.  See https://github.com/stanfordnlp/stanza/issues/1718
"""

from types import SimpleNamespace

import pytest
import torch

from stanza.models.coref.rough_scorer import RoughScorer

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def build_rough_scores(n_mentions, seed):
    """
    Random scores with the same -inf mask RoughScorer.forward applies
    """
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(n_mentions, n_mentions, generator=generator)
    pair_mask = torch.arange(n_mentions)
    pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
    pair_mask = torch.log((pair_mask > 0).to(torch.float))
    return scores + pair_mask

def reference_prune(rough_scores, k):
    """
    The layout of CUDA topk(sorted=False)

    The entries strictly greater than the kth value in ascending index
    order, followed by entries equal to the kth value in ascending
    index order, with the lowest indices chosen among the ties.
    """
    order = torch.sort(rough_scores, dim=1, descending=True, stable=True).indices
    kth_scores = rough_scores.gather(1, order[:, k-1:k])
    rows = []
    for row_scores, row_order, kth in zip(rough_scores, order, kth_scores):
        greater = [idx for idx in row_order[:k].tolist() if row_scores[idx] > kth]
        equal = (row_scores == kth).nonzero().flatten().tolist()
        rows.append(sorted(greater) + equal[:k - len(greater)])
    indices = torch.tensor(rows, dtype=torch.long)
    return rough_scores.gather(1, indices), indices

def build_scorer(k):
    config = SimpleNamespace(dropout_rate=0.0, rough_k=k)
    return RoughScorer(4, config)

@pytest.mark.parametrize("n_mentions, k", [(5, 10), (10, 10), (11, 10), (200, 50), (500, 50)])
def test_prune_matches_reference(n_mentions, k):
    scorer = build_scorer(k)
    for seed in range(5):
        rough_scores = build_rough_scores(n_mentions, seed)
        top_scores, indices, _ = scorer._prune(rough_scores)
        expected_scores, expected_indices = reference_prune(rough_scores, min(k, n_mentions))
        assert torch.equal(indices, expected_indices)
        assert torch.equal(top_scores, expected_scores)

def test_prune_rows_layout():
    """
    Rows with k real antecedents are the top k-1 in ascending order, then the lowest scoring antecedent
    """
    scorer = build_scorer(50)
    rough_scores = build_rough_scores(300, 0)
    top_scores, indices, _ = scorer._prune(rough_scores)
    assert torch.all(indices[:, 1:-1] > indices[:, :-2])
    assert torch.equal(top_scores[50:, -1], top_scores[50:].min(dim=1).values)

def test_prune_padding_rows():
    """
    The first k rows have fewer than k real antecedents and are padded with the lowest -inf indices
    """
    scorer = build_scorer(50)
    rough_scores = build_rough_scores(300, 0)
    top_scores, indices, _ = scorer._prune(rough_scores)
    for row in range(50):
        assert torch.equal(indices[row], torch.arange(50))
        assert torch.all(torch.isfinite(top_scores[row, :row]))
        assert torch.all(torch.isneginf(top_scores[row, row:]))
    for row in range(50, 300):
        assert torch.all(torch.isfinite(top_scores[row]))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n_mentions, k", [(5, 10), (11, 10), (200, 50), (2000, 50)])
def test_prune_matches_cuda_topk(n_mentions, k):
    """
    The existing models were trained on GPU, so _prune should reproduce raw CUDA topk(sorted=False)
    """
    scorer = build_scorer(k)
    for seed in range(5):
        rough_scores = build_rough_scores(n_mentions, seed).cuda()
        cuda_scores, cuda_indices = torch.topk(rough_scores, k=min(k, n_mentions), dim=1, sorted=False)
        top_scores, indices, _ = scorer._prune(rough_scores)
        assert torch.equal(indices, cuda_indices)
        assert torch.equal(top_scores, cuda_scores)
