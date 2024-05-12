import pytest

import torch

from stanza.models.common.relative_attn import RelativeAttention

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


def test_attn():
    foo = RelativeAttention(d_model=100, num_heads=2, window=8, dropout=0.0)
    bar = torch.randn(10, 13, 100)
    result = foo(bar)
    assert result.shape == bar.shape
    value = foo.value(bar)
    if not torch.allclose(result[:, -1, :], value[:, -1, :], atol=1e-06):
        raise ValueError(result[:, -1, :] - value[:, -1, :])
    assert torch.allclose(result[:, -1, :], value[:, -1, :], atol=1e-06)
    assert not torch.allclose(result[:, 0, :], value[:, 0, :])


def test_shorter_sequence():
    # originally this was failing because the batch was smaller than the window
    foo = RelativeAttention(d_model=20, num_heads=2, window=5, dropout=0.0)
    bar = torch.randn(10, 3, 20)
    result = foo(bar)
    assert result.shape == bar.shape

    value = foo.value(bar)
    if not torch.allclose(result[:, -1, :], value[:, -1, :], atol=1e-06):
        raise ValueError(result[:, -1, :] - value[:, -1, :])
    assert torch.allclose(result[:, -1, :], value[:, -1, :], atol=1e-06)
    assert not torch.allclose(result[:, 0, :], value[:, 0, :])

def test_reverse():
    foo = RelativeAttention(d_model=100, num_heads=2, window=8, reverse=True, dropout=0.0)
    bar = torch.randn(10, 13, 100)
    result = foo(bar)
    assert result.shape == bar.shape
    value = foo.value(bar)
    if not torch.allclose(result[:, 0, :], value[:, 0, :], atol=1e-06):
        raise ValueError(result[:, 0, :] - value[:, 0, :])
    assert torch.allclose(result[:, 0, :], value[:, 0, :], atol=1e-06)
    assert not torch.allclose(result[:, -1, :], value[:, -1, :])


