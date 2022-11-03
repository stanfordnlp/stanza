import pytest

import torch

from stanza import Pipeline
from stanza.models.constituency.positional_encoding import SinusoidalEncoding, AddSinusoidalEncoding

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


def test_positional_encoding():
    encoding = SinusoidalEncoding(model_dim=10, max_len=6)
    foo = encoding(torch.tensor([5]))
    assert foo.shape == (1, 10)
    # TODO: check the values

def test_resize():
    encoding = SinusoidalEncoding(model_dim=10, max_len=3)
    foo = encoding(torch.tensor([5]))
    assert foo.shape == (1, 10)


def test_arange():
    encoding = SinusoidalEncoding(model_dim=10, max_len=2)
    foo = encoding(torch.arange(4))
    assert foo.shape == (4, 10)
    assert encoding.max_len() == 4

def test_add():
    encoding = AddSinusoidalEncoding(d_model=10, max_len=4)
    x = torch.zeros(1, 4, 10)
    y = encoding(x)

    r = torch.randn(1, 4, 10)
    r2 = encoding(r)

    assert torch.allclose(r2 - r, y, atol=1e-07)

    r = torch.randn(2, 4, 10)
    r2 = encoding(r)

    assert torch.allclose(r2[0] - r[0], y, atol=1e-07)
    assert torch.allclose(r2[1] - r[1], y, atol=1e-07)
