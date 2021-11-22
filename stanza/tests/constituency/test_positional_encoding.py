import pytest

import torch

from stanza import Pipeline
from stanza.models.constituency.positional_encoding import SinusoidalEncoding

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
