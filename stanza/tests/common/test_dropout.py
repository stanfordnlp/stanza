import pytest

import torch

import stanza
from stanza.models.common.dropout import WordDropout

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_word_dropout():
    """
    Test that word_dropout is randomly dropping out the entire final dimension of a tensor

    Doing 600 small rows should be super fast, but it leaves us with
    something like a 1 in 10^180 chance of the test failing.  Not very
    common, in other words
    """
    wd = WordDropout(0.5)
    batch = torch.randn(600, 4)
    dropped = wd(batch)
    # the one time any of this happens, it's going to be really confusing
    assert not torch.allclose(batch, dropped)
    num_zeros = 0
    for i in range(batch.shape[0]):
        assert torch.allclose(dropped[i], batch[i]) or torch.sum(dropped[i]) == 0.0
        if torch.sum(dropped[i]) == 0.0:
            num_zeros += 1
    assert num_zeros > 0 and num_zeros < batch.shape[0]
