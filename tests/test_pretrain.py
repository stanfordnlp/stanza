import pytest
import numpy as np

from stanza.models.common import pretrain
from tests import *

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def check_pretrain(pt):
    # 4 base vectors, plus the 3 vectors actually present in the file
    assert len(pt.vocab) == 7
    assert 'unban' in pt.vocab
    assert 'mox' in pt.vocab
    assert 'opal' in pt.vocab

    expected = np.array([[ 0.,  0.,  0.,  0.,],
                         [ 0.,  0.,  0.,  0.,],
                         [ 0.,  0.,  0.,  0.,],
                         [ 0.,  0.,  0.,  0.,],
                         [ 1.,  2.,  3.,  4.,],
                         [ 5.,  6.,  7.,  8.,],
                         [ 9., 10., 11., 12.,]])
    np.testing.assert_allclose(pt.emb, expected)    

def test_text_pretrain():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.txt', save_to_file=False)
    check_pretrain(pt)

def test_xz_pretrain():
    pt = pretrain.Pretrain(vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz', save_to_file=False)
    check_pretrain(pt)

