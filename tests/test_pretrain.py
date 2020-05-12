import os
import tempfile

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

def test_resave_pretrain():
    """
    Test saving a pretrain and then loading from the existing file
    """
    test_pt_file = tempfile.NamedTemporaryFile(dir=f'{TEST_WORKING_DIR}/out', suffix=".pt", delete=False)
    try:
        test_pt_file.close()
        # note that this tests the ability to save a pretrain and the
        # ability to fall back when the existing pretrain isn't working
        pt = pretrain.Pretrain(filename=test_pt_file.name,
                               vec_filename=f'{TEST_WORKING_DIR}/in/tiny_emb.xz')
        check_pretrain(pt)

        pt2 = pretrain.Pretrain(filename=test_pt_file.name,
                               vec_filename=f'unban_mox_opal')
        check_pretrain(pt2)
    finally:
        os.unlink(test_pt_file.name)
