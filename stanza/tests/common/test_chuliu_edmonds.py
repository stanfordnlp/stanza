"""
Test some use cases of the chuliu_edmonds algorithm

(currently just the tarjan implementation)
"""

import numpy as np
import pytest

from stanza.models.common.chuliu_edmonds import tarjan

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_tarjan_basic():
    simple = np.array([0, 4, 4, 4, 0])
    result = tarjan(simple)
    assert result == []

    simple = np.array([0, 2, 0, 4, 2, 2])
    result = tarjan(simple)
    assert result == []

def test_tarjan_cycle():
    cycle_graph = np.array([0, 3, 1, 2])
    result = tarjan(cycle_graph)
    expected = np.array([False,  True,  True,  True])
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], expected)

    cycle_graph = np.array([0, 3, 1, 2, 5, 6, 4])
    result = tarjan(cycle_graph)
    assert len(result) == 2
    expected = [np.array([False,  True,  True,  True, False, False, False]),
                np.array([False, False, False, False,  True,  True,  True])]
    for r, e in zip(result, expected):
        np.testing.assert_array_equal(r, e)
