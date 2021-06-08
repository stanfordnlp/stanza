import pytest
import stanza

from stanza.tests import *
from stanza.models.common.data import get_augment_ratio, augment_punct

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_augment_ratio():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    should_augment = lambda x: x >= 3
    can_augment = lambda x: x >= 4
    # check that zero is returned if no augmentation is needed
    # which will be the case since 2 are already satisfactory
    assert get_augment_ratio(data, should_augment, can_augment, desired_ratio=0.1) == 0.0

    # this should throw an error
    with pytest.raises(AssertionError):
        get_augment_ratio(data, can_augment, should_augment)

    # with a desired ratio of 0.4,
    # there are already 2 that don't need augmenting
    # and 7 that are eligible to be augmented
    # so 2/7 will need to be augmented
    assert get_augment_ratio(data, should_augment, can_augment, desired_ratio=0.4) == pytest.approx(2/7)

def test_augment_punct():
    data = [["Simple", "test", "."]]
    should_augment = lambda x: x[-1] == "."
    can_augment = should_augment
    new_data = augment_punct(data, 1.0, should_augment, can_augment)
    assert new_data == [["Simple", "test"]]
