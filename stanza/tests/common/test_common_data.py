import pytest
import stanza

from stanza.tests import *
from stanza.models.common.data import get_augment_ratio, starts_with_initial_mark, INITIAL_INVERTED_PUNCT_MARKS

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

class TestStartsWithInitialMark:

    def test_simple_question(self):
        assert starts_with_initial_mark(['¿', 'Cómo', 'estás', '?']) is True

    def test_simple_exclamation(self):
        assert starts_with_initial_mark(['¡', 'Qué', 'bien', '!']) is True

    def test_false_when_no_leading_mark(self):
        assert starts_with_initial_mark(['Cómo', 'estás', '?']) is False

    def test_false_for_single_word_sentence(self):
        assert starts_with_initial_mark(['¿']) is False

    def test_false_when_leading_mark_repeated(self):
        """Two of the SAME mark (e.g. two ¿) should be excluded."""
        assert starts_with_initial_mark(['¿', 'Cómo', '¿', 'estás', '?']) is False

    def test_false_when_different_marks_both_present(self):
        """
        A DIFFERENT mark appearing elsewhere must also disqualify the
        sentence, not just a repeat of the leading mark -- '¿Dijo
        "¡hola!"?' has one ¿ and one ¡, and neither mark is individually
        repeated, but there are still two candidate marks overall.
        """
        words = ['¿', 'Dijo', '"', '¡', 'hola', '!', '"', '?']
        assert starts_with_initial_mark(words) is False

    def test_true_when_only_the_leading_mark_is_present(self):
        """Sanity check: the mixed-mark case above is specifically about a SECOND mark, not just any punctuation."""
        words = ['¿', 'Dijo', '"', 'hola', '"', '?']
        assert starts_with_initial_mark(words) is True

    def test_custom_marks_argument(self):
        assert starts_with_initial_mark(['!', 'foo', 'bar'], marks=('!',)) is True
        assert starts_with_initial_mark(['!', 'foo', '!', 'bar'], marks=('!',)) is False

    def test_default_marks_tuple(self):
        assert INITIAL_INVERTED_PUNCT_MARKS == ('¿', '¡')
