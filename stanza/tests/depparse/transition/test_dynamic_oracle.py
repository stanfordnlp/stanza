import pytest

from collections import Counter

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

from stanza.models.depparse.transition import dynamic_oracle
from stanza.models.depparse.transition.transitions import ProjectiveRight, NonprojectiveRight, ProjectiveLeft, NonprojectiveLeft, Shift, Finalize

def test_fix_left_shift():
    # test the result of rearranging a ProjectiveLeft with a single incorrect node
    gold_sequence = [Shift(), Shift(), Shift(), ProjectiveRight("nsubj"), ProjectiveRight("nsubj"), Finalize()]
    result = dynamic_oracle.fix_left_instead_of_shift_right_head(gold_sequence, 2, Shift(), ProjectiveLeft("nsubj"))
    expected = [Shift(), Shift(), ProjectiveLeft("nsubj"), Shift(), ProjectiveRight("nsubj"), Finalize()]
    assert result == expected

    # test when a whole subtree was expected
    # in this case, the 3rd and 4th shift are combined with a dobj
    # after that, word 2 was supposed to be connected with nsubj
    # however, we accidentally connected word 2 to the left earlier instead
    gold_sequence = [Shift(), Shift(), Shift(), Shift(), ProjectiveLeft("dobj"), ProjectiveRight("nsubj"), ProjectiveRight("iobj"), Finalize()]
    result = dynamic_oracle.fix_left_instead_of_shift_right_head(gold_sequence, 2, Shift(), ProjectiveLeft("csubj"))
    expected = [Shift(), Shift(), ProjectiveLeft("csubj"), Shift(), Shift(), ProjectiveLeft("dobj"), ProjectiveRight("iobj"), Finalize()]
    assert result == expected

def test_fix_right_shift():
    gold_sequence = [Shift(), Shift(), Shift(), ProjectiveRight("nsubj"), ProjectiveRight("dobj"), Finalize()]
    result = dynamic_oracle.fix_right_instead_of_shift_right_head(gold_sequence, 2, Shift(), ProjectiveRight("csubj"))
    # if we attach the first node to the wrong head, we can still finish the parse
    # without making any further errors
    # the PR(nsubj) is attaching node 2 to node 3
    expected = [Shift(), Shift(), ProjectiveRight("csubj"), Shift(), ProjectiveRight("nsubj"), Finalize()]
    assert result == expected

def test_fix_shift_right():
    gold_sequence = [Shift(), Shift(), ProjectiveRight("nsubj"), Shift(), ProjectiveLeft("dobj"), Finalize()]
    result = dynamic_oracle.fix_shift_instead_of_right(gold_sequence, 2, ProjectiveRight("nsubj"), Shift())
    expected = [Shift(), Shift(), Shift(), ProjectiveLeft("dobj"), ProjectiveRight("nsubj"), Finalize()]
    assert result == expected


