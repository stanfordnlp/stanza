from enum import Enum

from stanza.models.constituency.dynamic_oracle import advance_past_constituents, find_in_order_constituent_end, find_previous_open, DynamicOracle
from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent, CompoundUnary, Finalize

def fix_missing_unary_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    """
    A CompoundUnary transition was missed after a Shift, but the sequence was continued correctly otherwise
    """
    if not isinstance(gold_transition, CompoundUnary):
        return None

    if pred_transition != gold_sequence[gold_index + 1]:
        return None
    if isinstance(pred_transition, Finalize):
        # this can happen if the entire tree is a single word
        # but it can't be fixed if it means the parser missed the ROOT transition
        return None

    return gold_sequence[:gold_index] + gold_sequence[gold_index+1:]

def fix_wrong_unary_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    if not isinstance(gold_transition, CompoundUnary):
        return None

    if not isinstance(pred_transition, CompoundUnary):
        return None

    assert gold_transition != pred_transition

    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index+1:]

def fix_spurious_unary_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    if isinstance(gold_transition, CompoundUnary):
        return None

    if not isinstance(pred_transition, CompoundUnary):
        return None

    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:]

def fix_open_shift_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    """
    Fix a missed Open constituent where we predicted a Shift and the next transition was a Shift

    In fact, the subsequent transition MUST be a Shift with this transition scheme
    """
    if not isinstance(gold_transition, OpenConstituent):
        return None

    if not isinstance(pred_transition, Shift):
        return None

    #if not isinstance(gold_sequence[gold_index+1], Shift):
    #    return None
    assert isinstance(gold_sequence[gold_index+1], Shift)

    # close_index represents the Close for the missing Open
    close_index = advance_past_constituents(gold_sequence, gold_index+1)
    assert close_index is not None
    return gold_sequence[:gold_index] + gold_sequence[gold_index+1:close_index] + gold_sequence[close_index+1:]

def fix_open_open_two_subtrees_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    if gold_transition == pred_transition:
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    block_end = find_in_order_constituent_end(gold_sequence, gold_index+1)
    if isinstance(gold_sequence[block_end], Shift):
        # this is a multiple subtrees version of this error
        # we are only skipping the two subtrees errors for now
        return None

    # no fix is possible, so we just return here
    return RepairType.OPEN_OPEN_TWO_SUBTREES_ERROR, None

def fix_open_open_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, exactly_three):
    if gold_transition == pred_transition:
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    block_end = find_in_order_constituent_end(gold_sequence, gold_index+1)
    if not isinstance(gold_sequence[block_end], Shift):
        # this is a multiple subtrees version of this error
        # we are only skipping the two subtrees errors for now
        return None

    next_block_end = find_in_order_constituent_end(gold_sequence, block_end+1)
    if exactly_three and isinstance(gold_sequence[next_block_end], Shift):
        # for exactly three subtrees,
        # we can put back the missing open transition
        # and now we have no recall error, only precision error
        # for more than three, we separate that out as an ambiguous choice
        return None
    elif not exactly_three and isinstance(gold_sequence[next_block_end], CloseConstituent):
        # this is ambiguous, but we can still try this fix
        return None

    # at this point, we build a new sequence with the origin constituent inserted
    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index+1:block_end] + [CloseConstituent(), gold_transition] + gold_sequence[block_end:]


def fix_open_open_three_subtrees_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    return fix_open_open_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, exactly_three=True)

def fix_open_open_many_subtrees_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    return fix_open_open_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, exactly_three=False)

def fix_open_close_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    """
    Find the closed bracket, reopen it

    The Open we just missed must be forgotten - it cannot be reopened
    """
    if not isinstance(gold_transition, OpenConstituent):
        return None

    if not isinstance(pred_transition, CloseConstituent):
        return None

    # find the appropriate Open so we can reopen it
    open_idx = find_previous_open(gold_sequence, gold_index)
    # actually, if the Close is legal, this can't happen
    # but it might happen in a unit test which doesn't check legality
    if open_idx is None:
        return None

    # also, since we are punting on the missed Open, we need to skip
    # the Close which would have closed it
    close_idx = advance_past_constituents(gold_sequence, gold_index+1)

    return gold_sequence[:gold_index] + [pred_transition, gold_sequence[open_idx]] + gold_sequence[gold_index+1:close_idx] + gold_sequence[close_idx+1:]

def fix_shift_close_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    """
    Find the closed bracket, reopen it
    """
    if not isinstance(gold_transition, Shift):
        return None

    if not isinstance(pred_transition, CloseConstituent):
        return None

    # don't do this at the start or immediately after opening
    if gold_index == 0 or isinstance(gold_sequence[gold_index - 1], OpenConstituent):
        return None

    open_idx = find_previous_open(gold_sequence, gold_index)
    assert open_idx is not None

    return gold_sequence[:gold_index] + [pred_transition, gold_sequence[open_idx]] + gold_sequence[gold_index:]

def fix_shift_open_unambiguous_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    if not isinstance(gold_transition, Shift):
        return None

    if not isinstance(pred_transition, OpenConstituent):
        return None

    bracket_end = find_in_order_constituent_end(gold_sequence, gold_index)
    assert bracket_end is not None
    if isinstance(gold_sequence[bracket_end], Shift):
        # this is an ambiguous error
        # multiple possible places to end the wrong constituent
        return None
    assert isinstance(gold_sequence[bracket_end], CloseConstituent)

    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:bracket_end] + [CloseConstituent()] + gold_sequence[bracket_end:]

def fix_close_shift_unambiguous_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, model, state):
    if not isinstance(gold_transition, CloseConstituent):
        return None

    if not isinstance(pred_transition, Shift):
        return None
    if not isinstance(gold_sequence[gold_index+1], Shift):
        return None

    bracket_end = find_in_order_constituent_end(gold_sequence, gold_index+1)
    assert bracket_end is not None
    if isinstance(gold_sequence[bracket_end], Shift):
        # this is an ambiguous error
        # multiple possible places to end the wrong constituent
        return None
    assert isinstance(gold_sequence[bracket_end], CloseConstituent)

    return gold_sequence[:gold_index] + gold_sequence[gold_index+1:bracket_end] + [CloseConstituent()] + gold_sequence[bracket_end:]

class RepairType(Enum):
    """
    Keep track of which repair is used, if any, on an incorrect transition

    Effects of different repair types:
      no oracle:                0.9251  0.9226
     +missing_unary:            0.9246  0.9214
     +wrong_unary:              0.9236  0.9213
     +spurious_unary:           0.9247  0.9229
     +open_shift_error:         0.9258  0.9226
     +open_open_two_subtrees:   0.9256  0.9215    # nothing changes with this one...
     +open_open_three_subtrees: 0.9256  0.9226
     +open_open_many_subtrees:  0.9257  0.9234
     +shift_close:              0.9267  0.9250
     +shift_open:               0.9273  0.9247
     +close_shift:              0.9266  0.9229
     +open_close:               0.9267  0.9256
    """
    def __new__(cls, fn, correct=False, debug=False):
        """
        Enumerate values as normal, but also keep a pointer to a function which repairs that kind of error
        """
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value + 1
        obj.fn = fn
        obj.correct = correct
        obj.debug = debug
        return obj

    @property
    def is_correct(self):
        return self.correct

    # The correct sequence went Shift - Unary - Stuff
    # but the CompoundUnary was missed and Stuff predicted
    # so now we just proceed as if nothing happened
    # note that CompoundUnary happens immediately after a Shift
    # complicated nodes are created with single Open transitions
    MISSING_UNARY_ERROR                    = (fix_missing_unary_error,)

    # Predicted a wrong CompoundUnary.  No way to fix this, so just keep going
    WRONG_UNARY_ERROR                      = (fix_wrong_unary_error,)

    # The correct sequence went Shift - Stuff
    # but instead we predicted a CompoundUnary
    # again, we just keep going
    SPURIOUS_UNARY_ERROR                   = (fix_spurious_unary_error,)

    # Were supposed to open a new constituent,
    # but instead shifted an item onto the stack
    #
    # The missed Open cannot be recovered
    #
    # One could ask, is it possible to open a bigger constituent later,
    # but if the constituent patterns go
    #   X (good open) Y (missed open) Z
    # when we eventually close Y and Z, because of the missed Open,
    # it is guaranteed to capture X as well
    # since it will grab constituents until one left of the previous Open before Y
    #
    # Therefore, in this case, we must simply forget about this Open (recall error)
    OPEN_SHIFT_ERROR                       = (fix_open_shift_error,)

    # With this transition scheme, it is not possible to fix the following pattern:
    #   T1 O_x T2 C -> T1 O_y T2 C
    # seeing as how there are no unary transitions
    # so whatever precision & recall errors are caused by substituting O_x -> O_y
    # (which could include multiple transitions)
    # those errors are unfixable in any way
    OPEN_OPEN_TWO_SUBTREES_ERROR           = (fix_open_open_two_subtrees_error,)

    # With this transition scheme, a three subtree branch with a wrong Open
    # has a non-ambiguous fix
    #   T1 O_x T2 T3 C -> T1 O_y T2 T3 C
    # this can become
    #   T1 O_y T2 C O_x T3 C
    # now there are precision errors from the incorrectly added transition(s),
    # but the correctly replaced transitions are unambiguous
    OPEN_OPEN_THREE_SUBTREES_ERROR         = (fix_open_open_three_subtrees_error,)

    # We were supposed to shift a new item onto the stack,
    # but instead we closed the previous constituent
    # This causes a precision error, but we can avoid the recall error
    # by immediately reopening the closed constituent.
    SHIFT_CLOSE_ERROR                      = (fix_shift_close_error,)

    # We opened a new constituent instead of shifting
    # In the event that the next constituent ends with a close,
    # rather than building another new constituent,
    # then there is no ambiguity
    SHIFT_OPEN_UNAMBIGUOUS_ERROR           = (fix_shift_open_unambiguous_error,)

    # Suppose we were supposed to Close, then Shift
    # but instead we just did a Shift
    # Similar to shift_open_unambiguous, we now have an opened
    # constituent which shouldn't be there
    # We can scroll past the next constituent created to see
    # if the outer constituents close at that point
    # If so, we can close this constituent as well in an unambiguous manner
    # TODO: analyze the case where we were supposed to Close, Open
    # but instead did a Shift
    CLOSE_SHIFT_UNAMBIGUOUS_ERROR          = (fix_close_shift_unambiguous_error,)

    # Supposed to open a new constituent,
    # instead closed an existing constituent
    #
    #  X (good open) Y (open -> close) Z
    #
    # the constituent that should contain Y, Z is unfortunately lost
    # since now the stack has
    #
    #  XY ...
    #
    # furthermore, there is now a precision error for the extra XY
    # constituent that should not exist
    # however, what we can do to minimize further errors is
    # to at least reopen the label between X and Y
    OPEN_CLOSE_ERROR                       = (fix_open_close_error,)

    # this is ambiguous, but we can still try the same fix as three_subtrees (see above)
    OPEN_OPEN_MANY_SUBTREES_ERROR          = (fix_open_open_many_subtrees_error,)

    CORRECT                                = (None, True)

    UNKNOWN                                = None


class InOrderCompoundOracle(DynamicOracle):
    def __init__(self, root_labels, oracle_level, additional_oracle_levels, deactivated_oracle_levels):
        super().__init__(root_labels, oracle_level, RepairType, additional_oracle_levels, deactivated_oracle_levels)
