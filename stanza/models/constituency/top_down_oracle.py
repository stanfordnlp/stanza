from enum import Enum, auto

from stanza.models.constituency.dynamic_oracle import advance_past_constituents, DynamicOracle
from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent

def find_constituent_end(gold_sequence, cur_index):
    """
    Find the Close which ends the next constituent opened at or after cur_index
    """
    count = 0
    while cur_index < len(gold_sequence):
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
            if count == 0:
                return cur_index
        cur_index += 1
    raise AssertionError("Open constituent not closed starting from index %d in sequence %s" % (cur_index, gold_sequence))


def fix_shift_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Predicted a close when we should have shifted

    The fix here is to remove the corresponding close from later in
    the transition sequence.  The rest of the tree building is the same,
    including doing the missing Shift immediately after

    Anything else would make the situation of one precision, one
    recall error worse
    """
    if not isinstance(pred_transition, CloseConstituent):
        return None

    if not isinstance(gold_transition, Shift):
        return None

    close_index = advance_past_constituents(gold_sequence, gold_index)
    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:close_index] + gold_sequence[close_index+1:]

def fix_open_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Predicted a close when we should have opened a constituent

    In this case, the previous constituent is now a precision and
    recall error, BUT we can salvage the constituent we were about to
    open by proceeding as if everything else is still the same.

    The next thing the model should do is open the transition it forgot about
    """
    if not isinstance(pred_transition, CloseConstituent):
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None

    close_index = advance_past_constituents(gold_sequence, gold_index)
    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:close_index] + gold_sequence[close_index+1:]

def fix_one_open_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Predicted a shift when we should have opened a constituent

    This causes a single recall error if we just pretend that
    constituent didn't exist

    Keep the shift where it was, remove the next shift
    Also, scroll ahead, find the corresponding close, cut it out

    For the corresponding multiple opens, shift error, see fix_multiple_open_shift
    """
    if not isinstance(pred_transition, Shift):
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None

    if not isinstance(gold_sequence[gold_index + 1], Shift):
        return None

    shift_index = gold_index + 1
    close_index = advance_past_constituents(gold_sequence, gold_index + 1)
    if close_index is None:
        return None
    # gold_index is the skipped open constituent
    # close_index was the corresponding close
    # shift_index is the shift to remove
    updated_sequence = gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index+1:shift_index] + gold_sequence[shift_index+1:close_index] + gold_sequence[close_index+1:]
    #print("Input sequence: %s\nIndex %d\nGold %s Pred %s\nUpdated sequence %s" % (gold_sequence, gold_index, gold_transition, pred_transition, updated_sequence))
    return updated_sequence

def fix_multiple_open_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Predicted a shift when we should have opened multiple constituents instead

    This causes a single recall error per constituent if we just
    pretend those constituents don't exist

    For each open constituent, we find the corresponding close,
    then remove both the open & close
    """
    if not isinstance(pred_transition, Shift):
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None

    shift_index = gold_index
    while shift_index < len(gold_sequence) and isinstance(gold_sequence[shift_index], OpenConstituent):
        shift_index += 1
    if shift_index >= len(gold_sequence):
        raise AssertionError("Found a sequence of OpenConstituent at the end of a TOP_DOWN sequence!")
    if not isinstance(gold_sequence[shift_index], Shift):
        raise AssertionError("Expected to find a Shift after a sequence of OpenConstituent.  There should not be a %s" % gold_sequence[shift_index])

    #print("Input sequence: %s\nIndex %d\nGold %s Pred %s" % (gold_sequence, gold_index, gold_transition, pred_transition))
    updated_sequence = gold_sequence
    while shift_index > gold_index:
        close_index = advance_past_constituents(updated_sequence, shift_index)
        if close_index is None:
            raise AssertionError("Did not find a corresponding Close for this Open")
        # cut out the corresponding open and close
        updated_sequence = updated_sequence[:shift_index-1] + updated_sequence[shift_index:close_index] + updated_sequence[close_index+1:]
        shift_index -= 1
        #print("  %s" % updated_sequence)

    #print("Final updated sequence: %s" % updated_sequence)
    return updated_sequence

def fix_nested_open_constituent(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    We were supposed to predict Open(X), then Open(Y), but predicted Open(Y) instead

    We treat this as a single recall error.

    We could even go crazy and turn it into a Unary,
    such as Open(Y), Open(X), Open(Y)...
    presumably that would be very confusing to the parser
    not to mention ambiguous as to where to close the new constituent
    """
    if not isinstance(pred_transition, OpenConstituent):
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None

    assert len(gold_sequence) > gold_index + 1

    if not isinstance(gold_sequence[gold_index+1], OpenConstituent):
        return None

    # This replacement works if we skipped exactly one level
    if gold_sequence[gold_index+1].label != pred_transition.label:
        return None

    close_index = advance_past_constituents(gold_sequence, gold_index+1)
    assert close_index is not None
    updated_sequence = gold_sequence[:gold_index] + gold_sequence[gold_index+1:close_index] + gold_sequence[close_index+1:]
    return updated_sequence

def fix_shift_open_immediate_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    We were supposed to Shift, but instead we Opened

    The biggest problem with this type of error is that the Close of
    the Open is ambiguous.  We could put it immediately before the
    next Close, immediately after the Shift, or anywhere in between.

    One unambiguous case would be if the proper sequence was Shift - Close.
    Then it is unambiguous that the only possible repair is Open - Shift - Close - Close.
    """
    if not isinstance(pred_transition, OpenConstituent):
        return None

    if not isinstance(gold_transition, Shift):
        return None

    assert len(gold_sequence) > gold_index + 1
    if not isinstance(gold_sequence[gold_index+1], CloseConstituent):
        # this is the ambiguous case
        return None

    return gold_sequence[:gold_index] + [pred_transition, gold_transition, CloseConstituent()] + gold_sequence[gold_index+1:]

def fix_close_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, count_opens=False):
    """
    We were supposed to Close, but instead did a Shift

    In most cases, this will be ambiguous.  There is now a constituent
    which has been missed, no matter what we do, and we are on the
    hook for eventually closing this constituent, creating a precision
    error as well.  The ambiguity arises because there will be
    multiple places where the Close could occur if there are more
    constituents created between now and when the outer constituent is
    Closed.

    The non-ambiguous case is if the proper sequence was
      Close - Shift - Close
    similar cases are also non-ambiguous, such as
      Close - Close - Shift - Close
    for that matter, so is the following, although the Opens will be lost
      Close - Open - Shift - Close - Close

    count_opens is an option to make it easy to count with or without
      Open as different oracle fixes
    """
    if not isinstance(pred_transition, Shift):
        return None

    if not isinstance(gold_transition, CloseConstituent):
        return None

    num_closes = 0
    while isinstance(gold_sequence[gold_index + num_closes], CloseConstituent):
        num_closes += 1

    # We may allow unary transitions here
    # the opens will be lost in the repaired sequence
    num_opens = 0
    if count_opens:
        while isinstance(gold_sequence[gold_index + num_closes + num_opens], OpenConstituent):
            num_opens += 1

    if not isinstance(gold_sequence[gold_index + num_closes + num_opens], Shift):
        return None

    if not isinstance(gold_sequence[gold_index + num_closes + num_opens + 1], CloseConstituent):
        return None
    for idx in range(num_opens):
        if not isinstance(gold_sequence[gold_index + num_closes + num_opens + idx + 1], CloseConstituent):
            return None

    # Now we know it is Close x num_closes, Shift, Close
    # Since we have erroneously predicted a Shift now, the best we can
    # do is to follow that, then add num_closes Closes
    updated_sequence = gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:gold_index+num_closes] + gold_sequence[gold_index+num_closes+num_opens*2+1:]
    return updated_sequence

def fix_close_shift_with_opens(*args, **kwargs):
    return fix_close_shift(*args, **kwargs, count_opens=True)

def fix_close_open_one_con(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    We were supposed to Close, but instead did an Open

    In general this is ambiguous (like close/shift), as we need to know when to close the incorrect constituent

    A case that is not ambiguous is when exactly one constituent was
    supposed to come after the Close and it matches the Open we just
    created.  In that case, we treat that constituent as if it were
    part of the non-Closed constituent.  For example,
    "ate (NP spaghetti) (PP with a fork)" ->
    "ate (NP spaghetti (PP with a fork))"
    (delicious)
    """
    if not isinstance(pred_transition, OpenConstituent):
        return None

    if not isinstance(gold_transition, CloseConstituent):
        return None

    if gold_sequence[gold_index+1] != pred_transition:
        return None

    close_index = find_constituent_end(gold_sequence, gold_index+1)
    if not isinstance(gold_sequence[close_index+1], CloseConstituent):
        return None

    # at this point, we know we can put the Close at the end of the
    # Open which was accidentally added
    updated_sequence = gold_sequence[:gold_index] + gold_sequence[gold_index+1:close_index+2] + [gold_transition] + gold_sequence[close_index+2:]
    return updated_sequence

class RepairType(Enum):
    """
    Keep track of which repair is used, if any, on an incorrect transition

    A test of the top-down oracle with no charlm or transformer
      (eg, word vectors only) on EN PTB3 goes as follows.
      3x training rounds, best training parameters as of Jan. 2024
    unambiguous transitions only:
        oracle scheme         dev        test
      no oracle              0.9230     0.9194
       +shift/close          0.9224     0.9180
       +open/close           0.9225     0.9193
       +open/shift (one)     0.9245     0.9207
       +open/shift (mult)    0.9243     0.9211
       +open/open nested     0.9258     0.9213
       +shift/open           0.9266     0.9229
       +close/shift (only)   0.9270     0.9230
       +close/shift w/ opens 0.9262     0.9221
       +close/open one con   0.9273     0.9230
    """
    def __new__(cls, fn, correct=False):
        """
        Enumerate values as normal, but also keep a pointer to a function which repairs that kind of error
        """
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value + 1
        obj.fn = fn
        obj.correct = correct
        return obj

    def is_correct(self):
        return self.correct

    # The parser chose to close a bracket instead of shift something
    # into the bracket
    # This causes both a precision and a recall error as there is now
    # an incorrect bracket and a missing correct bracket
    # Any bracket creation here would cause more wrong brackets, though
    SHIFT_CLOSE_ERROR            = (fix_shift_close,)

    OPEN_CLOSE_ERROR             = (fix_open_close,)

    # open followed by shift was instead predicted to be shift
    ONE_OPEN_SHIFT_ERROR         = (fix_one_open_shift,)

    # open followed by shift was instead predicted to be shift
    MULTIPLE_OPEN_SHIFT_ERROR    = (fix_multiple_open_shift,)

    # should have done Open(X), Open(Y)
    # instead just did Open(Y)
    NESTED_OPEN_OPEN_ERROR       = (fix_nested_open_constituent,)

    SHIFT_OPEN_ERROR             = (fix_shift_open_immediate_close,)

    CLOSE_SHIFT_ERROR            = (fix_close_shift,)

    CLOSE_SHIFT_WITH_OPENS_ERROR = (fix_close_shift_with_opens,)

    CLOSE_OPEN_ONE_CON_ERROR     = (fix_close_open_one_con,)

    CORRECT                      = (None, True)

    UNKNOWN                      = None

class TopDownOracle(DynamicOracle):
    def __init__(self, root_labels, oracle_level):
        super().__init__(root_labels, oracle_level, RepairType)
