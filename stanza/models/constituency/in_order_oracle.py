from enum import Enum

from stanza.models.constituency.dynamic_oracle import advance_past_constituents, find_in_order_constituent_end, find_previous_open, DynamicOracle
from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent

def fix_wrong_open_root_error(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    If there is an open/open error specifically at the ROOT, close the wrong open and try again
    """
    if gold_transition == pred_transition:
        return None

    if isinstance(gold_transition, OpenConstituent) and isinstance(pred_transition, OpenConstituent) and gold_transition.top_label in root_labels:
        return gold_sequence[:gold_index] + [pred_transition, CloseConstituent()] + gold_sequence[gold_index:]

    return None

def fix_wrong_open_unary_chain(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix a wrong open/open in a unary chain by removing the skipped unary transitions

    Only applies is the wrong pred transition is a transition found higher up in the unary chain
    """
    # useful to have this check here in case the call is made independently in a unit test
    if gold_transition == pred_transition:
        return None

    if isinstance(gold_transition, OpenConstituent) and isinstance(pred_transition, OpenConstituent):
        cur_index = gold_index + 1  # This is now a Close if we are in this particular context
        while cur_index + 1 < len(gold_sequence) and isinstance(gold_sequence[cur_index], CloseConstituent) and isinstance(gold_sequence[cur_index+1], OpenConstituent):
            cur_index = cur_index + 1  # advance to the next Open
            if gold_sequence[cur_index] == pred_transition:
                return gold_sequence[:gold_index] + gold_sequence[cur_index:]
            cur_index = cur_index + 1  # advance to the next Close

    return None

def fix_wrong_open_subtrees(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, more_than_two):
    if gold_transition == pred_transition:
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    if isinstance(gold_sequence[gold_index+1], CloseConstituent):
        # if Close, the gold was a unary
        return None
    assert not isinstance(gold_sequence[gold_index+1], OpenConstituent)
    assert isinstance(gold_sequence[gold_index+1], Shift)

    block_end = find_in_order_constituent_end(gold_sequence, gold_index+1)
    assert block_end is not None

    if more_than_two and isinstance(gold_sequence[block_end], CloseConstituent):
        return None
    if not more_than_two and isinstance(gold_sequence[block_end], Shift):
        return None

    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index+1:block_end] + [CloseConstituent(), gold_transition] + gold_sequence[block_end:]

def fix_wrong_open_two_subtrees(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    return fix_wrong_open_subtrees(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, more_than_two=False)

def fix_wrong_open_multiple_subtrees(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    return fix_wrong_open_subtrees(gold_transition, pred_transition, gold_sequence, gold_index, root_labels, more_than_two=True)

def advance_past_unaries(gold_sequence, cur_index):
    while cur_index + 2 < len(gold_sequence) and isinstance(gold_sequence[cur_index], OpenConstituent) and isinstance(gold_sequence[cur_index+1], CloseConstituent):
        cur_index += 2
    return cur_index

def fix_wrong_open_stuff_unary(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix a wrong open/open when there is an intervening constituent and then the guessed NT

    This happens when the correct pattern is
      stuff_1 NT_X stuff_2 close NT_Y ...
    and instead of guessing the gold transition NT_X,
    the prediction was NT_Y
    """
    if gold_transition == pred_transition:
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None
    # TODO: Here we could advance past unary transitions while
    # watching for hitting pred_transition.  However, that is an open
    # question... is it better to try to keep such an Open as part of
    # the sequence, or is it better to skip them and attach the inner
    # nodes to the upper level
    stuff_start = gold_index + 1
    if not isinstance(gold_sequence[stuff_start], Shift):
        return None
    stuff_end = advance_past_constituents(gold_sequence, stuff_start)
    if stuff_end is None:
        return None
    # at this point, stuff_end points to the Close which occurred after stuff_2
    # also, stuff_start points to the first transition which makes stuff_2, the Shift
    cur_index = stuff_end + 1
    while isinstance(gold_sequence[cur_index], OpenConstituent):
        if gold_sequence[cur_index] == pred_transition:
            return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[stuff_start:stuff_end] + gold_sequence[cur_index+1:]
        # this was an OpenConstituent, but not the OpenConstituent we guessed
        # maybe there's a unary transition which lets us try again
        if cur_index + 2 < len(gold_sequence) and isinstance(gold_sequence[cur_index + 1], CloseConstituent):
            cur_index = cur_index + 2
        else:
            break

    # oh well, none of this worked
    return None

def fix_wrong_open_general(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix a general wrong open/open transition by accepting the open and continuing

    A couple other open/open patterns have already been carved out

    TODO: negative checks for the previous patterns, in case we turn those off
    """
    if gold_transition == pred_transition:
        return None

    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None
    # If the top is a ROOT, then replacing it with a non-ROOT creates an illegal
    # transition sequence.  The ROOT case was already handled elsewhere anyway
    if gold_transition.top_label in root_labels:
        return None

    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index+1:]

def fix_missed_unary(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix a missed unary which is followed by an otherwise correct transition

    (also handles multiple missed unary transitions)
    """
    if gold_transition == pred_transition:
        return None

    cur_index = gold_index
    cur_index = advance_past_unaries(gold_sequence, cur_index)
    if gold_sequence[cur_index] == pred_transition:
        return gold_sequence[:gold_index] + gold_sequence[cur_index:]
    return None

def fix_open_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix an Open replaced with a Shift

    Suppose we were supposed to guess NT_X and instead did S

    We derive the repair as follows.

    For simplicity, assume the open is not a unary for now

    Since we know an Open was legal, there must be stuff
      stuff NT_X
    Shift is also legal, so there must be other stuff and a previous Open
      stuff_1 NT_Y stuff_2 NT_X
    After the NT_X which we missed, there was a bunch of stuff and a close for NT_X
      stuff_1 NT_Y stuff_2 NT_X stuff_3 C
    There could be more stuff here which can be saved...
      stuff_1 NT_Y stuff_2 NT_X stuff_3 C stuff_4 C
      stuff_1 NT_Y stuff_2 NT_X stuff_3 C C
    """
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None

    cur_index = gold_index
    cur_index = advance_past_unaries(gold_sequence, cur_index)
    if not isinstance(gold_sequence[cur_index], OpenConstituent):
        return None
    if gold_sequence[cur_index].top_label in root_labels:
        return None
    # cur_index now points to the NT_X we missed (not counting unaries)

    stuff_start = cur_index + 1
    # can't be a Close, since we just went past an Open and checked for unaries
    # can't be an Open, since two Open in a row is illegal
    assert isinstance(gold_sequence[stuff_start], Shift)
    stuff_end = advance_past_constituents(gold_sequence, stuff_start)
    # stuff_end is now the Close which ends NT_X
    cur_index = stuff_end + 1
    if cur_index >= len(gold_sequence):
        return None
    if isinstance(gold_sequence[cur_index], OpenConstituent):
        cur_index = advance_past_unaries(gold_sequence, cur_index)
        if cur_index >= len(gold_sequence):
            return None
    if isinstance(gold_sequence[cur_index], OpenConstituent):
        # an Open here signifies that there was a bracket containing X underneath Y
        # TODO: perhaps try to salvage something out of that situation?
        return None
    # the repair starts with the sequence up through the error,
    # then stuff_3, which includes the error
    # skip the Close for the missed NT_X
    # then finish the sequence with any potential stuff_4, the next Close, and everything else
    repair = gold_sequence[:gold_index] + gold_sequence[stuff_start:stuff_end] + gold_sequence[cur_index:]
    return repair

def fix_open_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix an Open replaced with a Close

    Call the Open NT_X
    Open legal, so there must be stuff:
      stuff NT_X
    Close legal, so there must be something to close:
      stuff_1 NT_Y stuff_2 NT_X

    The incorrect close makes the following brackets:
      (Y stuff_1 stuff_2)
    We were supposed to build
      (Y stuff_1 (X stuff_2 ...) (possibly more stuff))
    The simplest fix here is to reopen Y at this point.

    One issue might be if there is another bracket which encloses X underneath Y
    So, for example, the tree was supposed to be
      (Y stuff_1 (Z (X stuff_2 stuff_3) stuff_4))
    The pattern for this case is
      stuff_1 NT_Y stuff_2 NY_X stuff_3 close NT_Z stuff_4 close close
    """
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, CloseConstituent):
        return None

    cur_index = advance_past_unaries(gold_sequence, gold_index)
    if cur_index >= len(gold_sequence):
        return None
    if not isinstance(gold_sequence[cur_index], OpenConstituent):
        return None
    if gold_sequence[cur_index].top_label in root_labels:
        return None

    prev_open_index = find_previous_open(gold_sequence, gold_index)
    if prev_open_index is None:
        return None
    prev_open = gold_sequence[prev_open_index]
    # prev_open is now NT_Y from above

    stuff_start = cur_index + 1
    assert isinstance(gold_sequence[stuff_start], Shift)
    stuff_end = advance_past_constituents(gold_sequence, stuff_start)
    # stuff_end is now the Close which ends NT_X
    # stuff_start:stuff_end is the stuff_3 block above
    cur_index = stuff_end + 1
    if cur_index >= len(gold_sequence):
        return None
    # if there are unary transitions here, we want to skip those.
    # those are unary transitions on X and cannot be recovered, since X is gone
    cur_index = advance_past_unaries(gold_sequence, cur_index)
    # now there is a certain failure case which has to be accounted for.

    # specifically, if there is a new non-terminal which opens
    # immediately after X closes, it is encompassing X in a way that
    # cannot be recovered now that part of X is stuck under Y.
    # The two choices at this point would be to eliminate the new
    # transition or just reject the tree from the repair
    # For now, we reject the tree
    if isinstance(gold_sequence[cur_index], OpenConstituent):
        return None

    repair = gold_sequence[:gold_index] + [pred_transition, prev_open] + gold_sequence[stuff_start:stuff_end] + gold_sequence[cur_index:]
    return repair

def fix_shift_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    This fixes Shift replaced with a Close transition.

    This error occurs in the following pattern:
      stuff_1 NT_X stuff... shift
    Instead of shift, you close the NT_X
    The easiest fix here is to just restore the NT_X.
    """

    if not isinstance(pred_transition, CloseConstituent):
        return None

    # this fix can also be applied if there were unaries on the
    # previous constituent.  we just skip those until the Shift
    cur_index = gold_index
    if isinstance(gold_transition, OpenConstituent):
        cur_index = advance_past_unaries(gold_sequence, cur_index)
    if not isinstance(gold_sequence[cur_index], Shift):
        return None

    prev_open_index = find_previous_open(gold_sequence, gold_index)
    if prev_open_index is None:
        return None
    prev_open = gold_sequence[prev_open_index]
    # prev_open is now NT_X from above

    return gold_sequence[:gold_index] + [pred_transition, prev_open] + gold_sequence[cur_index:]

def fix_close_shift_nested(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Fix a Close X..Open X..Shift pattern where both the Close and Open were skipped.

    Here the pattern we are trying to fix is
      stuff_A open_X stuff_B *close* open_X shift...
    replaced with
      stuff_A open_X stuff_B shift...
    the missed close & open means a missed recall error for (X A B)
    whereas the previous open_X can still get the outer bracket
    """
    if not isinstance(gold_transition, CloseConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None

    if len(gold_sequence) < gold_index + 3:
        return None
    if not isinstance(gold_sequence[gold_index+1], OpenConstituent):
        return None

    # handle the sequence:
    #   stuff_A open_X stuff_B close open_Y close open_X shift
    open_index = advance_past_unaries(gold_sequence, gold_index+1)
    if not isinstance(gold_sequence[open_index], OpenConstituent):
        return None
    if not isinstance(gold_sequence[open_index+1], Shift):
        return None

    # check that the next operation was to open the same constituent
    # we just closed
    prev_open_index = find_previous_open(gold_sequence, gold_index)
    if prev_open_index is None:
        return None
    prev_open = gold_sequence[prev_open_index]
    if gold_sequence[open_index] != prev_open:
        return None

    return gold_sequence[:gold_index] + gold_sequence[open_index+1:]

def fix_close_shift_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    """
    Repair Close/Shift -> Shift by moving the Close to after the next block is created
    """
    if not isinstance(gold_transition, CloseConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None
    if len(gold_sequence) < gold_index + 2:
        return None
    start_index = gold_index + 1
    start_index = advance_past_unaries(gold_sequence, start_index)
    if len(gold_sequence) < start_index + 2:
        return None
    if not isinstance(gold_sequence[start_index], Shift):
        return None

    end_index = find_in_order_constituent_end(gold_sequence, start_index)
    if end_index is None:
        return None
    # if this *isn't* a close, we don't allow it.
    # that case seems to be ambiguous...
    #   stuff_1 close stuff_2 stuff_3
    # if you would normally start building stuff_3,
    # it is not clear if you want to close at the end of
    # stuff_2 or build stuff_3 instead.
    if not isinstance(gold_sequence[end_index], CloseConstituent):
        return None

    return gold_sequence[:gold_index] + gold_sequence[start_index:end_index] + [CloseConstituent()] + gold_sequence[end_index:]

def ambiguous_shift_open_unary_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, Shift):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    return gold_sequence[:gold_index] + [pred_transition, CloseConstituent()] + gold_sequence[gold_index:]

def ambiguous_shift_open_early_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, Shift):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    # Find when the current block ends,
    # either via a Shift or a Close
    end_index = find_in_order_constituent_end(gold_sequence, gold_index)
    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:end_index] + [CloseConstituent()] + gold_sequence[end_index:]

def ambiguous_shift_open_late_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, Shift):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    end_index = advance_past_constituents(gold_sequence, gold_index)
    return gold_sequence[:gold_index] + [pred_transition] + gold_sequence[gold_index:end_index] + [CloseConstituent()] + gold_sequence[end_index:]


def report_close_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, CloseConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None

    return RepairType.OTHER_CLOSE_SHIFT, None

def report_close_open(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, CloseConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    return RepairType.OTHER_CLOSE_OPEN, None

def report_open_open(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    return RepairType.OTHER_OPEN_OPEN, None

def report_open_shift(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, Shift):
        return None

    return RepairType.OTHER_OPEN_SHIFT, None

def report_open_close(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, OpenConstituent):
        return None
    if not isinstance(pred_transition, CloseConstituent):
        return None

    return RepairType.OTHER_OPEN_CLOSE, None

def report_shift_open(gold_transition, pred_transition, gold_sequence, gold_index, root_labels):
    if not isinstance(gold_transition, Shift):
        return None
    if not isinstance(pred_transition, OpenConstituent):
        return None

    return RepairType.OTHER_SHIFT_OPEN, None

class RepairType(Enum):
    """
    Keep track of which repair is used, if any, on an incorrect transition

    Statistics on English w/ no charlm, no transformer,
      eg word vectors only, best model as of January 2024

    unambiguous transitions only:
        oracle scheme          dev      test
         no oracle            0.9245   0.9226
          +wrong_open_root    0.9244   0.9224
          +wrong_unary_chain  0.9243   0.9237
          +wrong_open_unary   0.9249   0.9223
          +wrong_open_general 0.9251   0.9215
          +missed_unary       0.9248   0.9215
          +open_shift         0.9243   0.9216
          +open_close         0.9254   0.9217
          +shift_close        0.9261   0.9238
          +close_shift_nested 0.9253   0.9250

    Redoing the wrong_open_general, which seemed to hurt test scores:
          wrong_open_two_subtrees          0.9244   0.9220
          w/o ambiguous open               0.9261   0.9246
          w/ ambiguous open_three_subtrees 0.9264   0.9243

    Testing three different possible repairs for shift-open:
          immediate close (unary)          0.9267   0.9246
          close after first bracket        0.9265   0.9256
          close after last bracket         0.9264   0.9240
    """
    def __new__(cls, fn, correct=False, debug=False):
        """
        Enumerate values as normal, but also keep a pointer to a function which repairs that kind of error

        correct: this represents a correct transition

        debug: always run this, as it just counts statistics
        """
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value + 1
        obj.fn = fn
        obj.correct = correct
        obj.debug = debug
        return obj

    def is_correct(self):
        return self.correct

    # The first section is a sequence of repairs when the parser
    # should have chosen NTx but instead chose NTy

    # Blocks of transitions which can be abstracted away to be
    # anything will be represented as S1, S2, etc... S for stuff

    # We carve out an exception for a wrong open at the root
    # The only possble transtions at this point are to close
    # the error and try again with the root
    WRONG_OPEN_ROOT_ERROR  = (fix_wrong_open_root_error,)

    # The simplest form of such an error is when there is a sequence
    # of unary transitions and the parser chose a wrong parent.
    # Remember that a unary transition is represented by a pair
    # of transitions, NTx, Close.
    # In this case, the correct sequence was
    #   S1 NTx Close NTy Close NTz ...
    # but the parser chose NTy, NTz, etc
    # The repair in this case is to simply discard the unchosen
    # unary transitions and continue
    WRONG_OPEN_UNARY_CHAIN = (fix_wrong_open_unary_chain,)

    # Similar to the UNARY_CHAIN error, but in this case there is a
    # bunch of stuff (one or more constituents built) between the
    # missed open transition and the close transition
    WRONG_OPEN_STUFF_UNARY = (fix_wrong_open_stuff_unary,)

    # If the correct sequence is
    #   T1 O_x T2 C
    # and instead we predicted
    #   T1 O_y ...
    # this can be fixed with a unary transition after
    #   T1 O_y T2 C O_x C
    # note that this is technically ambiguous
    # could have done
    #   T1 O_x C O_y T2 C
    # but doing this should be easier for the parser to detect (untested)
    # also this way the same code paths can be used for two subtrees
    # and for multiple subtrees
    WRONG_OPEN_TWO_SUBTREES = (fix_wrong_open_two_subtrees,)

    # If the gold transition is an Open because it is part of
    # a unary transition, and the following transition is a
    # correct Shift or Close, we can just skip past the unary.
    MISSED_UNARY           = (fix_missed_unary,)

    # Open -> Shift errors which don't just represent a unary
    # generally represent a missing bracket which cannot be
    # recovered using the in-order mechanism.  Dropping the
    # missing transition is generally the only fix.
    # (This means removing the corresponding Close)
    # One could theoretically create a new transition which
    # grabs two constituents, though
    OPEN_SHIFT             = (fix_open_shift,)

    # Open -> Close is a rather drastic break in the
    # potential structure of the tree.  We can no longer
    # recover the missed Open, and we might not be able
    # to recover other following missed Opens as well.
    # In most cases, the only thing to do is reopen the
    # incorrectly closed outer bracket and keep going.
    OPEN_CLOSE             = (fix_open_close,)

    # Similar to the Open -> Close error, but at least
    # in this case we are just introducing one wrong bracket
    # rather than also breaking some existing brackets.
    # The fix here is to reopen the closed bracket.
    SHIFT_CLOSE            = (fix_shift_close,)

    # Specifically fixes an error where bracket X is
    # closed and then immediately opened to build a
    # new X bracket.  In this case, the simplest fix
    # will be to skip both the close and the new open
    # and continue from there.
    CLOSE_SHIFT_NESTED     = (fix_close_shift_nested,)

    # If the model is supposed to build a block after a Close
    # operation, attach that block to the piece to the left
    # a couple different variations on this were tried
    # we tried attaching all constituents to the
    #   bracket which should have been closed
    # we tried attaching exactly one constituent
    # and we tried attaching only if there was
    #   exactly one following constituent
    # none of these improved f1.  for example, on the VI dataset, we
    # lost 0.15 F1 with the exactly one following constituent version
    # it might be worthwhile double checking some of the other
    # versions to make sure those also fail, though
    # CLOSE_SHIFT_SHIFT      = (fix_close_shift_shift,)

    # Similarly to WRONG_OPEN_TWO_SUBTREES, if the correct sequence is
    #   T1 O_x T2 T3 C
    # and instead we predicted
    #   T1 O_y ...
    # this can be fixed by closing O_y in any number of places
    #   T1 O_y T2 C O_x T3 C
    #   T1 O_y T2 C T3 O_x C
    # Either solution is a single precision error,
    # but keeps the O_x subtree correct
    # This is an ambiguous transition - we can experiment with different fixes
    WRONG_OPEN_MULTIPLE_SUBTREES = (fix_wrong_open_multiple_subtrees,)

    CORRECT                = (None, True)

    UNKNOWN                = None

    # This particular repair effectively turns the shift -> ambiguous open
    # into a unary transition
    SHIFT_OPEN_UNARY_CLOSE       = (ambiguous_shift_open_unary_close,)

    # Fix the shift -> ambiguous open by closing after the first constituent
    # This is an ambiguous solution because it could also be closed either
    # as a unary transition or with a close at the end of the outer bracket
    SHIFT_OPEN_EARLY_CLOSE       = (ambiguous_shift_open_early_close,)

    # Fix the shift -> ambiguous open by closing after all constituents
    # This is an ambiguous solution because it could also be closed either
    # as a unary transition or with a close at the end of the first constituent
    SHIFT_OPEN_LATE_CLOSE        = (ambiguous_shift_open_late_close,)

    OTHER_CLOSE_SHIFT            = (report_close_shift, False, True)

    OTHER_CLOSE_OPEN             = (report_close_open, False, True)

    OTHER_OPEN_OPEN              = (report_open_open, False, True)

    OTHER_OPEN_CLOSE             = (report_open_close, False, True)

    OTHER_OPEN_SHIFT             = (report_open_shift, False, True)

    OTHER_SHIFT_OPEN             = (report_shift_open, False, True)

    # any other open transition we get wrong, which hasn't already
    # been carved out as an exception above, we just accept the
    # incorrect Open and keep going
    #
    # TODO: check if there is a way to improve this
    # it appears to hurt scores simply by existing
    # explanation: this is wrong logic
    # Suppose the correct sequence had been
    #   T1 open(NP) T2 T3 close
    # Instead we had done
    #   T1 open(VP) T2 T3 close
    # We can recover the missing NP!
    #   T1 open(VP) T2 close open(NP) T3 close
    # Can also recover it as
    #   T1 open(VP) T2 T3 close open(NP) close
    # So this is actually an ambiguous transition
    # except in the case of
    #   T1 open(...) close
    # In this case, a unary transition can fix make it so we only have
    # a precision error, not also a recall error
    # Currently, the approach is to put this after the default fixes
    # and use the two & more-than-two versions of the fix above
    WRONG_OPEN_GENERAL     = (fix_wrong_open_general,)

class InOrderOracle(DynamicOracle):
    def __init__(self, root_labels, oracle_level, additional_oracle_levels, deactivated_oracle_levels):
        super().__init__(root_labels, oracle_level, RepairType, additional_oracle_levels, deactivated_oracle_levels)
