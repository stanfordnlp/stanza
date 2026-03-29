"""
A series of repairs for possible errors in a transition sequence chosen at training time.

For example, if the parser chooses a left reduce transition with the wrong label,
the simplest repair is to allow that transition and make no further changes to the sequence.

Example of the proportion of errors covered by the end of a Chinese training run:

Oracle repairs:
  CORRECT (7): 177818
  UNKNOWN (8): 4335
  RIGHT_WRONG_RELATION (1): 3184
  RIGHT_INSTEAD_OF_SHIFT_RIGHT_HEAD (5): 1394
  SHIFT_INSTEAD_OF_RIGHT (6): 921
  LEFT_WRONG_RELATION (2): 829
  LEFT_INSTEAD_OF_SHIFT_RIGHT_HEAD (4): 462
  LEFT_WRONG_RELATION_WRONG_HEAD (3): 179

Scores of using this oracle on a few different treebanks:

5 model dev avg LAS     without          oracle
de_gsd                   88.93            89.00
en_ewt                   93.45            93.53
fi_tdt                   92.82            92.81
it_vit                   90.02            90.05
ta_ttb                   72.05            72.13
zh-hans_gsdsimp          85.14            85.33

5 model test avg LAS    without          oracle
de_gsd                   86.49            86.66
en_ewt                   93.23            93.40
fi_tdt                   92.91            93.01
it_vit                   90.35            90.32
ta_ttb                   68.90            68.62
zh-hans_gsdsimp          85.26            85.45

Interestingly, it seems to help more with larger treebanks than smaller.
"""

from enum import Enum

from stanza.models.depparse.transition.transitions import Shift, ProjectiveLeft, NonprojectiveLeft, ProjectiveRight, NonprojectiveRight

def fix_right_wrong_relation(gold_sequence, transition_idx, gold_transition, chosen_transition):
    """
    If the head attachments are correct, but the relation chosen is wrong, this is both a precision and a recall error.
    However, there is no way to fix that
    Otherwise, this does not change the structure of the tree.  So we can continue as if nothing had happened
    """
    # TODO: is it possible to choose a NonprojectiveRight with the exact head we want?
    if isinstance(gold_transition, ProjectiveRight):
        if not isinstance(chosen_transition, ProjectiveRight):
            return None
        # actually, this should already be considered CORRECT
        if gold_transition.deprel == chosen_transition.deprel:
            return None
    elif isinstance(gold_transition, NonprojectiveRight):
        if not isinstance(chosen_transition, NonprojectiveRight):
            return None
        if gold_transition.word_idx != chosen_transition.word_idx:
            return None
        # actually, this should already be considered CORRECT
        if gold_transition.deprel == chosen_transition.deprel:
            return None
    else:
        return None

    new_sequence = gold_sequence[:transition_idx] + [chosen_transition] + gold_sequence[transition_idx+1:]
    return new_sequence

def fix_left_wrong_relation(gold_sequence, transition_idx, gold_transition, chosen_transition):
    """
    Same as with the right attached head, we can continue with the wrong relation
    """
    # TODO: is it possible to choose a NonprojectiveLeft with the exact head we want?
    if isinstance(gold_transition, ProjectiveLeft):
        if not isinstance(chosen_transition, ProjectiveLeft):
            return None
        # actually, this should already be considered CORRECT
        if gold_transition.deprel == chosen_transition.deprel:
            return None
    elif isinstance(gold_transition, NonprojectiveLeft):
        if not isinstance(chosen_transition, NonprojectiveLeft):
            return None
        if gold_transition.word_idx != chosen_transition.word_idx:
            return None
        # actually, this should already be considered CORRECT
        if gold_transition.deprel == chosen_transition.deprel:
            return None
    else:
        return None

    new_sequence = gold_sequence[:transition_idx] + [chosen_transition] + gold_sequence[transition_idx+1:]
    return new_sequence

def fix_left_wrong_relation_wrong_head(gold_sequence, transition_idx, gold_transition, chosen_transition):
    """
    Although the structure of the tree changes, if we attached to the wrong left head,
    the remaining heads will be the same.
    Therefore, left attach to the wrong head is still fixable by
    continuing the remaining sequence
    Note that this is different from attaching the wrong head in the right direction,
    as the remaining heads after a wrong right attach are different
    """
    # TODO: is it possible to choose a NonprojectiveLeft with the exact head we want?
    if isinstance(gold_transition, ProjectiveLeft):
        if not isinstance(chosen_transition, NonprojectiveLeft):
            return None
    elif isinstance(gold_transition, NonprojectiveLeft):
        if isinstance(chosen_transition, NonprojectiveLeft):
            if gold_transition.word_idx == chosen_transition.word_idx:
                # this is covered under left_wrong_relation
                return None
        elif not isinstance(chosen_transition, ProjectiveLeft):
            return None
    else:
        return None

    #print(gold_transition, chosen_transition)
    #print(gold_sequence)
    new_sequence = gold_sequence[:transition_idx] + [chosen_transition] + gold_sequence[transition_idx+1:]
    #print(new_sequence)
    return new_sequence

def find_subtree_end(gold_sequence, current_index):
    """
    Build onto the current subtree on the stack until we would connect it somewhere else,
    such as a PL or PR that affects the location before the current subtree
    """
    subtree_count = 1
    while current_index < len(gold_sequence) and (subtree_count > 1 or (subtree_count == 1 and isinstance(gold_sequence[current_index], Shift))):
        if isinstance(gold_sequence[current_index], Shift):
            subtree_count += 1
        elif isinstance(gold_sequence[current_index], (ProjectiveLeft, ProjectiveRight)):
            subtree_count -= 1
        else:
            # TODO: could theoretically save some trees where
            # there is a Nonprojective that stays inside the
            # current list of subtrees
            return None
        current_index += 1
    if current_index == len(gold_sequence):
        return None
    return current_index

def fix_left_instead_of_shift_right_head(gold_sequence, transition_idx, gold_transition, chosen_transition):
    if not isinstance(gold_transition, Shift):
        return None
    if not isinstance(chosen_transition, (ProjectiveLeft, NonprojectiveLeft)):
        return None

    # TODO: a ProjectiveLeft can technically be fixed with a NonprojectiveLeft...
    if isinstance(gold_sequence[transition_idx+1], (NonprojectiveRight, NonprojectiveLeft, ProjectiveLeft)):
        return None
    # in this case, the word to the left would have immediately attached to the newly added word
    # this is clearly fixable by just ignoring that attachment
    if isinstance(gold_sequence[transition_idx+1], ProjectiveRight):
        new_sequence = gold_sequence[:transition_idx] + [chosen_transition, gold_sequence[transition_idx]] + gold_sequence[transition_idx+2:]
        return new_sequence

    if isinstance(gold_sequence[transition_idx+1], Shift):
        current_index = find_subtree_end(gold_sequence, transition_idx+1)
        if current_index is None:
            return None
        # now we should have a single subtree and gold_sequence[current_index] is not a Shift
        if isinstance(gold_sequence[current_index], ProjectiveRight):
            # the case of a ProjectiveRight is the fixable case, as now the edge has been replaced with chosen_transition
            # and there are no further errors being made when we build the rest of the tree
            new_sequence = gold_sequence[:transition_idx] + [chosen_transition] + gold_sequence[transition_idx:current_index] + gold_sequence[current_index+1:]
            return new_sequence
    return None

def fix_right_instead_of_shift_right_head(gold_sequence, transition_idx, gold_transition, chosen_transition):
    """
    This pattern occurs when a subtree was supposed to be  attached at a later point, but was instead attached now.

    so the sequence was

    A B C

    attached
    B -> C

    when instead we should have done

    A B C D
    somehow connect C D
    *then* attach B to the result of connecting C and D
    """
    if not isinstance(gold_transition, Shift):
        return None
    if not isinstance(chosen_transition, ProjectiveRight):
        return None

    current_index = find_subtree_end(gold_sequence, transition_idx)
    if current_index is not None and isinstance(gold_sequence[current_index], ProjectiveRight):
        # This is the case where we had
        #  A B C
        # connected B -> C immediately
        # but instead should have shifted D
        # then connected C to D immediately (either head)
        # and then connected B -> C/D
        # or there can also be stuff built onto C/D first
        new_sequence = gold_sequence[:transition_idx] + [chosen_transition] + gold_sequence[transition_idx:current_index] + gold_sequence[current_index+1:]
        return new_sequence

    return None

def fix_shift_instead_of_right(gold_sequence, transition_idx, gold_transition, chosen_transition):
    """A fixable scenario here is when the 2nd-from-top subtree could
    be merged right, but instead the top subtree is built on further.

    In such a situation, the right transition is still possible

    A B C
    should merge B -> C
    instead build C' (same root)
    can still merge B -> C
    """
    if not isinstance(gold_transition, ProjectiveRight):
        return None
    if not isinstance(chosen_transition, Shift):
        return None
    if not isinstance(gold_sequence[transition_idx+1], Shift):
        return None

    current_index = find_subtree_end(gold_sequence, transition_idx+2)
    if not current_index:
        return None

    # ProjectiveLeft is the fixable case, as it means the subtree still has the same root.
    # Thus attaching the 2nd from top subtree actually creates zero errors
    if not isinstance(gold_sequence[current_index], ProjectiveLeft):
        return None
    new_sequence = gold_sequence[:transition_idx] + gold_sequence[transition_idx+1:current_index+1] + [gold_transition] + gold_sequence[current_index+1:]
    return new_sequence

class RepairType(Enum):
    def __new__(cls, fn, correct=False, debug=False):
        """
        Enumerate values as normal, but also keep a pointer to a function which repairs that kind of error
        """
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value + 1
        obj.fn = fn
        return obj

    RIGHT_WRONG_RELATION                         = (fix_right_wrong_relation,)

    LEFT_WRONG_RELATION                          = (fix_left_wrong_relation,)

    LEFT_WRONG_RELATION_WRONG_HEAD               = (fix_left_wrong_relation_wrong_head,)

    LEFT_INSTEAD_OF_SHIFT_RIGHT_HEAD             = (fix_left_instead_of_shift_right_head,)

    RIGHT_INSTEAD_OF_SHIFT_RIGHT_HEAD            = (fix_right_instead_of_shift_right_head,)

    SHIFT_INSTEAD_OF_RIGHT                       = (fix_shift_instead_of_right,)

    CORRECT                                      = (None, )

    UNKNOWN                                      = (None, )

def repair(state, gold_transition, oracle_transition):
    gold_sequence = state.gold_sequence
    transition_idx = len(state.transitions)
    assert gold_sequence[transition_idx] == gold_transition

    if gold_transition == oracle_transition:
        return RepairType.CORRECT, None

    for repair_type in RepairType:
        if repair_type.fn is None:
            continue
        repair = repair_type.fn(gold_sequence, transition_idx, gold_transition, oracle_transition)
        if repair is None:
            continue

        return repair_type, repair

    return RepairType.UNKNOWN, None


