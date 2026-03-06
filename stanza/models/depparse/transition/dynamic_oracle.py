"""
A series of repairs for possible errors in a transition sequence chosen at training time.

For example, if the parser chooses a left reduce transition with the wrong label,
the simplest repair is to allow that transition and make no further changes to the sequence.
"""

from enum import Enum

from stanza.models.depparse.transition.transitions import ProjectiveLeft, NonprojectiveLeft, ProjectiveRight, NonprojectiveRight

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


