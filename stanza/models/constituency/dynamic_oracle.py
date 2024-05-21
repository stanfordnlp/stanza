from collections import namedtuple

import numpy as np

from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent

RepairEnum = namedtuple("RepairEnum", "name value is_correct")

def score_candidates(model, state, candidates, candidate_idx):
    """
    score candidate fixed sequences by summing up the transition scores of the most important block

    the candidate with the best summed score is chosen, and the candidate sequence is reconstructed from the blocks
    """
    scores = []
    # could bulkify this if we wanted
    for candidate in candidates:
        current_state = [state]
        for block in candidate[1:candidate_idx]:
            for transition in block:
                current_state = model.bulk_apply(current_state, [transition])
        score = 0.0
        for transition in candidate[candidate_idx]:
            predictions = model.forward(current_state)
            t_idx = model.transition_map[transition]
            score += predictions[0, t_idx].cpu().item()
            current_state = model.bulk_apply(current_state, [transition])
        scores.append(score)
    best_idx = np.argmax(scores)
    best_candidate = [x for block in candidates[best_idx] for x in block]
    return scores, best_idx, best_candidate

def advance_past_constituents(gold_sequence, cur_index):
    """
    Advance cur_index through gold_sequence until we have seen 1 more Close than Open

    The index returned is the index of the Close which occurred after all the stuff
    """
    count = 0
    while cur_index < len(gold_sequence):
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
            if count == -1: return cur_index
        cur_index = cur_index + 1
    return None

def find_previous_open(gold_sequence, cur_index):
    """
    Go backwards from cur_index to find the open which opens the previous block of stuff.

    Return None if it can't be found.
    """
    count = 0
    cur_index = cur_index - 1
    while cur_index >= 0:
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
            if count > 0:
                return cur_index
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
        cur_index = cur_index - 1
    return None

def find_in_order_constituent_end(gold_sequence, cur_index):
    """
    Advance cur_index through gold_sequence until the next block has ended

    This is different from advance_past_constituents in that it will
    also return when there is a Shift when count == 0.  That way, we
    return the first block of things we know attach to the left
    """
    count = 0
    saw_shift = False
    while cur_index < len(gold_sequence):
        if isinstance(gold_sequence[cur_index], OpenConstituent):
            count = count + 1
        elif isinstance(gold_sequence[cur_index], CloseConstituent):
            count = count - 1
            if count == -1: return cur_index
        elif isinstance(gold_sequence[cur_index], Shift):
            if saw_shift and count == 0:
                return cur_index
            else:
                saw_shift = True
        cur_index = cur_index + 1
    return None

class DynamicOracle():
    def __init__(self, root_labels, oracle_level, repair_types, additional_levels, deactivated_levels):
        self.root_labels = root_labels
        # default oracle_level will be the UNKNOWN repair type (which each oracle should have)
        # transitions after that as experimental or ambiguous, not to be used by default
        self.oracle_level = oracle_level if oracle_level is not None else repair_types.UNKNOWN.value
        self.repair_types = repair_types
        self.additional_levels = set()
        if additional_levels:
            self.additional_levels = set([repair_types[x.upper()] for x in additional_levels.split(",")])
        self.deactivated_levels = set()
        if deactivated_levels:
            self.deactivated_levels = set([repair_types[x.upper()] for x in deactivated_levels.split(",")])

    def fix_error(self, pred_transition, model, state):
        """
        Return which error has been made, if any, along with an updated transition list

        We assume the transition sequence builds a correct tree, meaning
        that there will always be a CloseConstituent sometime after an
        OpenConstituent, for example
        """
        gold_transition = state.gold_sequence[state.num_transitions]
        if gold_transition == pred_transition:
            return self.repair_types.CORRECT, None

        for repair_type in self.repair_types:
            if repair_type.fn is None:
                continue
            if self.oracle_level is not None and repair_type.value > self.oracle_level and repair_type not in self.additional_levels and not repair_type.debug:
                continue
            if repair_type in self.deactivated_levels:
                continue
            repair = repair_type.fn(gold_transition, pred_transition, state.gold_sequence, state.num_transitions, self.root_labels, model, state)
            if repair is None:
                continue

            if isinstance(repair, tuple) and len(repair) == 2:
                return repair

            # TODO: could update all of the returns to be tuples of length 2
            if repair is not None:
                return repair_type, repair

        return self.repair_types.UNKNOWN, None
