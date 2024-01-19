from stanza.models.constituency.parse_transitions import Shift, OpenConstituent, CloseConstituent

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

class DynamicOracle():
    def __init__(self, root_labels, oracle_level, repair_types):
        self.root_labels = root_labels
        self.oracle_level = oracle_level
        self.repair_types = repair_types

    def fix_error(self, gold_transition, pred_transition, gold_sequence, gold_index):
        """
        Return which error has been made, if any, along with an updated transition list

        We assume the transition sequence builds a correct tree, meaning
        that there will always be a CloseConstituent sometime after an
        OpenConstituent, for example
        """
        assert gold_sequence[gold_index] == gold_transition

        if gold_transition == pred_transition:
            return self.repair_types.CORRECT, None

        for repair_type in self.repair_types:
            if repair_type.fn is None:
                continue
            if self.oracle_level is not None and repair_type.value > self.oracle_level:
                continue
            repair = repair_type.fn(gold_transition, pred_transition, gold_sequence, gold_index, self.root_labels)
            if repair is not None:
                return repair_type, repair

        return self.repair_types.UNKNOWN, None
