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
