from abc import ABC, abstractmethod

class DynamicOracle(ABC):
    @abstractmethod
    def fix_error(self, gold_transition, pred_transition, gold_sequence, gold_index):
        """
        If this oracle can fix this error, return the error type and the new sequnce

        gold_transition: expected transition
        pred_transition: what was predicted instead
        gold_sequence: the entire sequence of gold transitions
        gold_index: where we are in the sequence

        return: an enum describing the repair, and a replacement for gold_sequence
          None for gold_sequence means no repair was possible
            (or needed, in the case of a correct pred_transition)
        """
