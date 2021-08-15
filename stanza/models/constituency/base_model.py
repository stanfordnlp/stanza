from abc import ABC, abstractmethod

from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack

"""
The BaseModel is used to perform dependency injection on the transitions.

For example, a SimpleModel simply looks at the top of the various stacks in the state.

A model with LSTM scores for the different transitions may attach the
hidden and output states of the LSTM to the stacks, with the result
that some extra effort is needed to get the previous value
"""

class BaseModel(ABC):
    """
    This base class defines abstract methods for manipulating a State.

    Applying transitions may change important metadata about a State
    such as the vectors associated with LSTM hidden states, for example.
    """

    @abstractmethod
    def initial_word_queues(self, tagged_word_lists):
        """
        For each list of tagged words, builds a TreeStack of word nodes

        The word lists should be backwards so that the first word is the last word put on the stack (LIFO)
        """
        pass

    @abstractmethod
    def initial_transitions(self):
        pass

    @abstractmethod
    def initial_constituents(self):
        pass

    @abstractmethod
    def get_top_word(self, word_queue):
        pass

    @abstractmethod
    def transform_word_to_constituent(self, state):
        """
        Transform the top node of word_queue to something that can push on the constituent stack
        """
        pass

    @abstractmethod
    def dummy_constituent(self, dummy):
        """
        When using a dummy node as a sentinel, transform it to something usable by this model
        """
        pass

    @abstractmethod
    def unary_transform(self, constituents, labels):
        """
        Transform the top of the constituent stack using a unary transform to the new label
        """
        pass

    @abstractmethod
    def build_constituents(self, labels, children_lists):
        """
        Build multiple constituents at once.  This gives the opportunity for batching operations
        """
        pass

    @abstractmethod
    def push_constituents(self, constituent_stacks, constituents):
        pass

    @abstractmethod
    def get_top_constituent(self, constituents):
        pass

    @abstractmethod
    def push_transitions(self, transition_stacks, transitions):
        pass

    @abstractmethod
    def get_top_transition(self, transitions):
        pass

    def get_root_labels(self):
        return ("ROOT",)

    @abstractmethod
    def has_unary_transitions(self):
        pass

class SimpleModel(BaseModel):
    """
    This model allows pushing and popping with no extra data
    """
    def initial_word_queues(self, tagged_word_lists):
        word_queues = []
        for tagged_words in tagged_word_lists:
            word_queue = TreeStack(value=None)
            for tag_node in tagged_words:
                word_queue = word_queue.push(tag_node)
            word_queues.append(word_queue)
        return word_queues

    def initial_transitions(self):
        return TreeStack(value=None)

    def initial_constituents(self):
        return TreeStack(value=None)

    def get_top_word(self, word_queue):
        return word_queue.value

    def transform_word_to_constituent(self, state):
        return state.word_queue.value

    def dummy_constituent(self, dummy):
        return dummy

    def unary_transform(self, constituents, labels):
        top_constituent = constituents.value
        for label in reversed(labels):
            top_constituent = Tree(label=label, children=[top_constituent])
        return top_constituent

    def build_constituents(self, labels, children_lists):
        constituents = []
        for label, children in zip(labels, children_lists):
            if isinstance(label, str):
                label = (label,)
            for value in reversed(label):
                children = Tree(label=value, children=children)
            constituents.append(children)
        return constituents

    def push_constituents(self, constituent_stacks, constituents):
        return [stack.push(constituent) for stack, constituent in zip(constituent_stacks, constituents)]

    def get_top_constituent(self, constituents):
        return constituents.value

    def push_transitions(self, transition_stacks, transitions):
        return [stack.push(transition) for stack, transition in zip(transition_stacks, transitions)]

    def get_top_transition(self, transitions):
        return transitions.value

    def has_unary_transitions(self):
        # TODO: make this an option for testing purposes
        return True
