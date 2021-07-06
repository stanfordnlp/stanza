from abc import ABC, abstractmethod

from stanza.models.constituency.parse_tree import Tree

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
    def push_word(self, word_queue, word):
        """
        word actually means a ParseTree with a tag node and word node
        """
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
    def build_constituent(self, label, children):
        """
        Combine the given children into a new node using the label

        The children are the entire elements popped from the stack, not just the nodes
        """
        pass

    @abstractmethod
    def push_constituent(self, constituents, constituent):
        pass

    @abstractmethod
    def get_top_constituent(self, constituents):
        pass

    @abstractmethod
    def push_transition(self, transitions, transition):
        pass

    @abstractmethod
    def get_top_transition(self, transitions):
        pass

    def get_root_labels(self):
        return ("ROOT",)

class SimpleModel(BaseModel):
    """
    This model allows pushing and popping with no extra data
    """
    def push_word(self, word_queue, word):
        return word_queue.push(word)

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

    def build_constituent(self, label, children):
        return Tree(label=label, children=children)

    def push_constituent(self, constituents, constituent):
        return constituents.push(constituent)

    def get_top_constituent(self, constituents):
        return constituents.value

    def push_transition(self, transitions, transition):
        return transitions.push(transition)

    def get_top_transition(self, transitions):
        return transitions.value
