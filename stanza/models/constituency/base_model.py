"""
The BaseModel is passed to the transitions so that the transitions
can operate on a parsing state without knowing the exact
representation used in the model.

For example, a SimpleModel simply looks at the top of the various stacks in the state.

A model with LSTM representations for the different transitions may
attach the hidden and output states of the LSTM to the word /
constituent / transition stacks.

Reminder: the parsing state is a list of words to parse, the
transitions used to build a (possibly incomplete) parse, and the
constituent(s) built so far by those transitions.  Each of these
components are represented using stacks to improve the efficiency
of operations such as "combine the most recent 4 constituents"
or "turn the next input word into a constituent"
"""

from abc import ABC, abstractmethod

from stanza.models.constituency.parse_transitions import State, TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack

# default unary limit.  some treebanks may have longer chains (CTB, for example)
UNARY_LIMIT = 4

class BaseModel(ABC):
    """
    This base class defines abstract methods for manipulating a State.

    Applying transitions may change important metadata about a State
    such as the vectors associated with LSTM hidden states, for example.

    The constructor forwards all unused arguments to other classes in the
    constructor sequence, so put this before other classes such as nn.Module
    """
    def __init__(self, transition_scheme, unary_limit, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

        self._transition_scheme = transition_scheme
        self._unary_limit = unary_limit

    @abstractmethod
    def initial_word_queues(self, tagged_word_lists):
        """
        For each list of tagged words, builds a TreeStack of word nodes

        The word lists should be backwards so that the first word is the last word put on the stack (LIFO)
        """

    @abstractmethod
    def initial_transitions(self):
        """
        Builds an initial transition stack with whatever values need to go into first position
        """

    @abstractmethod
    def initial_constituents(self):
        """
        Builds an initial constituent stack with whatever values need to go into first position
        """

    @abstractmethod
    def get_word(self, word_node):
        """
        Get the word corresponding to this position in the word queue
        """

    @abstractmethod
    def transform_word_to_constituent(self, state):
        """
        Transform the top node of word_queue to something that can push on the constituent stack
        """

    @abstractmethod
    def dummy_constituent(self, dummy):
        """
        When using a dummy node as a sentinel, transform it to something usable by this model
        """

    @abstractmethod
    def unary_transform(self, constituents, labels):
        """
        Transform the top of the constituent stack using a unary transform to the new label
        """

    @abstractmethod
    def build_constituents(self, labels, children_lists):
        """
        Build multiple constituents at once.  This gives the opportunity for batching operations
        """

    @abstractmethod
    def push_constituents(self, constituent_stacks, constituents):
        """
        Add a multiple constituents to multiple constituent_stacks

        Useful to factor this out in case batching will help
        """

    @abstractmethod
    def get_top_constituent(self, constituents):
        """
        Get the first constituent from the constituent stack

        For example, a model might want to remove embeddings and LSTM state vectors
        """

    @abstractmethod
    def push_transitions(self, transition_stacks, transitions):
        """
        Add a multiple transitions to multiple transition_stacks

        Useful to factor this out in case batching will help
        """

    @abstractmethod
    def get_top_transition(self, transitions):
        """
        Get the first transition from the transition stack

        For example, a model might want to remove transition embeddings before returning the transition
        """

    def get_root_labels(self):
        """
        Return ROOT labels for this model.  Probably ROOT, TOP, or both
        """
        return ("ROOT",)

    def unary_limit(self):
        """
        Limit on the number of consecutive unary transitions
        """
        return self._unary_limit


    def transition_scheme(self):
        """
        Transition scheme used - see parse_transitions
        """
        return self._transition_scheme

    def has_unary_transitions(self):
        """
        Whether or not this model uses unary transitions, based on transition_scheme
        """
        return self._transition_scheme is TransitionScheme.TOP_DOWN_UNARY

    def is_top_down(self):
        """
        Whether or not this model is TOP_DOWN
        """
        return not self._transition_scheme is TransitionScheme.IN_ORDER

    def predict(self, states, is_legal=True):
        raise NotImplementedError("LSTMModel can predict, but SimpleModel cannot")

    def weighted_choice(self, states):
        raise NotImplementedError("LSTMModel can weighted_choice, but SimpleModel cannot")

    def predict_gold(self, states):
        """
        For each State, return the next item in the gold_sequence
        """
        return None, [y.gold_sequence[y.num_transitions()] for y in states]

    def initial_state_from_preterminals(self, preterminal_lists, gold_trees):
        """
        what is passed in should be a list of list of preterminals
        """
        word_queues = self.initial_word_queues(preterminal_lists)
        # this is the bottom of the TreeStack and will be the same for each State
        transitions = self.initial_transitions()
        constituents = self.initial_constituents()
        states = [State(sentence_length=len(wq)-2,   # -2 because it starts and ends with a sentinel
                        num_opens=0,
                        word_queue=wq,
                        gold_tree=None,
                        gold_sequence=None,
                        transitions=transitions,
                        constituents=constituents,
                        word_position=0)
                  for idx, wq in enumerate(word_queues)]
        if gold_trees:
            states = [state._replace(gold_tree=gold_tree) for gold_tree, state in zip(gold_trees, states)]
        return states

    def initial_state_from_words(self, word_lists):
        preterminal_lists = [[Tree(tag, Tree(word)) for word, tag in words]
                             for words in word_lists]
        return self.initial_state_from_preterminals(preterminal_lists, gold_trees=None)

    def initial_state_from_gold_trees(self, trees):
        preterminal_lists = [[Tree(pt.label, Tree(pt.children[0].label))
                              for pt in tree.yield_preterminals()]
                             for tree in trees]
        return self.initial_state_from_preterminals(preterminal_lists, gold_trees=trees)

class SimpleModel(BaseModel):
    """
    This model allows pushing and popping with no extra data
    """
    def __init__(self, transition_scheme=TransitionScheme.TOP_DOWN_UNARY, unary_limit=UNARY_LIMIT):
        super().__init__(transition_scheme=transition_scheme, unary_limit=unary_limit)

    def initial_word_queues(self, tagged_word_lists):
        word_queues = []
        for tagged_words in tagged_word_lists:
            word_queue =  [None]
            word_queue += [tag_node for tag_node in tagged_words]
            word_queue.append(None)
            word_queues.append(word_queue)
        return word_queues

    def initial_transitions(self):
        return TreeStack(value=None, parent=None, length=1)

    def initial_constituents(self):
        return TreeStack(value=None, parent=None, length=1)

    def get_word(self, word_node):
        return word_node

    def transform_word_to_constituent(self, state):
        return state.get_word(state.word_position)

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
