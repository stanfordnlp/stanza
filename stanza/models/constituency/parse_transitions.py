"""
Defines a series of transitions (open a constituent, close a constituent, etc

Also defines a State which holds the various data needed to build
a parse tree out of tagged words.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from enum import Enum
import functools
import logging

from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack

logger = logging.getLogger('stanza')

class TransitionScheme(Enum):
    TOP_DOWN           = 1
    TOP_DOWN_COMPOUND  = 2
    TOP_DOWN_UNARY     = 3

EMPTY_TREE_STACK = TreeStack(value=None)

class State:
    def __init__(self, original_state=None, sentence_length=None, num_opens=None,
                 word_queue=None, transitions=None, constituents=None, gold_tree=None, gold_sequence=None):
        """
        num_opens is useful for tracking
           1) if the parser is in a stuck state where it is making infinite opens
           2) if a close transition is impossible because there are no previous opens

        non-stack information such as sentence_length and num_opens
        will be copied from the original_state if possible, with the
        exact arguments overriding the values in the original_state

        gold_tree: the original tree, if made from a gold tree.  might be None
        gold_sequence: the original transition sequence, if available
        Note that at runtime, gold values will not be available
        """
        # first, copy all information from the original_state if it exists
        if original_state is None:
            assert not sentence_length is None, "Must provide either an original_state or a sentence_length"
            assert not num_opens is None, "Must provide either an original_state or num_opens"

            self.word_queue = EMPTY_TREE_STACK
            self.transitions = EMPTY_TREE_STACK
            self.constituents = EMPTY_TREE_STACK

            self.gold_tree = None
            self.gold_sequence = None
        else:
            self.sentence_length = original_state.sentence_length
            self.num_opens = original_state.num_opens

            self.word_queue = original_state.word_queue
            self.transitions = original_state.transitions
            self.constituents = original_state.constituents

            self.gold_tree = original_state.gold_tree
            self.gold_sequence = original_state.gold_sequence

        if word_queue is not None:
            self.word_queue = word_queue

        if transitions is not None:
            self.transitions = transitions

        if constituents is not None:
            self.constituents = constituents

        if num_opens is not None:
            self.num_opens = num_opens

        if sentence_length is not None:
            self.sentence_length = sentence_length

        if gold_tree is not None:
            self.gold_tree = gold_tree

        if gold_sequence is not None:
            self.gold_sequence = gold_sequence

    def empty_word_queue(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.word_queue.parent is None

    def empty_transitions(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.transitions.parent is None

    def has_one_constituent(self):
        # a length of 1 represents no constituents
        return len(self.constituents) == 2

    def num_transitions(self):
        # -1 for the sentinel value
        return len(self.transitions) - 1

    def finished(self, model):
        return self.empty_word_queue() and self.has_one_constituent() and model.get_top_constituent(self.constituents).label in model.get_root_labels()

    def get_tree(self, model):
        return model.get_top_constituent(self.constituents)

    def all_transitions(self, model):
        # TODO: rewrite this to be nicer / faster?  or just refactor?
        all_transitions = []
        transitions = self.transitions
        while transitions.parent is not None:
            all_transitions.append(model.get_top_transition(transitions))
            transitions = transitions.parent
        return list(reversed(all_transitions))

    def all_constituents(self, model):
        # TODO: rewrite this to be nicer / faster?
        all_constituents = []
        constituents = self.constituents
        while constituents.parent is not None:
            all_constituents.append(model.get_top_constituent(constituents))
            constituents = constituents.parent
        return list(reversed(all_constituents))

    def all_words(self, model):
        # TODO: rewrite this to be nicer / faster?
        all_words = []
        words = self.word_queue
        while words.parent is not None:
            all_words.append(model.get_top_word(words))
            words = words.parent
        return list(reversed(all_words))

    def to_string(self, model):
        return "State(\n  buffer:%s\n  transitions:%s\n  constituents:%s)" % (str(self.all_words(model)), str(self.all_transitions(model)), str(self.all_constituents(model)))

    def __str__(self):
        return "State(\n  buffer:%s\n  transitions:%s\n  constituents:%s)" % (str(self.word_queue), str(self.transitions), str(self.constituents))

def initial_state_from_preterminals(preterminal_lists, model, gold_trees):
    """
    what is passed in should be a list of list of preterminals
    """
    word_queues = model.initial_word_queues(preterminal_lists)
    states = [State(sentence_length=len(preterminal_list),
                    num_opens=0,
                    word_queue=wq,
                    transitions=model.initial_transitions(),
                    constituents=model.initial_constituents())
              for wq, preterminal_list in zip(word_queues, preterminal_lists)]
    if gold_trees is not None:
        for state, gold_tree in zip(states, gold_trees):
            state.gold_tree = gold_tree
    return states

def initial_state_from_words(word_lists, model):
    preterminal_lists = []
    for words in word_lists:
        preterminals = []
        for word, tag in reversed(words):
            word_node = Tree(label=word)
            tag_node = Tree(label=tag, children=[word_node])
            preterminals.append(tag_node)
        preterminal_lists.append(preterminals)
    return initial_state_from_preterminals(preterminal_lists, model, gold_trees=None)

def initial_state_from_gold_trees(trees, model):
    # reversed so we put the words on the stack backwards
    preterminal_lists = [[Tree(label=pt.label, children=Tree(label=pt.children[0].label))
                          for pt in tree.yield_reversed_preterminals()]
                         for tree in trees]
    return initial_state_from_preterminals(preterminal_lists, model, gold_trees=trees)

@functools.total_ordering
class Transition(ABC):
    """
    model is passed in as a dependency injection
    for example, an LSTM model can update hidden & output vectors when transitioning
    """
    @abstractmethod
    def update_state(self, state, model):
        """
        update the word queue, possibly remove old pieces from the constituents state, and return the new constituent

        the return value should be a tuple:
          updated word_queue
          updated constituents
          new constituent to put on the queue and None
            - note that the constituent shouldn't be on the queue yet
              that allows putting it on as a batch operation, which
              saves a significant amount of time in an LSTM, for example
          OR
          data used to make a new constituent and the method used
            - for example, CloseConstituent can return the children needed
              and itself.  this allows a batch operation to build
              the constituent
        """
        pass

    def delta_opens(self):
        return 0

    def apply(self, state, model):
        """
        return a new State transformed via this transition
        """
        word_queue, constituents, new_constituent, callback = self.update_state(state, model)
        if callback is not None:
            new_constituent = callback.build_constituents(model, [new_constituent])[0]
        constituents = model.push_constituents([constituents], [new_constituent])[0]

        return State(original_state=state,
                     num_opens=state.num_opens + self.delta_opens(),
                     word_queue=word_queue,
                     transitions=model.push_transitions([state.transitions], [self])[0],
                     constituents=constituents)

    @abstractmethod
    def is_legal(self, state, model):
        """
        assess whether or not this transition is legal in this state

        at parse time, the parser might choose a transition which cannot be made
        """
        pass

    def __lt__(self, other):
        # put the Shift at the front of a list, and otherwise sort alphabetically
        if self == other:
            return False
        if isinstance(self, Shift):
            return True
        if isinstance(other, Shift):
            return False
        return str(self) < str(other)

class Shift(Transition):
    def update_state(self, state, model):
        """
        This will handle all aspects of a shift transition

        - push the top element of the word queue onto constituents
        - pop the top element of the word queue
        """
        new_constituent = model.transform_word_to_constituent(state)
        word_queue = state.word_queue.pop()
        return word_queue, state.constituents, new_constituent, None

    def is_legal(self, state, model):
        """
        Disallow shifting when the word queue is empty or there are no opens to eventually eat this word
        """
        if state.num_opens == 0:
            return False
        if state.empty_word_queue():
            return False
        if state.num_opens == 1:
            # there must be at least one transition, since there is an open
            assert state.transitions.parent is not None
            if state.transitions.parent.parent is None:
                # only one transition
                trans = model.get_top_transition(state.transitions)
                # must be an Open, since there is one open and one transitions
                # note that an S, FRAG, etc could happen if we're using unary
                # and ROOT-S is possible in the case of compound Open
                # in both cases, Shift is legal
                # Note that the corresponding problem of shifting after the ROOT-S
                # has been closed to just ROOT is handled in CloseConstituent
                if len(trans.label) == 1 and trans.top_label in model.get_root_labels():
                    # don't shift a word at the very start of a parse
                    # we want there to be an extra layer below ROOT
                    return False
        return True

    def __repr__(self):
        return "Shift"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Shift):
            return True
        return False

    def __hash__(self):
        return hash(37)

class CompoundUnary(Transition):
    # TODO: run experiments to see if this is actually useful
    def __init__(self, labels):
        # the FIRST label will be the top of the tree
        # so CompoundUnary that results in root will have root as labels[0], for example
        if isinstance(labels, str):
            self.labels = (labels,)
        else:
            self.labels = tuple(labels)

    def update_state(self, state, model):
        # remove the top constituent
        # apply the labels
        # put the constituent back on the state
        constituents = state.constituents
        new_constituent = model.unary_transform(state.constituents, self.labels)
        constituents = constituents.pop()
        return state.word_queue, constituents, new_constituent, None

    def is_legal(self, state, model):
        """
        Disallow consecutive CompoundUnary transitions, force final transition to go to ROOT
        """
        # can't unary transition nothing
        if model.get_top_constituent(state.constituents) is None:
            return False
        # don't unary transition a dummy, dummy
        # and don't stack CompoundUnary transitions
        if isinstance(model.get_top_transition(state.transitions), (CompoundUnary, OpenConstituent)):
            return False
        is_root = self.labels[0] in model.get_root_labels()
        if not state.empty_word_queue() or not state.has_one_constituent():
            return not is_root
        else:
            return is_root

    def __repr__(self):
        return "CompoundUnary(%s)" % ",".join(self.labels)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, CompoundUnary):
            return False
        if self.labels == other.labels:
            return True
        return False

    def __hash__(self):
        return hash(self.labels)

class Dummy():
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return "Dummy({})".format(self.label)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Dummy):
            return False
        if self.label == other.label:
            return True
        return False

    def __hash__(self):
        return hash(self.label)

class OpenConstituent(Transition):
    def __init__(self, *label):
        self.label = tuple(label)
        self.top_label = self.label[0]

    def delta_opens(self):
        return 1

    def update_state(self, state, model):
        # open a new constituent which can later be closed
        # puts a DUMMY constituent on the stack to mark where the constituents end
        return state.word_queue, state.constituents, model.dummy_constituent(Dummy(self.label)), None

    def is_legal(self, state, model):
        """
        disallow based on the length of the sentence
        """
        if state.num_opens > state.sentence_length + 5:
            # fudge a bit so we don't miss root nodes etc in very small trees
            return False
        if state.empty_word_queue():
            return False
        if not model.has_unary_transitions():
            # TODO: maybe cache this value if this is an expensive operation
            is_root = self.top_label in model.get_root_labels()
            if is_root:
                return state.empty_transitions()
            else:
                return not state.empty_transitions()
        return True

    def __repr__(self):
        return "OpenConstituent({})".format(self.label)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, OpenConstituent):
            return False
        if self.label == other.label:
            return True
        return False

    def __hash__(self):
        return hash(self.label)

class CloseConstituent(Transition):
    def delta_opens(self):
        return -1

    def update_state(self, state, model):
        # pop constituents until we are done
        children = []
        constituents = state.constituents
        while not isinstance(model.get_top_constituent(constituents), Dummy):
            # keep the entire value from the stack - the model may need
            # the whole thing to transform the children into a new node
            children.append(constituents.value)
            constituents = constituents.pop()
        # the Dummy has the label on it
        label = model.get_top_constituent(constituents).label
        # pop past the Dummy as well
        constituents = constituents.pop()
        # the children are in the opposite order of what we expect
        children.reverse()

        return state.word_queue, constituents, (label, children), CloseConstituent

    @staticmethod
    def build_constituents(model, data):
        labels, children_lists = list(map(list, zip(*data)))
        new_constituents = model.build_constituents(labels, children_lists)
        return new_constituents


    def is_legal(self, state, model):
        """
        Disallow if the previous transition was the Open (nothing built yet)
        or if there is no Open on the stack yet
        """
        if isinstance(model.get_top_transition(state.transitions), OpenConstituent):
            return False
        if state.num_opens <= 0:
            return False
        if state.num_opens <= 1 and not state.empty_word_queue():
            # don't close the last open until all words have been used
            return False
        if not model.has_unary_transitions():
            # in fact, we have to leave the top level constituent
            # under the ROOT open if unary transitions are not possible
            # TODO: a compound open should be okay (in fact, this is a bug hurting performance)
            if state.num_opens == 2 and not state.empty_word_queue():
                return False
        return True

    def __repr__(self):
        return "CloseConstituent"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, CloseConstituent):
            return True
        return False

    def __hash__(self):
        return hash(93)

def bulk_apply(model, tree_batch, transitions, fail=False, max_transitions=1000):
    finished = []

    remove = set()

    word_queues = []
    constituents = []
    new_constituents = []
    callbacks = defaultdict(list)

    for idx, (tree, transition) in enumerate(zip(tree_batch, transitions)):
        if not transition:
            error = "Got stuck and couldn't find a legal transition on the following gold tree:\n{}\n\nFinal state:\n{}".format(tree.gold_tree, tree.to_string(model))
            if fail:
                raise ValueError(error)
            else:
                logger.error(error)
                remove.add(idx)
                continue

        if max_transitions and tree.num_transitions() >= max_transitions:
            # too many transitions
            # TODO: this error shouldn't use the gold_tree if it happens in a pipeline
            error = "Went infinite on the following gold tree:\n{}\n\nFinal state:\n{}".format(tree.gold_tree, tree.to_string(model))
            if fail:
                raise ValueError(error)
            else:
                logger.error(error)
                remove.add(idx)
                continue

        wq, c, nc, callback = transition.update_state(tree, model)

        word_queues.append(wq)
        constituents.append(c)
        new_constituents.append(nc)
        if callback:
            callbacks[callback].append(idx)

    for key, idxs in callbacks.items():
        data = [new_constituents[x] for x in idxs]
        callback_constituents = key.build_constituents(model, data)
        for idx, constituent in zip(idxs, callback_constituents):
            new_constituents[idx] = constituent

    tree_batch = [tree for idx, tree in enumerate(tree_batch) if idx not in remove]
    transitions = [trans for idx, trans in enumerate(transitions) if idx not in remove]

    new_transitions = model.push_transitions([tree.transitions for tree in tree_batch], transitions)
    new_constituents = model.push_constituents(constituents, new_constituents)

    tree_batch = [State(original_state=state,
                        num_opens=state.num_opens + transition.delta_opens(),
                        word_queue=word_queue,
                        transitions=transition_stack,
                        constituents=constituents)
                  for (state, transition, word_queue, transition_stack, constituents)
                  in zip(tree_batch, transitions, word_queues, new_transitions, new_constituents)]

    return tree_batch
