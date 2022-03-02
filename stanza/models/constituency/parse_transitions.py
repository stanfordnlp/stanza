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

logger = logging.getLogger('stanza')

class TransitionScheme(Enum):
    TOP_DOWN           = 1
    TOP_DOWN_COMPOUND  = 2
    TOP_DOWN_UNARY     = 3

    IN_ORDER           = 4

class State(namedtuple('State', ['word_queue', 'transitions', 'constituents', 'gold_tree', 'gold_sequence',
                                 'sentence_length', 'num_opens', 'word_position'])):
    """
    Represents a partially completed transition parse

    Includes stack/buffers for unused words, already executed transitions, and partially build constituents
    At training time, also keeps track of the gold data we are reparsing

    num_opens is useful for tracking
       1) if the parser is in a stuck state where it is making infinite opens
       2) if a close transition is impossible because there are no previous opens

    sentence_length tracks how long the sentence is so we abort if we go infinite

    non-stack information such as sentence_length and num_opens
    will be copied from the original_state if possible, with the
    exact arguments overriding the values in the original_state

    gold_tree: the original tree, if made from a gold tree.  might be None
    gold_sequence: the original transition sequence, if available
    Note that at runtime, gold values will not be available

    word_position tracks where in the word queue we are.  cheaper than
      manipulating the list itself.  this can be handled differently
      from transitions and constituents as it is processed once
      at the start of parsing
    """
    def empty_word_queue(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.word_position == self.sentence_length

    def empty_transitions(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.transitions.parent is None

    def has_one_constituent(self):
        # a length of 1 represents no constituents
        return len(self.constituents) == 2

    def num_constituents(self):
        return len(self.constituents) - 1

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
        return [model.get_word(x) for x in self.word_queue]

    def to_string(self, model):
        return "State(\n  buffer:%s\n  transitions:%s\n  constituents:%s\n  word_position:%d num_opens:%d)" % (str(self.all_words(model)), str(self.all_transitions(model)), str(self.all_constituents(model)), self.word_position, self.num_opens)

    def __str__(self):
        return "State(\n  buffer:%s\n  transitions:%s\n  constituents:%s)" % (str(self.word_queue), str(self.transitions), str(self.constituents))

def initial_state_from_preterminals(preterminal_lists, model, gold_trees):
    """
    what is passed in should be a list of list of preterminals
    """
    word_queues = model.initial_word_queues(preterminal_lists)
    # this is the bottom of the TreeStack and will be the same for each State
    transitions=model.initial_transitions()
    constituents=model.initial_constituents()
    states = [State(sentence_length=len(wq)-1,   # -1 because it ends with a sentinel
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

def initial_state_from_words(word_lists, model):
    preterminal_lists = [[Tree(tag, Tree(word)) for word, tag in words]
                         for words in word_lists]
    return initial_state_from_preterminals(preterminal_lists, model, gold_trees=None)

def initial_state_from_gold_trees(trees, model):
    preterminal_lists = [[Tree(pt.label, Tree(pt.children[0].label))
                          for pt in tree.yield_preterminals()]
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
        update the word queue position, possibly remove old pieces from the constituents state, and return the new constituent

        the return value should be a tuple:
          updated word_position
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

    def delta_opens(self):
        return 0

    def apply(self, state, model):
        """
        return a new State transformed via this transition

        convenience method to call bulk_apply, which is significantly
        faster than single operations for an NN based model
        """
        update = bulk_apply(model, [state], [self])
        return update[0]

    @abstractmethod
    def is_legal(self, state, model):
        """
        assess whether or not this transition is legal in this state

        at parse time, the parser might choose a transition which cannot be made
        """

    @abstractmethod
    def short_name(self):
        """
        A short name to identify this transition
        """

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
        return state.word_position+1, state.constituents, new_constituent, None

    def is_legal(self, state, model):
        """
        Disallow shifting when the word queue is empty or there are no opens to eventually eat this word
        """
        if state.empty_word_queue():
            return False
        if model.is_top_down():
            # top down transition sequences cannot shift if there are currently no
            # Open transitions on the stack.  in such a case, the new constituent
            # will never be reduced
            if state.num_opens == 0:
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
        else:
            # in-order k==1 (the only other option currently)
            # can shift ONCE, but note that there is no way to consume
            # two items in a row if there is no Open on the stack.
            # As long as there is one or more open transitions,
            # everything can be eaten
            if state.num_opens == 0:
                if state.num_constituents() > 0:
                    return False
        return True

    def short_name(self):
        return "Shift"

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
        return state.word_position, constituents, new_constituent, None

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

    def short_name(self):
        return "Unary"

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
    """
    Takes a space on the constituent stack to represent where an Open transition occurred
    """
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

def too_many_unary_nodes(tree, unary_limit):
    """
    Return True iff there are UNARY_LIMIT unary nodes in a tree in a row

    helps prevent infinite open/close patterns
    otherwise, the model can get stuck in essentially an infinite loop
    """
    if tree is None:
        return False
    for _ in range(unary_limit + 1):
        if len(tree.children) != 1:
            return False
        tree = tree.children[0]
    return True

class OpenConstituent(Transition):
    def __init__(self, *label):
        self.label = tuple(label)
        self.top_label = self.label[0]

    def delta_opens(self):
        return 1

    def update_state(self, state, model):
        # open a new constituent which can later be closed
        # puts a DUMMY constituent on the stack to mark where the constituents end
        return state.word_position, state.constituents, model.dummy_constituent(Dummy(self.label)), None

    def is_legal(self, state, model):
        """
        disallow based on the length of the sentence
        """
        if state.num_opens > state.sentence_length + 5:
            # fudge a bit so we don't miss root nodes etc in very small trees
            return False
        if model.is_top_down():
            # If the model is top down, you can't Open if there are
            # no word to eventually eat
            if state.empty_word_queue():
                return False
            # Also, you can only Open a ROOT iff it is at the root position
            # The assumption in the unary scheme is there will be no
            # root open transitions
            if not model.has_unary_transitions():
                # TODO: maybe cache this value if this is an expensive operation
                is_root = self.top_label in model.get_root_labels()
                if is_root:
                    return state.empty_transitions()
                else:
                    return not state.empty_transitions()
        else:
            # in-order nodes can Open as long as there is at least one thing
            # on the constituency stack
            # since closing the in-order involves removing one more
            # item before the open, and it can close at any time
            # (a close immediately after the open represents a unary)
            if state.num_constituents() == 0:
                return False
            if isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                # consecutive Opens don't make sense in the context of in-order
                return False
            # one other restriction - we assume all parse trees
            # start with (ROOT (first_real_con ...))
            # therefore ROOT can only occur via Open after everything
            # else has been pushed and processed
            # there are no further restrictions
            is_root = self.top_label in model.get_root_labels()
            if is_root:
                # can't make a root node if it will be in the middle of the parse
                # can't make a root node if there's still words to eat
                # note that the second assumption wouldn't work,
                # except we are assuming there will never be multiple
                # nodes under one root
                return state.num_opens == 0 and state.empty_word_queue()
            else:
                if (state.num_opens > 0 or state.empty_word_queue()) and too_many_unary_nodes(model.get_top_constituent(state.constituents), model.unary_limit()):
                    # looks like we've been in a loop of lots of unary transitions
                    # note that we check `num_opens > 0` because otherwise we might wind up stuck
                    # in a state where the only legal transition is open, such as if the
                    # constituent stack is otherwise empty, but the open is illegal because
                    # it causes too many unaries
                    # in such a case we can forbid the corresponding close instead...
                    # if empty_word_queue, that means it is trying to make infinitiely many
                    # non-ROOT Open transitions instead of just transitioning ROOT
                    return False
                return True
        return True

    def short_name(self):
        return "Open"

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
        if not model.is_top_down():
            # the alternative to TOP_DOWN_... is IN_ORDER
            # in which case we want to pop one more constituent
            children.append(constituents.value)
            constituents = constituents.pop()
        # the children are in the opposite order of what we expect
        children.reverse()

        return state.word_position, constituents, (label, children), CloseConstituent

    @staticmethod
    def build_constituents(model, data):
        """
        builds new constituents out of the incoming data

        data is a list of tuples: (label, children)
        the model will batch the build operation
        again, the purpose of this batching is to do multiple deep learning operations at once
        """
        labels, children_lists = map(list, zip(*data))
        new_constituents = model.build_constituents(labels, children_lists)
        return new_constituents


    def is_legal(self, state, model):
        """
        Disallow if there is no Open on the stack yet
        in TOP_DOWN, if the previous transition was the Open (nothing built yet)
        in IN_ORDER, previous transition does not matter, except for one small corner case
        """
        if state.num_opens <= 0:
            return False
        if model.is_top_down():
            if isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                return False
            if state.num_opens <= 1 and not state.empty_word_queue():
                # don't close the last open until all words have been used
                return False
            if model.transition_scheme() == TransitionScheme.TOP_DOWN_COMPOUND:
                # when doing TOP_DOWN_COMPOUND, we assume all transitions
                # at the ROOT level have an S, SQ, FRAG, etc underneath
                # this is checked when the model is first trained
                if state.num_opens == 1 and not state.empty_word_queue():
                    return False
            elif not model.has_unary_transitions():
                # in fact, we have to leave the top level constituent
                # under the ROOT open if unary transitions are not possible
                if state.num_opens == 2 and not state.empty_word_queue():
                    return False
        else:
            if not isinstance(model.get_top_transition(state.transitions), OpenConstituent):
                # we're not stuck in a loop of unaries
                return True
            if state.num_opens > 1 or state.empty_word_queue():
                # in either of these cases, the corresponding Open should be eliminated
                # if we're stuck in a loop of unaries
                return True
            node = model.get_top_constituent(state.constituents.pop())
            if too_many_unary_nodes(node, model.unary_limit()):
                # at this point, we are in a situation where
                # - multiple unaries have happened in a row
                # - there is stuff on the word_queue, so a ROOT open isn't legal
                # - there's only one constituent on the stack, so the only legal
                #   option once there are no opens left will be an open
                # this means we'll be stuck having to open again if we do close
                # this node, so instead we make the Close illegal
                return False
        return True

    def short_name(self):
        return "Close"

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

def bulk_apply(model, tree_batch, transitions, fail=False):
    """
    Apply the given list of Transitions to the given list of States, using the model as a reference

    model: SimpleModel, LSTMModel, or any other form of model
    tree_batch: list of States (legacy name)
    transitions: list of transitions
    fail: throw an exception on a failed transition, as opposed to skipping the tree
    """
    remove = set()

    word_positions = []
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

        if tree.num_transitions() >= len(tree.word_queue) * 20:
            # too many transitions
            # x20 is somewhat empirically chosen based on certain
            # treebanks having deep unary structures, especially early
            # on when the model is fumbling around
            if tree.gold_tree:
                error = "Went infinite on the following gold tree:\n{}\n\nFinal state:\n{}".format(tree.gold_tree, tree.to_string(model))
            else:
                error = "Went infinite!:\nFinal state:\n{}".format(tree.to_string(model))
            if fail:
                raise ValueError(error)
            else:
                logger.error(error)
                remove.add(idx)
                continue

        wq, c, nc, callback = transition.update_state(tree, model)

        word_positions.append(wq)
        constituents.append(c)
        new_constituents.append(nc)
        if callback:
            # not `idx` in case something was removed
            callbacks[callback].append(len(new_constituents)-1)

    for key, idxs in callbacks.items():
        data = [new_constituents[x] for x in idxs]
        callback_constituents = key.build_constituents(model, data)
        for idx, constituent in zip(idxs, callback_constituents):
            new_constituents[idx] = constituent

    tree_batch = [tree for idx, tree in enumerate(tree_batch) if idx not in remove]
    transitions = [trans for idx, trans in enumerate(transitions) if idx not in remove]

    if len(tree_batch) == 0:
        return tree_batch

    new_transitions = model.push_transitions([tree.transitions for tree in tree_batch], transitions)
    new_constituents = model.push_constituents(constituents, new_constituents)

    tree_batch = [state._replace(num_opens=state.num_opens + transition.delta_opens(),
                                 word_position=word_position,
                                 transitions=transition_stack,
                                 constituents=constituents)
                  for (state, transition, word_position, transition_stack, constituents)
                  in zip(tree_batch, transitions, word_positions, new_transitions, new_constituents)]

    return tree_batch
