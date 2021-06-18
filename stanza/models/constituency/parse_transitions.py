from abc import ABC, abstractmethod

from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack

class State:
    def __init__(self, original_state=None, sentence_length=None, num_opens=None,
                 word_queue=None, transitions=None, constituents=None):
        """
        num_opens is useful for tracking
           1) if the parser is in a stuck state where it is making infinite opens
           2) if a close transition is impossible because there are no previous opens

        non-stack information such as sentence_length and num_opens
        will be copied from the original_state if possible, with the
        exact arguments overriding the values in the original_state
        """
        if word_queue is None:
            self.word_queue = TreeStack(value=None)
        else:
            self.word_queue = word_queue

        if transitions is None:
            self.transitions = TreeStack(value=None)
        else:
            self.transitions = transitions

        if constituents is None:
            self.constituents = TreeStack(value=None)
        else:
            self.constituents = constituents

        # copy non-stack information such as number of opens and sentence length
        if original_state is None:
            assert not sentence_length is None, "Must provide either an original_state or a sentence_length"
            assert not num_opens is None, "Must provide either an original_state or num_opens"
        else:
            self.sentence_length = original_state.sentence_length
            self.num_opens = original_state.num_opens

        if not num_opens is None:
            self.num_opens = num_opens

        if not sentence_length is None:
            self.sentence_length = sentence_length


    def empty_word_queue(self):
        # the first element of each stack is a sentinel with no value
        # and no parent
        return self.word_queue.parent is None

    def __str__(self):
        return "State(\n  buffer:%s\n  transitions:%s\n  constituents:%s)" % (str(self.word_queue), str(self.transitions), str(self.constituents))

def intiial_state_from_tagged_words(words, tags):
    word_queue = TreeStack(value=None)
    for word, tag in zip(reversed(words), reversed(tags)):
        word_node = Tree(label=word)
        tag_node = Tree(label=tag, children=[word_node])
        word_queue = word_queue.push(tag_node)
    return State(sentence_length=len(words), num_opens=0, word_queue=word_queue)

def initial_state_from_gold_tree(tree):
    word_queue = TreeStack(value=None)
    preterminals = [x for x in tree.yield_preterminals()]
    # put the words on the stack backwards
    preterminals.reverse()
    for pt in preterminals:
        word_node = Tree(label=pt.children[0].label)
        tag_node = Tree(label=pt.label, children=[word_node])
        word_queue = word_queue.push(tag_node)
    return State(sentence_length=len(preterminals), num_opens=0, word_queue=word_queue)

class Transition(ABC):
    @abstractmethod
    def apply(self, state):
        """
        return a new State transformed via this transition
        """
        pass

    @abstractmethod
    def is_legal(self, state):
        """
        assess whether or not this transition is legal in this state

        at parse time, the parser might choose a transition which cannot be made
        """
        pass

class Shift(Transition):
    def apply(self, state):
        # move the top word from the word queue to the constituency stack
        transitions = state.transitions.push(self)

        word = state.word_queue.value
        word_queue = state.word_queue.pop()

        constituents = state.constituents.push(word)

        return State(original_state=state,
                     word_queue=word_queue,
                     transitions=transitions,
                     constituents=constituents)

    def is_legal(self, state):
        """
        Disallow shifting when the word queue is empty
        """
        return not state.empty_word_queue()

    def __repr__(self):
        return "Shift"

class CompoundUnary(Transition):
    # TODO: run experiments to see if this is actually useful
    def __init__(self, labels):
        # the FIRST label will be the top of the tree
        # so CompoundUnary that results in root will have root as labels[0], for example
        self.labels = labels

    def apply(self, state):
        # remove the top constituent
        # apply the labels
        # put the constituent back on the state
        transitions = state.transitions.push(self)

        constituents = state.constituents
        last_constituent = constituents.value
        constituents = constituents.pop()
        for label in reversed(self.labels):
            last_constituent = Tree(label=label, children=[last_constituent])
        constituents = constituents.push(last_constituent)

        return State(original_state=state,
                     word_queue=state.word_queue,
                     transitions=transitions,
                     constituents=constituents)

    def is_legal(self, state):
        """
        Disallow consecutive CompoundUnary transitions
        """
        return not isinstance(state.transitions.value, CompoundUnary)

    def __repr__(self):
        return "CompoundUnary(%s)" % ",".join(self.labels)

class Dummy():
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return "Dummy(%s)" % self.label

class OpenConstituent(Transition):
    def __init__(self, label):
        self.label = label

    def apply(self, state):
        # open a new constituent which can later be closed
        # puts a DUMMY constituent on the stack to mark where the constituents end
        return State(original_state=state,
                     num_opens=state.num_opens+1,
                     word_queue=state.word_queue,
                     transitions=state.transitions.push(self),
                     constituents=state.constituents.push(Dummy(self.label)))

    def is_legal(self, state):
        """
        disallow based on the length of the sentence
        """
        if state.num_opens > state.sentence_length + 5:
            # fudge a bit so we don't miss root nodes etc in very small trees
            return False
        if state.empty_word_queue():
            return False
        return True

    def __repr__(self):
        return "OpenConstituent(%s)" % self.label

class CloseConstituent(Transition):
    def apply(self, state):
        # pop constituents until we are done
        children = []
        constituents = state.constituents
        while not isinstance(constituents.value, Dummy):
            children.append(constituents.value)
            constituents = constituents.pop()
        # the Dummy has the label on it
        label = constituents.value.label
        # pop past the Dummy as well
        constituents = constituents.pop()
        # the children are in the opposite order of what we expect
        children.reverse()
        new_constituent = Tree(label=label, children=children)
        constituents = constituents.push(new_constituent)

        return State(original_state=state,
                     num_opens=state.num_opens-1,
                     word_queue=state.word_queue,
                     transitions=state.transitions.push(self),
                     constituents=constituents)

    def is_legal(self, state):
        """
        Disallow if the previous transition was the Open (nothing built yet)
        or if there is no Open on the stack yet
        """
        if isinstance(state.transitions.value, OpenConstituent):
            return False
        if state.num_opens <= 0:
            return False
        return True

    def __repr__(self):
        return "CloseConstituent"
