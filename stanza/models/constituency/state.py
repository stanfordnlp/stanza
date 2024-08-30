from collections import namedtuple

class State(namedtuple('State', ['word_queue', 'transitions', 'constituents', 'gold_tree', 'gold_sequence',
                                 'sentence_length', 'num_opens', 'word_position', 'score'])):
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

    The word_queue should have both a start and an end word.
    Those can be None in the case of the endpoints if they are unused.
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
        return self.constituents.length == 2

    @property
    def empty_constituents(self):
        return self.constituents.parent is None

    def num_constituents(self):
        return self.constituents.length - 1

    @property
    def num_transitions(self):
        # -1 for the sentinel value
        return self.transitions.length - 1

    def get_word(self, pos):
        # +1 to handle the initial sentinel value
        # (which you can actually get with pos=-1)
        return self.word_queue[pos+1]

    def finished(self, model):
        return self.empty_word_queue() and self.has_one_constituent() and model.get_top_constituent(self.constituents).label in model.root_labels

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

class MultiState(namedtuple('MultiState', ['states', 'gold_tree', 'gold_sequence', 'score'])):
    def finished(self, ensemble):
        return self.states[0].finished(ensemble.models[0])

    def get_tree(self, ensemble):
        return self.states[0].get_tree(ensemble.models[0])

    @property
    def empty_constituents(self):
        return self.states[0].empty_constituents

    def num_constituents(self):
        return len(self.states[0].constituents) - 1

    @property
    def num_transitions(self):
        # -1 for the sentinel value
        return len(self.states[0].transitions) - 1

    @property
    def num_opens(self):
        return self.states[0].num_opens

    @property
    def sentence_length(self):
        return self.states[0].sentence_length

    def empty_word_queue(self):
        return self.states[0].empty_word_queue()

    def empty_transitions(self):
        return self.states[0].empty_transitions()

    @property
    def constituents(self):
        # warning! if there is information in the constituents such as
        # the embedding of the constituent, this will only contain the
        # first such embedding
        # the other models' constituent states won't be returned
        return self.states[0].constituents

    @property
    def transitions(self):
        # warning! if there is information in the transitions such as
        # the embedding of the transition, this will only contain the
        # first such embedding
        # the other models' transition states won't be returned
        return self.states[0].transitions
