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
from collections import defaultdict
import logging

import torch

from stanza.models.common import utils
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import transition_sequence
from stanza.models.constituency.parse_transitions import State, TransitionScheme, CloseConstituent
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.tree_stack import TreeStack
from stanza.server.parser_eval import ParseResult, ScoredTree

# default unary limit.  some treebanks may have longer chains (CTB, for example)
UNARY_LIMIT = 4

logger = logging.getLogger('stanza.constituency.trainer')

class BaseModel(ABC):
    """
    This base class defines abstract methods for manipulating a State.

    Applying transitions may change important metadata about a State
    such as the vectors associated with LSTM hidden states, for example.

    The constructor forwards all unused arguments to other classes in the
    constructor sequence, so put this before other classes such as nn.Module
    """
    def __init__(self, transition_scheme, unary_limit, reverse_sentence, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

        self._transition_scheme = transition_scheme
        self._unary_limit = unary_limit
        self._reverse_sentence = reverse_sentence

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
        return (self._transition_scheme is TransitionScheme.TOP_DOWN or
                self._transition_scheme is TransitionScheme.TOP_DOWN_UNARY or
                self._transition_scheme is TransitionScheme.TOP_DOWN_COMPOUND)

    def reverse_sentence(self):
        """
        Whether or not this model is built to parse backwards
        """
        return self._reverse_sentence

    def predict(self, states, is_legal=True):
        raise NotImplementedError("LSTMModel can predict, but SimpleModel cannot")

    def weighted_choice(self, states):
        raise NotImplementedError("LSTMModel can weighted_choice, but SimpleModel cannot")

    def predict_gold(self, states, is_legal=True):
        """
        For each State, return the next item in the gold_sequence
        """
        transitions = [y.gold_sequence[y.num_transitions()] for y in states]
        if is_legal:
            for trans, state in zip(transitions, states):
                if not trans.is_legal(state, self):
                    raise RuntimeError("Transition {}:{} was not legal in a transition sequence:\nOriginal tree: {}\nTransitions: {}".format(state.num_transitions(), trans, state.gold_tree, state.gold_sequence))
        return None, transitions, None

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
                        word_position=0,
                        score=0.0)
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

    def build_batch_from_trees(self, batch_size, data_iterator):
        """
        Read from the data_iterator batch_size trees and turn them into new parsing states
        """
        state_batch = []
        for _ in range(batch_size):
            gold_tree = next(data_iterator, None)
            if gold_tree is None:
                break
            state_batch.append(gold_tree)

        if len(state_batch) > 0:
            state_batch = self.initial_state_from_gold_trees(state_batch)
        return state_batch

    def build_batch_from_trees_with_gold_sequence(self, batch_size, data_iterator):
        """
        Same as build_batch_from_trees, but use the model parameters to turn the trees into gold sequences and include the sequence
        """
        state_batch = self.build_batch_from_trees(batch_size, data_iterator)
        if len(state_batch) == 0:
            return state_batch

        gold_sequences = transition_sequence.build_treebank([state.gold_tree for state in state_batch], self.transition_scheme(), self.reverse_sentence())
        state_batch = [state._replace(gold_sequence=sequence) for state, sequence in zip(state_batch, gold_sequences)]
        return state_batch

    def build_batch_from_tagged_words(self, batch_size, data_iterator):
        """
        Read from the data_iterator batch_size tagged sentences and turn them into new parsing states

        Expects a list of list of (word, tag)
        """
        state_batch = []
        for _ in range(batch_size):
            sentence = next(data_iterator, None)
            if sentence is None:
                break
            state_batch.append(sentence)

        if len(state_batch) > 0:
            state_batch = self.initial_state_from_words(state_batch)
        return state_batch


    def parse_sentences(self, data_iterator, build_batch_fn, batch_size, transition_choice, keep_state=False, keep_constituents=False, keep_scores=False):
        """
        Repeat transitions to build a list of trees from the input batches.

        The data_iterator should be anything which returns the data for a parse task via next()
        build_batch_fn is a function that turns that data into State objects
        This will be called to generate batches of size batch_size until the data is exhausted

        The return is a list of tuples: (gold_tree, [(predicted, score) ...])
        gold_tree will be left blank if the data did not include gold trees
        if keep_scores is true, the score will be the sum of the values
          returned by the model for each transition

        transition_choice: which method of the model to use for choosing the next transition
          predict for predicting the transition based on the model
          predict_gold to just extract the gold transition from the sequence
        """
        treebank = []
        treebank_indices = []
        state_batch = build_batch_fn(batch_size, data_iterator)
        # used to track which indices we are currently parsing
        # since the parses get finished at different times, this will let us unsort after
        batch_indices = list(range(len(state_batch)))
        horizon_iterator = iter([])

        if keep_constituents:
            constituents = defaultdict(list)

        while len(state_batch) > 0:
            pred_scores, transitions, scores = transition_choice(state_batch)
            if keep_scores and scores is not None:
                state_batch = [state._replace(score=state.score + score) for state, score in zip(state_batch, scores)]
            state_batch = parse_transitions.bulk_apply(self, state_batch, transitions)

            if keep_constituents:
                for t_idx, transition in enumerate(transitions):
                    if isinstance(transition, CloseConstituent):
                        # constituents is a TreeStack with information on how to build the next state of the LSTM or attn
                        # constituents.value is the TreeStack node
                        # constituents.value.value is the Constituent itself (with the tree and the embedding)
                        constituents[batch_indices[t_idx]].append(state_batch[t_idx].constituents.value.value)

            remove = set()
            for idx, state in enumerate(state_batch):
                if state.finished(self):
                    predicted_tree = state.get_tree(self)
                    if self.reverse_sentence():
                        predicted_tree = predicted_tree.reverse()
                    gold_tree = state.gold_tree
                    treebank.append(ParseResult(gold_tree, [ScoredTree(predicted_tree, state.score)], state if keep_state else None, constituents[batch_indices[idx]] if keep_constituents else None))
                    treebank_indices.append(batch_indices[idx])
                    remove.add(idx)

            if len(remove) > 0:
                state_batch = [state for idx, state in enumerate(state_batch) if idx not in remove]
                batch_indices = [batch_idx for idx, batch_idx in enumerate(batch_indices) if idx not in remove]

            for _ in range(batch_size - len(state_batch)):
                horizon_state = next(horizon_iterator, None)
                if not horizon_state:
                    horizon_batch = build_batch_fn(batch_size, data_iterator)
                    if len(horizon_batch) == 0:
                        break
                    horizon_iterator = iter(horizon_batch)
                    horizon_state = next(horizon_iterator, None)

                state_batch.append(horizon_state)
                batch_indices.append(len(treebank) + len(state_batch))

        treebank = utils.unsort(treebank, treebank_indices)
        return treebank

    def parse_sentences_no_grad(self, data_iterator, build_batch_fn, batch_size, transition_choice, keep_state=False, keep_constituents=False, keep_scores=False):
        """
        Given an iterator over the data and a method for building batches, returns a list of parse trees.

        no_grad() is so that gradients aren't kept, which makes the model
        run faster and use less memory at inference time
        """
        with torch.no_grad():
            return self.parse_sentences(data_iterator, build_batch_fn, batch_size, transition_choice, keep_state, keep_constituents, keep_scores)

    def analyze_trees(self, trees, batch_size=None, keep_state=True, keep_constituents=True, keep_scores=True):
        """
        Return a ParseResult for each tree in the trees list

        The transitions run will be the transitions represented by the tree
        The output layers will be available in result.state for each result

        keep_state=True as a default here as a method which keeps the grad
        is likely to want to keep the resulting state as well
        """
        if batch_size is None:
            # TODO: refactor?
            batch_size = self.args['eval_batch_size']
        tree_iterator = iter(trees)
        treebank = self.parse_sentences(tree_iterator, self.build_batch_from_trees_with_gold_sequence, batch_size, self.predict_gold, keep_state, keep_constituents, keep_scores=keep_scores)
        return treebank

    def parse_tagged_words(self, words, batch_size):
        """
        This parses tagged words and returns a list of trees.

        `parse_tagged_words` is useful at Pipeline time -
          it takes words & tags and processes that into trees.

        The tagged words should be represented:
          one list per sentence
            each sentence is a list of (word, tag)
        The return value is a list of ParseTree objects
        """
        logger.debug("Processing %d sentences", len(words))
        self.eval()

        sentence_iterator = iter(words)
        treebank = self.parse_sentences_no_grad(sentence_iterator, self.build_batch_from_tagged_words, batch_size, self.predict, keep_state=False, keep_constituents=False)

        results = [t.predictions[0].tree for t in treebank]
        return results

class SimpleModel(BaseModel):
    """
    This model allows pushing and popping with no extra data

    This class is primarily used for testing various operations which
    don't need the NN's weights

    Also, for rebuilding trees from transitions when verifying the
    transitions in situations where the NN state is not relevant,
    as this class will be faster than using the NN
    """
    def __init__(self, transition_scheme=TransitionScheme.TOP_DOWN_UNARY, unary_limit=UNARY_LIMIT, reverse_sentence=False):
        super().__init__(transition_scheme=transition_scheme, unary_limit=unary_limit, reverse_sentence=reverse_sentence)

    def initial_word_queues(self, tagged_word_lists):
        word_queues = []
        for tagged_words in tagged_word_lists:
            word_queue =  [None]
            word_queue += [tag_node for tag_node in tagged_words]
            word_queue.append(None)
            if self.reverse_sentence():
                word_queue.reverse()
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
