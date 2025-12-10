"""
Represents a parsing state for a transition based dependency parser.

When created from a gold sentence, there will be a networkx graph that
represents the completed graph using indices for the words, 0 for root.
"""

from collections import defaultdict
from collections import namedtuple

import networkx as nx

from stanza.models.depparse.transition.transitions import ProjectiveRight, NonprojectiveRight, ProjectiveLeft, NonprojectiveLeft, Shift, Finalize

TransitionLSTMEmbedding = namedtuple('TransitionLSTMEmbedding', 'h0 c0')
SubtreeLSTMEmbedding = namedtuple('SubtreeLSTMEmbedding', 'h0 c0')

# transitions and parsed_graph represent the current state of a parse
# gold_graph and gold_sequence are gold, if that information exists
# current_heads is a list of the word IDs for the heads of the subtrees
# transition_lstm_embeddings is a list of the above TransitionLSTMEmbedding namedtuple - one per transition
State = namedtuple('State', ['transitions', 'parsed_graph', 'word_position', 'num_words', 'current_heads',
                             'gold_graph', 'gold_sequence', 'word_embeddings', 'subtree_embeddings',
                             'transition_lstm_embeddings', 'subtree_lstm_embeddings'])

def is_nonproj(gold_graph, node, pred):
    for middle in range(node+1, pred):
        if any(x < node or x > pred for x in gold_graph.successors(middle)):
            return True
        if any(x < node or x > pred for x in gold_graph.predecessors(middle)):
            return True
    return False

def build_gold_sequence(gold_graph):
    num_words = len(gold_graph.nodes()) - 1
    state = State([], nx.DiGraph(), 0, num_words, [], None, None, None, None, [], [])

    # determine which arcs are non-projective
    # key is the head, value is a set of the children which are non-proj
    nonproj_right = defaultdict(set)
    for node in gold_graph.nodes():
        # predecessors = children of node
        for pred in gold_graph.predecessors(node):
            if node < pred:
                # this child has a higher idx
                if is_nonproj(gold_graph, node, pred):
                    nonproj_right[node].add(pred)

    #print("NONPROJ", nonproj_right)

    proj_right_children = defaultdict(set)
    for node in gold_graph.nodes():
        for pred in gold_graph.predecessors(node):
            if node < pred:
                if node not in nonproj_right or pred not in nonproj_right[node]:
                    proj_right_children[node].add(pred)
                    #print(nonproj_right[node], node, pred)

    while state.word_position < state.num_words or len(state.current_heads) > 1:
        #print("Examining state at word queue %d with heads %s" % (state.word_position, state.current_heads))
        if len(state.current_heads) >= 2:
            # -1 is the head of -2 is the projective right attachment
            # (in RtL languages)
            if gold_graph.has_edge(state.current_heads[-2], state.current_heads[-1]):
                #print("Right attach %d to %d" % (state.current_heads[-2], state.current_heads[-1]))
                transition = ProjectiveRight(gold_graph.get_edge_data(state.current_heads[-2], state.current_heads[-1])['deprel'])
                state = transition.apply(state)
                continue
        if len(state.current_heads) >= 2:
            # check for non-projective right transitions
            found = False
            for head_idx, head in enumerate(state.current_heads[:-2]):
                if gold_graph.has_edge(head, state.current_heads[-1]):
                    #print("Right nonproj attach %d to %d" % (head, state.current_heads[-1]))
                    transition = NonprojectiveRight(gold_graph.get_edge_data(head, state.current_heads[-1])['deprel'], head)
                    state = transition.apply(state)
                    found = True
                    break
            if found:
                continue
        if len(state.current_heads) >= 2:
            # this represents that -2 is the head of -1, the projective left attachment
            # only attach if there are no projective children remaining for the right node
            if gold_graph.has_edge(state.current_heads[-1], state.current_heads[-2]) and len(proj_right_children[state.current_heads[-1]]) == 0:
                #print("Left attach %d to %d" % (state.current_heads[-1], state.current_heads[-2]))
                # discard instead of remove in case it was originally judged to be non-projective,
                # but other non-projective transitions make it now appear projective
                proj_right_children[state.current_heads[-2]].discard(state.current_heads[-1])
                transition = ProjectiveLeft(gold_graph.get_edge_data(state.current_heads[-1], state.current_heads[-2])['deprel'])
                state = transition.apply(state)
                continue
        if len(state.current_heads) >= 2 and len(proj_right_children[state.current_heads[-1]]) == 0:
            # check for nonprojective left
            successors = gold_graph.successors(state.current_heads[-1])
            found = False
            for successor in successors:
                # successor > 0: must not try to NonprojectiveLeft the root
                if successor > 0 and successor < state.current_heads[-1]:
                    # this qualifies as a node that can attach non-proj to the left
                    #print("Non-proj left attach %d to %d" % (state.current_heads[-1], successor))
                    transition = NonprojectiveLeft(gold_graph.get_edge_data(state.current_heads[-1], successor)['deprel'], successor)
                    state = transition.apply(state)
                    found = True
                    break
            if found:
                continue
        if state.word_position < state.num_words:
            #print("SHIFT")
            transition = Shift()
            state = transition.apply(state)
            continue
        #print(proj_right_children)
        raise AssertionError("Couldn't find transition: position %d, heads %s, transitions %s" % (state.word_position, state.current_heads, state.transitions))

    transition = Finalize()
    state = transition.apply(state)

    return state.transitions

def state_from_graph(gold_graph):
    transitions = []
    empty_graph = nx.DiGraph()

    gold_sequence = build_gold_sequence(gold_graph)
    num_words = len(gold_graph.nodes()) - 1
    return State(transitions, empty_graph, 0, num_words, [], gold_graph, gold_sequence, None, None, [], [])

def from_gold(sentence):
    gold_graph = nx.DiGraph()
    for word_idx, word in enumerate(sentence.words):
        # +1 as the nodes are indexed from 1, with 0 as the root
        gold_graph.add_edge(word_idx+1, word.head, deprel=word.deprel)
    return state_from_graph(gold_graph)

def states_from_heads(heads, deprels, texts, sentlens):
    """
    the head and text as passed around by the dependency parser training.
    deprel should be the actual names, perhaps using vocab.unmap
    sentlens should be the exact length - the data module adds 1, so subtract that first
    """
    states = []
    for head, deprel, text, sentlen in zip(heads, deprels, texts, sentlens):
        gold_graph = nx.DiGraph()
        for word_idx in range(sentlen):
            gold_graph.add_edge(word_idx+1, head[word_idx].item(), deprel=deprel[word_idx])
        try:
            states.append(state_from_graph(gold_graph))
        except ValueError as e:
            raise ValueError("Found an error building a sequence from:\n%s\n%s\n%s" % (text, head, deprel)) from e
    return states

def states_from_data_batch(vocab, heads, deprels, texts, sentlens):
    sentlens = [x-1 for x in sentlens]
    deprels = [vocab.unmap(deps) for deps in deprels]
    states = states_from_heads(heads, deprels, texts, sentlens)
    return states

def state_from_text(text):
    """ text should be a list of words """
    transitions = []
    num_words = len(text)
    empty_graph = nx.DiGraph()
    return State(transitions, empty_graph, 0, num_words, [], None, None, None, None, [], [])
