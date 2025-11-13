"""An implementation of a transition system for non-projective dependencies.

The implementation is actually quite straightforward:

Current partially built trees are kept on a stack

Words not yet processed are kept on a queue

If the root of one of the subtrees on the stack has for its parent the
head of the stack, that can be done with either an immediate
connection or with a connection from another head earlier in the stack
to the top of the stack.  The latter will be done with attention.

If the word on top of the stack has no later projective children, and
it has its parent to the left, that can be done as either an immediate
head-to-head connection or by skipping multiple spaces into the stack.
Skipping will be done with an attention mechanism, allowing for
attachment at any depth (not just the subtree roots).  Disallowing
projective children to the right reduces ambiguity and reduces the
likelihood that a later tree needs to attach to a deep subtree.

If neither of these work, there should be leftover words in the word queue,
so a Shift operation moves that word to the partially built graph as a
single word tree.

The first thing to note (a lemma), when considering a particular top
of the stack k0, it is not possible for there to be a word earlier in the stack
k1 with a head between k1 and k0, eg k1 < k2 < k0.  Otherwise, at the time
k2 was the head of the stack, attaching k1 to k2 would have been a legal
transition and Shift would not be an option yet.

TODO: finish proof or reconsider the attachment restrictions

One thing to note is that this transition sequence does not enforce uniqueness.
example:
  I saw the man yesterday with a hat
attach "hat" to "man" before attaching "with" and "a"
now attaching "with" and "a" will be guaranteed wrong unless there are no uniqueness restrictions
so, restricting to uniqueness will make things worse after making a mistake
"""


class Shift():
    def __repr__(self):
        return "Shift"

    def apply(self, state):
        # words are indexed from 1 in the graph, so we add +1 to the position first before shifting
        state = state._replace(word_position = state.word_position+1)
        state.current_heads.append(state.word_position)
        state.transitions.append(self)
        return state

    def is_legal(self, state):
        if state.word_position < state.num_words:
            return True
        return False

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Shift):
            return True
        return False

    def simplify(self):
        return self

    def __hash__(self):
        return hash(99)

class ProjectiveRight():
    def __init__(self, deprel):
        self.deprel = deprel

    def apply(self, state):
        state.parsed_graph.add_edge(state.current_heads[-2], state.current_heads[-1], deprel=self.deprel)
        state.transitions.append(self)
        state.current_heads.pop(-2)
        state = state._replace(subtree_lstm_embeddings=state.subtree_lstm_embeddings[:-2])
        return state

    def is_legal(self, state):
        if len(state.current_heads) >= 2:
            return True
        return False

    def __repr__(self):
        return "PR(%s)" % self.deprel

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ProjectiveRight):
            return False
        if other.deprel == self.deprel:
            return True
        return False

    def simplify(self):
        return self

    def __hash__(self):
        return hash((self.deprel, 15))

class NonprojectiveRight():
    def __init__(self, deprel, word_idx):
        # TODO: would it make more sense to have this store the number
        # of heads rather than the word idx?
        self.deprel = deprel
        self.word_idx = word_idx

    def apply(self, state):
        state.parsed_graph.add_edge(self.word_idx, state.current_heads[-1], deprel=self.deprel)
        state.transitions.append(self)
        # this is only legal to apply when the node being merged is a subtree head
        head_idx = state.current_heads.index(self.word_idx)
        state.current_heads.pop(head_idx)
        if head_idx == 0:
            state = state._replace(subtree_lstm_embeddings=[])
        else:
            state = state._replace(subtree_lstm_embeddings=state.subtree_lstm_embeddings[:head_idx-1])
        return state

    def is_legal(self, state):
        if len(state.current_heads) < 2:
            return False
        if state.current_heads[-1] == self.word_idx:
            return False
        return True

    def __repr__(self):
        if self.word_idx is None:
            return "NPR(%s)" % self.deprel
        return "NPR(%s, %d)" % (self.deprel, self.word_idx)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, NonprojectiveRight):
            return False
        if other.deprel == self.deprel and other.word_idx == self.word_idx:
            return True
        return False

    def simplify(self):
        return NonprojectiveRight(self.deprel, None)

    def __hash__(self):
        return hash((self.deprel, 25))

class ProjectiveLeft():
    def __init__(self, deprel):
        self.deprel = deprel

    def apply(self, state):
        state.parsed_graph.add_edge(state.current_heads[-1], state.current_heads[-2], deprel=self.deprel)
        state.transitions.append(self)
        state.current_heads.pop(-1)
        state = state._replace(subtree_lstm_embeddings=state.subtree_lstm_embeddings[:-2])
        return state

    def is_legal(self, state):
        if len(state.current_heads) >= 2:
            return True
        return False

    def __repr__(self):
        return "PL(%s)" % self.deprel

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ProjectiveLeft):
            return False
        if other.deprel == self.deprel:
            return True
        return False

    def simplify(self):
        return self

    def __hash__(self):
        return hash((self.deprel, 55))

class NonprojectiveLeft():
    def __init__(self, deprel, word_idx):
        self.deprel = deprel
        self.word_idx = word_idx

    def apply(self, state):
        state.parsed_graph.add_edge(state.current_heads[-1], self.word_idx, deprel=self.deprel)
        state.transitions.append(self)
        state.current_heads.pop(-1)

        word_idx = self.word_idx
        # out_degree is head, which should be either 0 (no head) or 1 (has a head)
        # checked is a sanity check for if someone feeds in a circular graph
        checked = 0
        while state.parsed_graph.out_degree(word_idx) > 0:
            checked += 1
            if checked > state.num_words:
                raise AssertionError("Found a circular graph!")
            for successor in state.parsed_graph.successors(word_idx):
                word_idx = successor
                # there should only be one
                break
        head_idx = state.current_heads.index(word_idx)
        if head_idx == 0:
            state = state._replace(subtree_lstm_embeddings=[])
        else:
            state = state._replace(subtree_lstm_embeddings=state.subtree_lstm_embeddings[:head_idx-1])
        return state

    def is_legal(self, state):
        if len(state.current_heads) < 2:
            return False
        word_idx = self.word_idx
        if state.current_heads[-1] == word_idx:
            return False
        # this can happen if a word is shifted but no edges are built yet
        # we are not necessarily adding shifted words to the graph
        if word_idx not in state.parsed_graph:
            return True
        # out_degree is head, which should be either 0 (no head) or 1 (has a head)
        # checked is a sanity check for if someone feeds in a circular graph
        checked = 0
        while state.parsed_graph.out_degree(word_idx) > 0:
            checked += 1
            if checked > state.num_words:
                return False
            for successor in state.parsed_graph.successors(word_idx):
                word_idx = successor
                if state.current_heads[-1] == word_idx:
                    return False
                # there should only be one
                break
        return True

    def __repr__(self):
        if self.word_idx is None:
            return "NPL(%s)" % self.deprel
        return "NPL(%s, %d)" % (self.deprel, self.word_idx)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, NonprojectiveLeft):
            return False
        if other.deprel == self.deprel and other.word_idx == self.word_idx:
            return True
        return False

    def simplify(self):
        return NonprojectiveLeft(self.deprel, None)

    def __hash__(self):
        return hash((self.deprel, 65))

# TODO: do we actually need this?
class Finalize():
    def apply(self, state):
        state.parsed_graph.add_edge(state.current_heads[-1], 0, deprel="root")
        state.transitions.append(self)
        state.current_heads.pop(-1)
        return state

    def is_legal(self, state):
        if state.word_position == state.num_words and len(state.current_heads) == 1:
            return True
        return False

    def __repr__(self):
        return "Finalize"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Finalize):
            return True
        return False

    def simplify(self):
        return self

    def __hash__(self):
        return hash(2000)

