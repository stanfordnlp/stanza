"""
Tree datastructure
"""

from collections import deque, Counter
from io import StringIO

from stanza.models.common.doc import StanzaObject

# useful more for the "is" functionality than the time savings
CLOSE_PAREN = ')'
SPACE_SEPARATOR = ' '
OPEN_PAREN = '('

class Tree(StanzaObject):
    """
    A data structure to represent a parse tree
    """
    def __init__(self, label=None, children=None):
        if children is None:
            self.children = []
        elif isinstance(children, Tree):
            self.children = (children,)
        else:
            self.children = children

        self.label = label

    def is_leaf(self):
        return len(self.children) == 0

    def is_preterminal(self):
        return len(self.children) == 1 and len(self.children[0].children) == 0

    def yield_preterminals(self):
        if self.is_leaf():
            pass
        elif self.is_preterminal():
            yield self
        else:
            for child in self.children:
                for preterminal in child.yield_preterminals():
                    yield preterminal

    def __repr__(self):
        """
        Turn the tree into a string representing the tree

        Note that this is not a recursive traversal
        Otherwise, a tree too deep might blow up the call stack
        """
        with StringIO() as buf:
            stack = deque()
            stack.append(self)
            while len(stack) > 0:
                node = stack.pop()
                # note that == can recursively call == in some circumstances!
                if node is CLOSE_PAREN or node is SPACE_SEPARATOR:
                    buf.write(node)
                    continue
                if not node.children:
                    buf.write(node.label)
                    continue
                buf.write(OPEN_PAREN)
                buf.write(node.label)
                stack.append(CLOSE_PAREN)
                for child in reversed(node.children):
                    stack.append(child)
                    stack.append(SPACE_SEPARATOR)
            buf.seek(0)
            return buf.read()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Tree):
            return False
        if self.label != other.label:
            return False
        if len(self.children) != len(other.children):
            return False
        if any(c1 != c2 for c1, c2 in zip(self.children, other.children)):
            return False
        return True

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(x.depth() for x in self.children)

    def visit_preorder(self, internal=None, preterminal=None, leaf=None):
        """
        Visit the tree in a preorder order

        Applies the given functions to each node.
        internal: if not None, applies this function to each non-leaf, non-preterminal node
        preterminal: if not None, applies this functiion to each preterminal
        leaf: if not None, applies this function to each leaf

        The functions should *not* destructively alter the trees.
        There is no attempt to interpret the results of calling these functions.
        Rather, you can use visit_preorder to collect stats on trees, etc.
        """
        if self.is_leaf():
            if leaf:
                leaf(self)
        elif self.is_preterminal():
            if preterminal:
                preterminal(self)
        else:
            if internal:
                internal(self)
        for child in self.children:
            child.visit_preorder(internal, preterminal, leaf)

    @staticmethod
    def get_unique_constituent_labels(trees):
        """
        Walks over all of the trees and gets all of the unique constituent names from the trees
        """
        if isinstance(trees, Tree):
            trees = [trees]

        constituents = set()
        for tree in trees:
            tree.visit_preorder(internal = lambda x: constituents.add(x.label))
        return sorted(constituents)

    @staticmethod
    def get_unique_tags(trees):
        """
        Walks over all of the trees and gets all of the unique tags from the trees
        """
        if isinstance(trees, Tree):
            trees = [trees]

        tags = set()
        for tree in trees:
            tree.visit_preorder(preterminal = lambda x: tags.add(x.label))
        return sorted(tags)

    @staticmethod
    def get_unique_words(trees):
        """
        Walks over all of the trees and gets all of the unique words from the trees
        """
        if isinstance(trees, Tree):
            trees = [trees]

        words = set()
        for tree in trees:
            tree.visit_preorder(leaf = lambda x: words.add(x.label))
        return sorted(words)

    @staticmethod
    def get_rare_words(trees, threshold=0.05):
        """
        Walks over all of the trees and gets the least frequently occurring words.

        threshold: choose the bottom X percent
        """
        if isinstance(trees, Tree):
            trees = [trees]

        words = Counter()
        for tree in trees:
            tree.visit_preorder(leaf = lambda x: words.update([x.label]))
        threshold = max(int(len(words) * threshold), 1)
        return sorted(x[0] for x in words.most_common()[:-threshold-1:-1])

    @staticmethod
    def get_root_labels(trees):
        return sorted(set(x.label for x in trees))

    @staticmethod
    def get_compound_constituents(trees):
        constituents = set()
        stack = deque()
        for tree in trees:
            stack.append(tree)
            while len(stack) > 0:
                node = stack.pop()
                if node.is_leaf() or node.is_preterminal():
                    continue
                labels = [node.label]
                while len(node.children) == 1 and not node.children[0].is_preterminal():
                    node = node.children[0]
                    labels.append(node.label)
                constituents.add(tuple(labels))
                for child in node.children:
                    stack.append(child)
        return sorted(constituents)
    
    def simplify_labels(self):
        """
        Return a copy of the tree with the -=# removed

        Leaves the text of the leaves alone.
        """
        new_label = self.label
        # check len(new_label) just in case it's a tag of - or =
        if new_label and not self.is_leaf() and len(new_label) > 1 and new_label not in ('-LRB-', '-RRB-'):
            new_label = new_label.split("-")[0].split("=")[0].split("#")[0]
        new_children = [child.simplify_labels() for child in self.children]
        return Tree(new_label, new_children)

    def prune_none(self):
        """
        Return a copy of the tree, eliminating all nodes which are in one of two categories:
            they are a preterminal -NONE-, such as appears in PTB
            they have been pruned to 0 children by the recursive call
        """
        if self.is_leaf():
            return Tree(self.label)
        if self.is_preterminal():
            if self.label == '-NONE-':
                return None
            return Tree(self.label, Tree(self.children[0].label))
        # must be internal node
        new_children = [child.prune_none() for child in self.children]
        new_children = [child for child in new_children if child is not None]
        if len(new_children) == 0:
            return None
        return Tree(self.label, new_children)
