"""
Tree datastructure
"""

from collections import deque, Counter
from io import StringIO
import re

from stanza.models.common.doc import StanzaObject

# useful more for the "is" functionality than the time savings
CLOSE_PAREN = ')'
SPACE_SEPARATOR = ' '
OPEN_PAREN = '('

EMPTY_CHILDREN = ()

CONSTITUENT_SPLIT = re.compile("[-=#]")

class Tree(StanzaObject):
    """
    A data structure to represent a parse tree
    """
    def __init__(self, label=None, children=None):
        if children is None:
            self.children = EMPTY_CHILDREN
        elif isinstance(children, Tree):
            self.children = (children,)
        else:
            self.children = children

        self.label = label

    def is_leaf(self):
        return len(self.children) == 0

    def is_preterminal(self):
        return len(self.children) == 1 and len(self.children[0].children) == 0

    def yield_reversed_preterminals(self):
        """
        Yield the preterminals one at a time in BACKWARDS order

        This is done reversed as it is a frequently used method in the
        parser, so this is a tiny optimization
        """
        nodes = deque()
        nodes.append(self)
        while len(nodes) > 0:
            node = nodes.pop()
            if len(node.children) == 0:
                raise ValueError("Got called with an unexpected tree layout: {}".format(self))
            elif node.is_preterminal():
                yield node
            else:
                nodes.extend(node.children)

    def leaf_labels(self):
        """
        Get the labels of the leaves

        Not optimized whatsoever - current not an important part of
        the parser
        """
        preterminals = reversed([x for x in self.yield_reversed_preterminals()])
        words = [x.children[0].label for x in preterminals]
        return words

    def preterminals(self):
        return list(reversed(list(self.yield_reversed_preterminals())))

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
                if len(node.children) == 0:
                    if node.label is not None:
                        buf.write(node.label)
                    continue
                buf.write(OPEN_PAREN)
                if node.label is not None:
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

    # TODO: test different pattern
    def simplify_labels(self, pattern=CONSTITUENT_SPLIT):
        """
        Return a copy of the tree with the -=# removed

        Leaves the text of the leaves alone.
        """
        new_label = self.label
        # check len(new_label) just in case it's a tag of - or =
        if new_label and not self.is_leaf() and len(new_label) > 1 and new_label not in ('-LRB-', '-RRB-'):
            new_label = pattern.split(new_label)[0]
        new_children = [child.simplify_labels(pattern) for child in self.children]
        return Tree(new_label, new_children)

    def remap_constituent_labels(self, label_map):
        """
        Copies the tree with some labels replaced.

        Labels in the map are replaced with the mapped value.
        Labels not in the map are unchanged.
        """
        if self.is_leaf():
            return Tree(self.label)
        if self.is_preterminal():
            return Tree(self.label, Tree(self.children[0].label))
        new_label = label_map.get(self.label, self.label)
        return Tree(new_label, [child.remap_constituent_labels(label_map) for child in self.children])

    def remap_words(self, word_map):
        """
        Copies the tree with some labels replaced.

        Labels in the map are replaced with the mapped value.
        Labels not in the map are unchanged.
        """
        if self.is_leaf():
            new_label = word_map.get(self.label, self.label)
            return Tree(new_label)
        if self.is_preterminal():
            return Tree(self.label, self.children[0].remap_words(word_map))
        return Tree(self.label, [child.remap_words(word_map) for child in self.children])

    def replace_words(self, words):
        """
        Replace all leaf words with the words in the given list (or iterable)

        Returns a new tree
        """
        word_iterator = iter(words)
        def recursive_replace_words(subtree):
            if subtree.is_leaf():
                word = next(word_iterator, None)
                if word is None:
                    raise ValueError("Not enough words to replace all leaves")
                return Tree(word)
            return Tree(subtree.label, [recursive_replace_words(x) for x in subtree.children])

        new_tree = recursive_replace_words(self)
        if any(True for _ in word_iterator):
            raise ValueError("Too many tags for the given tree")
        return new_tree


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
