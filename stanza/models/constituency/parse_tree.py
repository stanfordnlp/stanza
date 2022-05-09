"""
Tree datastructure
"""

from collections import deque, Counter
from enum import Enum
from io import StringIO
import itertools
import re
import warnings

from stanza.models.common.doc import StanzaObject

# useful more for the "is" functionality than the time savings
CLOSE_PAREN = ')'
SPACE_SEPARATOR = ' '
OPEN_PAREN = '('

EMPTY_CHILDREN = ()

CONSTITUENT_SPLIT = re.compile("[-=#]")

# These words occur in the VLSP dataset.
# The documentation claims there might be *O*, although those don't
# seem to exist in practice
WORDS_TO_PRUNE = ('*E*', '*T*', '*O*')

class TreePrintMethod(Enum):
    """
    Describes a few options for printing trees.

    This probably doesn't need to be used directly.  See __format__
    """
    ONE_LINE          = 1  # (ROOT (S ...  ))
    LABELED_PARENS    = 2  # (_ROOT (_S ... )_S )_ROOT
    PRETTY            = 3  # multiple lines


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
            self.children = tuple(children)

        self.label = label

    def is_leaf(self):
        return len(self.children) == 0

    def is_preterminal(self):
        return len(self.children) == 1 and len(self.children[0].children) == 0

    def yield_preterminals(self):
        """
        Yield the preterminals one at a time in order
        """
        if self.is_preterminal():
            yield self
            return

        if self.is_leaf():
            raise ValueError("Attempted to iterate preterminals on non-internal node")

        iterator = iter(self.children)
        node = next(iterator, None)
        while node is not None:
            if node.is_preterminal():
                yield node
            else:
                iterator = itertools.chain(node.children, iterator)
            node = next(iterator, None)

    def leaf_labels(self):
        """
        Get the labels of the leaves
        """
        words = [x.children[0].label for x in self.yield_preterminals()]
        return words

    def all_leaves_are_preterminals(self):
        """
        Returns True if all leaves are under preterminals, False otherwise
        """
        if self.is_leaf():
            return False

        if self.is_preterminal():
            return True

        return all(t.all_leaves_are_preterminals() for t in self.children)

    def pretty_print(self, normalize=None):
        """
        Print with newlines & indentation on each line

        Preterminals and nodes with all preterminal children go on their own line

        You can pass in your own normalize() function.  If you do,
        make sure the function updates the parens to be something
        other than () or the brackets will be broken
        """
        if normalize is None:
            normalize = lambda x: x.replace("(", "-LRB-").replace(")", "-RRB-")

        indent = 0
        with StringIO() as buf:
            stack = deque()
            stack.append(self)
            while len(stack) > 0:
                node = stack.pop()

                if node is CLOSE_PAREN:
                    # if we're trying to pretty print trees, pop all off close parens
                    # then write a newline
                    while node is CLOSE_PAREN:
                        indent -= 1
                        buf.write(CLOSE_PAREN)
                        if len(stack) == 0:
                            node = None
                            break
                        node = stack.pop()
                    buf.write("\n")
                    if node is None:
                        break
                    stack.append(node)
                elif node.is_preterminal():
                    buf.write("  " * indent)
                    buf.write("%s%s %s%s" % (OPEN_PAREN, normalize(node.label), normalize(node.children[0].label), CLOSE_PAREN))
                    if len(stack) == 0 or stack[-1] is not CLOSE_PAREN:
                        buf.write("\n")
                elif all(x.is_preterminal() for x in node.children):
                    buf.write("  " * indent)
                    buf.write("%s%s" % (OPEN_PAREN, normalize(node.label)))
                    for child in node.children:
                        buf.write(" %s%s %s%s" % (OPEN_PAREN, normalize(child.label), normalize(child.children[0].label), CLOSE_PAREN))
                    buf.write(CLOSE_PAREN)
                    if len(stack) == 0 or stack[-1] is not CLOSE_PAREN:
                        buf.write("\n")
                else:
                    buf.write("  " * indent)
                    buf.write("%s%s\n" % (OPEN_PAREN, normalize(node.label)))
                    stack.append(CLOSE_PAREN)
                    for child in reversed(node.children):
                        stack.append(child)
                    indent += 1

            buf.seek(0)
            return buf.read()

    def __format__(self, spec):
        """
        Turn the tree into a string representing the tree

        Note that this is not a recursive traversal
        Otherwise, a tree too deep might blow up the call stack

        There is a type specific format:
          O      -> one line PTB format, which is the default anyway
          L      -> open and close brackets are labeled, spaces in the tokens are replaced with _
          P      -> pretty print over multiple lines
          ?      -> spaces in the tokens are replaced with ? for any value of ? other than OLP
                    warning: this may be removed in the future
          ?{OLP} -> specific format AND a custom space replacement
        """
        space_replacement = " "
        print_format = TreePrintMethod.ONE_LINE
        if spec == 'L':
            print_format = TreePrintMethod.LABELED_PARENS
            space_replacement = "_"
        elif spec and spec[-1] == 'L':
            print_format = TreePrintMethod.LABELED_PARENS
            space_replacement = spec[0]
        elif spec == 'O':
            print_format = TreePrintMethod.ONE_LINE
        elif spec and spec[-1] == 'O':
            print_format = TreePrintMethod.ONE_LINE
            space_replacement = spec[0]
        elif spec == 'P':
            print_format = TreePrintMethod.PRETTY
        elif spec and spec[-1] == 'P':
            print_format = TreePrintMethod.PRETTY
            space_replacement = spec[0]
        elif spec:
            space_replacement = spec[0]
            warnings.warn("Use of a custom replacement without a format specifier is deprecated.  Please use {}O instead".format(space_replacement), stacklevel=2)

        def normalize(text):
            return text.replace(" ", space_replacement).replace("(", "-LRB-").replace(")", "-RRB-")

        if print_format is TreePrintMethod.PRETTY:
            return self.pretty_print(normalize)

        with StringIO() as buf:
            stack = deque()
            stack.append(self)
            while len(stack) > 0:
                node = stack.pop()

                if isinstance(node, str):
                    buf.write(node)
                    continue
                if len(node.children) == 0:
                    if node.label is not None:
                        buf.write(normalize(node.label))
                    continue

                if print_format is TreePrintMethod.ONE_LINE:
                    buf.write(OPEN_PAREN)
                    if node.label is not None:
                        buf.write(normalize(node.label))
                    stack.append(CLOSE_PAREN)
                elif print_format is TreePrintMethod.LABELED_PARENS:
                    buf.write("%s_%s" % (OPEN_PAREN, normalize(node.label)))
                    stack.append(CLOSE_PAREN + "_" + normalize(node.label))
                    stack.append(SPACE_SEPARATOR)

                for child in reversed(node.children):
                    stack.append(child)
                    stack.append(SPACE_SEPARATOR)
            buf.seek(0)
            return buf.read()

    def __repr__(self):
        return "{}".format(self)

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
        constituents = Tree.get_constituent_counts(trees)
        return sorted(set(constituents.keys()))

    @staticmethod
    def get_constituent_counts(trees):
        """
        Walks over all of the trees and gets the count of the unique constituent names from the trees
        """
        if isinstance(trees, Tree):
            trees = [trees]

        constituents = Counter()
        for tree in trees:
            tree.visit_preorder(internal = lambda x: constituents.update([x.label]))
        return constituents

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
    def get_common_words(trees, num_words):
        """
        Walks over all of the trees and gets the most frequently occurring words.
        """
        if num_words == 0:
            return set()

        if isinstance(trees, Tree):
            trees = [trees]

        words = Counter()
        for tree in trees:
            tree.visit_preorder(leaf = lambda x: words.update([x.label]))
        return sorted(x[0] for x in words.most_common()[:num_words])

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
            raise ValueError("Too many words for the given tree")
        return new_tree


    def prune_none(self):
        """
        Return a copy of the tree, eliminating all nodes which are in one of two categories:
            they are a preterminal -NONE-, such as appears in PTB
              *E* shows up in a VLSP dataset
            they have been pruned to 0 children by the recursive call
        """
        if self.is_leaf():
            return Tree(self.label)
        if self.is_preterminal():
            if self.label == '-NONE-' or self.children[0].label in WORDS_TO_PRUNE:
                return None
            return Tree(self.label, Tree(self.children[0].label))
        # must be internal node
        new_children = [child.prune_none() for child in self.children]
        new_children = [child for child in new_children if child is not None]
        if len(new_children) == 0:
            return None
        return Tree(self.label, new_children)

    def count_unary_depth(self):
        if self.is_preterminal() or self.is_leaf():
            return 0
        if len(self.children) == 1:
            t = self
            score = 0
            while not t.is_preterminal() and not t.is_leaf() and len(t.children) == 1:
                score = score + 1
                t = t.children[0]
            child_score = max(tc.count_unary_depth() for tc in t.children)
            score = max(score, child_score)
            return score
        score = max(t.count_unary_depth() for t in self.children)
        return score

    @staticmethod
    def write_treebank(trees, out_file, fmt="{}"):
        with open(out_file, "w", encoding="utf-8") as fout:
            for tree in trees:
                fout.write(fmt.format(tree))
                fout.write("\n")
