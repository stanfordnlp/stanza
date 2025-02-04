"""
Tree datastructure
"""

from collections import deque, Counter
import copy
from enum import Enum
from io import StringIO
import itertools
import re
import warnings

from stanza.models.common.stanza_object import StanzaObject

# useful more for the "is" functionality than the time savings
CLOSE_PAREN = ')'
SPACE_SEPARATOR = ' '
OPEN_PAREN = '('

EMPTY_CHILDREN = ()

# used to split off the functional tags from various treebanks
# for example, the Icelandic treebank (which we don't currently
# incorporate) uses * to distinguish 'ADJP', 'ADJP*OC' but we treat
# those as the same
CONSTITUENT_SPLIT = re.compile("[-=#*]")

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
    VLSP              = 4  # <s> (S ... ) </s>
    LATEX_TREE        = 5  # \Tree [.S [.NP ... ] ]


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
        if self.is_leaf():
            return [self.label]

        words = [x.children[0].label for x in self.yield_preterminals()]
        return words

    def __len__(self):
        return len(self.leaf_labels())

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
          O       -> one line PTB format, which is the default anyway
          L       -> open and close brackets are labeled, spaces in the tokens are replaced with _
          P       -> pretty print over multiple lines
          V       -> surround lines with <s>...</s>, don't print ROOT, and turn () into L/RBKT
          ?       -> spaces in the tokens are replaced with ? for any value of ? other than OLP
                     warning: this may be removed in the future
          ?{OLPV} -> specific format AND a custom space replacement
          Vi      -> add an ID to the <s> in the V format.  Also works with ?Vi
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
        elif spec and spec[0] == 'V':
            print_format = TreePrintMethod.VLSP
            use_tree_id = spec[-1] == 'i'
        elif spec and len(spec) > 1 and spec[1] == 'V':
            print_format = TreePrintMethod.VLSP
            space_replacement = spec[0]
            use_tree_id = spec[-1] == 'i'
        elif spec == 'T':
            print_format = TreePrintMethod.LATEX_TREE
        elif spec and len(spec) > 1 and spec[1] == 'T':
            print_format = TreePrintMethod.LATEX_TREE
            space_replacement = spec[0]
        elif spec:
            space_replacement = spec[0]
            warnings.warn("Use of a custom replacement without a format specifier is deprecated.  Please use {}O instead".format(space_replacement), stacklevel=2)

        LRB = "LBKT" if print_format == TreePrintMethod.VLSP else "-LRB-"
        RRB = "RBKT" if print_format == TreePrintMethod.VLSP else "-RRB-"
        def normalize(text):
            return text.replace(" ", space_replacement).replace("(", LRB).replace(")", RRB)

        if print_format is TreePrintMethod.PRETTY:
            return self.pretty_print(normalize)

        with StringIO() as buf:
            stack = deque()
            if print_format == TreePrintMethod.VLSP:
                if use_tree_id:
                    buf.write("<s id={}>\n".format(self.tree_id))
                else:
                    buf.write("<s>\n")
                if len(self.children) == 0:
                    raise ValueError("Cannot print an empty tree with V format")
                elif len(self.children) > 1:
                    raise ValueError("Cannot print a tree with %d branches with V format" % len(self.children))
                stack.append(self.children[0])
            elif print_format == TreePrintMethod.LATEX_TREE:
                buf.write("\\Tree ")
                if len(self.children) == 0:
                    raise ValueError("Cannot print an empty tree with T format")
                elif len(self.children) == 1 and len(self.children[0].children) == 0:
                    buf.write("[.? ")
                    buf.write(normalize(self.children[0].label))
                    buf.write(" ]")
                elif self.label == 'ROOT':
                    stack.append(self.children[0])
                else:
                    stack.append(self)
            else:
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

                if print_format is TreePrintMethod.LATEX_TREE:
                    if node.is_preterminal():
                        buf.write(normalize(node.children[0].label))
                        continue
                    buf.write("[.%s" % normalize(node.label))
                    stack.append(" ]")
                elif print_format is TreePrintMethod.ONE_LINE or print_format is TreePrintMethod.VLSP:
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
            if print_format == TreePrintMethod.VLSP:
                buf.write("\n</s>")
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
        if isinstance(trees, Tree):
            trees = [trees]
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
    def get_compound_constituents(trees, separate_root=False):
        constituents = set()
        stack = deque()
        for tree in trees:
            if separate_root:
                constituents.add((tree.label,))
                for child in tree.children:
                    stack.append(child)
            else:
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

    def reverse(self):
        """
        Flip a tree backwards

        The intent is to train a parser backwards to see if the
        forward and backwards parsers can augment each other
        """
        if self.is_leaf():
            return Tree(self.label)

        new_children = [child.reverse() for child in reversed(self.children)]
        return Tree(self.label, new_children)

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


    def replace_tags(self, tags):
        if self.is_leaf():
            raise ValueError("Must call replace_tags with non-leaf")

        if isinstance(tags, Tree):
            tag_iterator = (x.label for x in tags.yield_preterminals())
        else:
            tag_iterator = iter(tags)

        new_tree = copy.deepcopy(self)
        queue = deque()
        queue.append(new_tree)
        while len(queue) > 0:
            next_node = queue.pop()
            if next_node.is_preterminal():
                try:
                    label = next(tag_iterator)
                except StopIteration:
                    raise ValueError("Not enough tags in sentence for given tree")
                next_node.label = label
            elif next_node.is_leaf():
                raise ValueError("Got a badly structured tree: {}".format(self))
            else:
                queue.extend(reversed(next_node.children))

        if any(True for _ in tag_iterator):
            raise ValueError("Too many tags for the given tree")

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

    def mark_spans(self):
        self._mark_spans(0)

    def _mark_spans(self, start_index):
        self.start_index = start_index

        if len(self.children) == 0:
            self.end_index = start_index + 1
            return

        for child in self.children:
            child._mark_spans(start_index)
            start_index = child.end_index

        self.end_index = start_index

    @staticmethod
    def single_missing_node_errors(original, predicted):
        """
        Given the correct tree and the predicted tree, returns a list of single node missing / added errors.

        The return format will be:
          (outer label, missing/added label, left child label, right child label, T/F whether the node is supposed to be there)
        Interestingly the nodes themselves aren't needed, at least not yet

        Operates by recursively going through both trees at the same time.
          If the span lengths of a pair of nodes are the same, it recurses on that pair of nodes.
          Otherwise, if two consecutive spans from one tree add up to one span of the other tree, this is a candidate.
          This candidate is accepted if the other tree's larger span is a node with exactly two children,
            matching the labels of the original tree's children, with the spans the same as well.
          Note that this does not guarantee the internal structure of those spans is the same.
        """
        original.mark_spans()
        predicted.mark_spans()

        def check_missing_error(separate_tree, combined_tree, separate_idx, combined_idx, should_nest):
            if separate_tree.label != combined_tree.label:
                return None
            if len(combined_tree.children) != 2:
                return False
            if (separate_tree.children[separate_idx].start_index == combined_tree.children[combined_idx].children[0].start_index and
                separate_tree.children[separate_idx].end_index   == combined_tree.children[combined_idx].children[0].end_index and
                separate_tree.children[separate_idx].label       == combined_tree.children[combined_idx].children[0].label and
                separate_tree.children[separate_idx+1].start_index == combined_tree.children[combined_idx].children[-1].start_index and
                separate_tree.children[separate_idx+1].end_index   == combined_tree.children[combined_idx].children[-1].end_index and
                separate_tree.children[separate_idx+1].label       == combined_tree.children[combined_idx].children[-1].label):
                return (combined_tree.label, combined_tree.children[combined_idx].label,
                        combined_tree.children[combined_idx].children[0].label, combined_tree.children[combined_idx].children[-1].label,
                        should_nest)

        errors = []
        def missing_node_helper(original, predicted):
            #print("Checking: %s %s" % (original, predicted))
            if original.is_preterminal() or predicted.is_preterminal():
                return

            orig_idx = 0
            pred_idx = 0
            while orig_idx < len(original.children) and pred_idx < len(predicted.children):
                #print(original.children[orig_idx].start_index, original.children[orig_idx].end_index,
                #      predicted.children[pred_idx].start_index, predicted.children[pred_idx].end_index)
                if original.children[orig_idx].start_index < predicted.children[pred_idx].start_index:
                    orig_idx += 1
                    continue
                if original.children[orig_idx].start_index > predicted.children[pred_idx].start_index:
                    pred_idx += 1
                    continue
                # the start indices are the same
                # first thing to check: if the end indices are the same, can recurse
                if original.children[orig_idx].end_index == predicted.children[pred_idx].end_index:
                    missing_node_helper(original.children[orig_idx], predicted.children[pred_idx])
                    orig_idx += 1
                    pred_idx += 1
                    continue
                # in this case, one of the end indices is lower.  there could potentially
                # be an attachment error in that case
                attachment = None
                if original.children[orig_idx].end_index < predicted.children[pred_idx].end_index:
                    if orig_idx + 1 < len(original.children) and original.children[orig_idx+1].end_index == predicted.children[pred_idx].end_index:
                        attachment = check_missing_error(original, predicted, orig_idx, pred_idx, False)
                elif original.children[orig_idx].end_index > predicted.children[pred_idx].end_index:
                    if pred_idx + 1 < len(predicted.children) and predicted.children[pred_idx+1].end_index == original.children[orig_idx].end_index:
                        attachment = check_missing_error(predicted, original, pred_idx, orig_idx, True)
                orig_idx += 1
                pred_idx += 1
                if attachment:
                    errors.append(attachment)

        missing_node_helper(original, predicted)
        return errors

    def count_candidate_missing_nodes(self, possible_edits):
        """
        Count how many times the possible edits show up in the tree.

        Edits should be of the form:
          (parent, binary, left, right)
        in other words, same as returned by single_missing_node_errors
        """
        if self.is_preterminal():
            return 0
        total = 0
        for child in self.children:
            total += child.count_candidate_missing_nodes(possible_edits)
        for edit in possible_edits:
            parent_label, binary_label, left_label, right_label = edit
            if self.label != parent_label:
                continue
            for child in self.children:
                if child.label == binary_label and len(child.children) == 2 and child.children[0].label == left_label and child.children[1].label == right_label:
                    total += 1
            for child_idx in range(len(self.children) - 1):
                if self.children[child_idx].label == left_label and self.children[child_idx+1].label == right_label:
                    total += 1
        return total

    def flip_missing_node_errors(self, possible_edits):
        """
        Build a new tree with the edits in possible_edits applied recursively, top down

        Edits should be of the form:
          (parent, binary, left, right)
        in other words, same as returned by single_missing_node_errors
        """
        if self.is_preterminal():
            return Tree(self.label, Tree(self.children[0].label))

        # TODO: maybe make the possible edits a tuple?
        candidate_edits = [x for x in possible_edits if x[0] == self.label]
        new_children = []

        skip = False
        for child_idx, child in enumerate(self.children):
            if skip:
                skip = False
                continue

            if (len(child.children) == 2 and
                any(x[1] == child.label and x[2] == child.children[0].label and x[3] == child.children[1].label for x in possible_edits)):
                # If this node matches one of the edits that removes a node,
                # execute that edit by recursively building the two children
                # and keeping those as new children
                new_children.append(child.children[0].flip_missing_node_errors(possible_edits))
                new_children.append(child.children[1].flip_missing_node_errors(possible_edits))
            elif child_idx + 1 == len(self.children):
                # this is the last node, so we can't possibly combine this node with a next node
                new_children.append(child.flip_missing_node_errors(possible_edits))
            else:
                next_child = self.children[child_idx+1]
                if (len(next_child.children) == 2 and
                    any(x[1] == next_child.label and x[2] == next_child.children[0].label and x[3] == next_child.children[1].label for x in possible_edits)):
                    # next child matches one of the edits, so we will remove that child's node as well
                    new_children.append(child.flip_missing_node_errors(possible_edits))
                else:
                    left_label = child.label
                    right_label = next_child.label
                    for candidate_edit in candidate_edits:
                        if candidate_edit[2] == left_label and candidate_edit[3] == right_label:
                            new_child = Tree(candidate_edit[1],
                                             [child.flip_missing_node_errors(possible_edits),
                                              next_child.flip_missing_node_errors(possible_edits)])
                            new_children.append(new_child)
                            skip = True
                            break
                    else:
                        new_children.append(child.flip_missing_node_errors(possible_edits))

        return Tree(self.label, new_children)


    def flip_first_missing_node_error(self, possible_edits):
        """
        Build a new tree with exactly the first of the edits in possible_edits applied

        Edits should be of the form:
          (parent, binary, left, right)
        in other words, same as returned by single_missing_node_errors
        """
        if self.is_preterminal():
            return Tree(self.label, Tree(self.children[0].label))

        edits_performed = 0
        def flip_helper(tree, possible_edits):
            nonlocal edits_performed
            # TODO: maybe make the possible edits a tuple?
            candidate_edits = [x for x in possible_edits if x[0] == tree.label]
            new_children = []

            skip = False
            for child_idx, child in enumerate(tree.children):
                if skip:
                    skip = False
                    continue

                if edits_performed > 0:
                    new_children.append(flip_helper(child, possible_edits))
                    continue

                if (len(child.children) == 2 and
                    any(x[1] == child.label and x[2] == child.children[0].label and x[3] == child.children[1].label for x in possible_edits)):
                    # If this node matches one of the edits that removes a node,
                    # execute that edit by recursively building the two children
                    # and keeping those as new children
                    edits_performed = 1
                    new_children.append(flip_helper(child.children[0], possible_edits))
                    new_children.append(flip_helper(child.children[1], possible_edits))
                elif child_idx + 1 == len(tree.children):
                    # this is the last node, so we can't possibly combine this node with a next node
                    new_children.append(flip_helper(child, possible_edits))
                else:
                    next_child = tree.children[child_idx+1]
                    if (len(next_child.children) == 2 and
                        any(x[1] == next_child.label and x[2] == next_child.children[0].label and x[3] == next_child.children[1].label for x in possible_edits)):
                        # next child matches one of the edits, so we will remove that child's node as well
                        new_children.append(flip_helper(child, possible_edits))
                    else:
                        left_label = child.label
                        right_label = next_child.label
                        for candidate_edit in candidate_edits:
                            if candidate_edit[2] == left_label and candidate_edit[3] == right_label:
                                edits_performed = 1
                                new_child = Tree(candidate_edit[1],
                                                 [flip_helper(child, possible_edits),
                                                  flip_helper(next_child, possible_edits)])
                                new_children.append(new_child)
                                skip = True
                                break
                        else:
                            new_children.append(flip_helper(child, possible_edits))

            return Tree(tree.label, new_children)

        return flip_helper(self, possible_edits)
