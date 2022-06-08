"""
Reads ParseTree objects from a file, string, or similar input

Works by first splitting the input into (, ), and all other tokens,
then recursively processing those tokens into trees.
"""

from collections import deque
import logging
import re

from stanza.models.common import utils
from stanza.models.constituency.parse_tree import Tree

tqdm = utils.get_tqdm()

OPEN_PAREN = "("
CLOSE_PAREN = ")"

logger = logging.getLogger('stanza.constituency')

# A few specific exception types to clarify parsing errors
# They store the line number where the error occurred

class UnclosedTreeError(ValueError):
    """
    A tree looked like (Foo
    """
    def __init__(self, line_num):
        super().__init__("Found an unfinished tree (missing close brackets).  Tree started on line %d" % line_num)
        self.line_num = line_num

class ExtraCloseTreeError(ValueError):
    """
    A tree looked like (Foo))
    """
    def __init__(self, line_num):
        super().__init__("Found a broken tree (extra close brackets).  Tree started on line %d" % line_num)
        self.line_num = line_num

class UnlabeledTreeError(ValueError):
    """
    A tree had no label, such as ((Foo) (Bar))

    This does not actually happen at the root, btw, as ROOT is silently added
    """
    def __init__(self, line_num):
        super().__init__("Found a tree with no label on a node!  Line number %d" % line_num)
        self.line_num = line_num

class MixedTreeError(ValueError):
    """
    Leaf and constituent children are mixed in the same node
    """
    def __init__(self, line_num, child_label, children):
        super().__init__("Found a tree with both text children and bracketed children!  Line number {}  Child label {}  Children {}".format(line_num, child_label, children))
        self.line_num = line_num
        self.child_label = child_label
        self.children = children

def normalize(text):
    return text.replace("-LRB-", "(").replace("-RRB-", ")")

def read_single_tree(token_iterator, broken_ok):
    """
    Build a tree from the tokens in the token_iterator
    """
    # we were called here at a open paren, so start the stack of
    # children with one empty list already on it
    children_stack = deque()
    children_stack.append([])
    text_stack = deque()
    text_stack.append([])

    token = next(token_iterator, None)
    token_iterator.set_mark()
    while token is not None:
        if token == OPEN_PAREN:
            children_stack.append([])
            text_stack.append([])
        elif token == CLOSE_PAREN:
            text = text_stack.pop()
            children = children_stack.pop()
            if text:
                pieces = " ".join(text).split()
                if len(pieces) == 1:
                    child = Tree(pieces[0], children)
                else:
                    # the assumption here is that a language such as VI may
                    # have spaces in the words, but it still represents
                    # just one child
                    label = pieces[0]
                    child_label = " ".join(pieces[1:])
                    if children:
                        if broken_ok:
                            child = Tree(label, children + [Tree(normalize(child_label))])
                        else:
                            raise MixedTreeError(token_iterator.line_num, child_label, children)
                    else:
                        child = Tree(label, Tree(normalize(child_label)))
                if not children_stack:
                    return child
            else:
                if not children_stack:
                    return Tree("ROOT", children)
                elif broken_ok:
                    child = Tree(None, children)
                else:
                    raise UnlabeledTreeError(token_iterator.line_num)
            children_stack[-1].append(child)
        else:
            text_stack[-1].append(token)
        token = next(token_iterator, None)
    raise UnclosedTreeError(token_iterator.get_mark())

LINE_SPLIT_RE = re.compile(r"([()])")

class TokenIterator:
    """
    A specific iterator for reading trees from a tree file

    The idea is that this will keep track of which line
    we are processing, so that an error can be logged
    from the correct line
    """
    def __init__(self, text):
        self.lines = text.split("\n")
        self.num_lines = len(self.lines)
        self.line_num = -1
        if self.num_lines > 1000:
            self.line_iterator = iter(tqdm(self.lines))
        else:
            self.line_iterator = iter(self.lines)
        self.token_iterator = iter([])
        self.mark = None

    def set_mark(self):
        self.mark = self.line_num

    def get_mark(self):
        if self.mark is None:
            raise ValueError("No mark set!")
        return self.mark

    def __iter__(self):
        return self

    def __next__(self):
        n = next(self.token_iterator, None)
        while n is None:
            self.line_num = self.line_num + 1
            if self.line_num >= self.num_lines:
                next(self.line_iterator, "")
                raise StopIteration

            line = next(self.line_iterator, "").strip()
            if not line:
                continue

            pieces = LINE_SPLIT_RE.split(line)
            pieces = [x.strip() for x in pieces]
            pieces = [x for x in pieces if x]
            self.token_iterator = iter(pieces)
            n = next(self.token_iterator, None)

        return n

def read_trees(text, broken_ok=False):
    """
    Reads multiple trees from the text

    TODO: some of the error cases we hit can be recovered from
    """
    trees = []
    token_iterator = TokenIterator(text)
    token = next(token_iterator, None)
    while token:
        if token == OPEN_PAREN:
            next_tree = read_single_tree(token_iterator, broken_ok=broken_ok)
            if next_tree is None:
                raise ValueError("Tree reader somehow created a None tree!  Line number %d" % token_iterator.line_num)
            trees.append(next_tree)
            token = next(token_iterator, None)
        elif token == CLOSE_PAREN:
            raise ExtraCloseTreeError(token_iterator.line_num)
        else:
            raise ValueError("Tree document had text between trees!  Line number %d" % token_iterator.line_num)

    return trees

def read_tree_file(filename):
    """
    Read all of the trees in the given file
    """
    with open(filename) as fin:
        trees = read_trees(fin.read())
    return trees

def read_treebank(filename):
    """
    Read a treebank and alter the trees to be a simpler format for learning to parse
    """
    logger.info("Reading trees from %s", filename)
    trees = read_tree_file(filename)
    trees = [t.prune_none().simplify_labels() for t in trees]

    illegal_trees = [t for t in trees if len(t.children) > 1]
    if len(illegal_trees) > 0:
        raise ValueError("Found {} tree(s) which had non-unary transitions at the ROOT.  First illegal tree: {}".format(len(illegal_trees), illegal_trees[0]))

    return trees

def main():
    """
    Reads a sample tree
    """
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = read_trees(text)
    print(trees)

if __name__ == '__main__':
    main()
