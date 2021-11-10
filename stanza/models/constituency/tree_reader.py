"""
Reads ParseTree objects from a file, string, or similar input

Works by first splitting the input into (, ), and all other tokens,
then recursively processing those tokens into trees.
"""

import logging

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
    def __init__(self, line_num):
        super().__init__("Found a tree with both text children and bracketed children!  Line number %d" % line_num)
        self.line_num = line_num

def normalize(text):
    return text.replace("-LRB-", "(").replace("-RRB-", ")")

def recursive_open_tree(token_iterator, at_root, broken_ok):
    """
    Build a tree from the tokens in the token_iterator
    """
    # TODO: unwind the recursion
    text = []
    children = []

    token = next(token_iterator, None)
    if at_root:
        token_iterator.set_mark()
    while token is not None:
        if token is OPEN_PAREN:
            children.append(recursive_open_tree(token_iterator, at_root=False, broken_ok=broken_ok))
        elif token is CLOSE_PAREN:
            if len(text) == 0:
                if at_root:
                    return Tree(label="ROOT", children=children)
                elif broken_ok:
                    return Tree(label=None, children=children)
                else:
                    raise UnlabeledTreeError(token_iterator.line_num)

            pieces = " ".join(text).split()
            if len(pieces) == 1:
                return Tree(label=pieces[0], children=children)

            # the assumption here is that a language such as VI may
            # have spaces in the words, but it still represents
            # just one child
            label = pieces[0]
            child_label = " ".join(pieces[1:])
            if len(children) > 0:
                if broken_ok:
                    return Tree(label=label, children=children + [Tree(label=child_label)])
                else:
                    raise MixedTreeError(token_iterator.line_num)
            return Tree(label=label, children=Tree(label=normalize(child_label)))
        else:
            text.append(token)
        token = next(token_iterator, None)
    raise UnclosedTreeError(token_iterator.get_mark())

def recursive_read_trees(token_iterator, broken_ok):
    """
    Read all of the trees from the token_iterator

    TODO: some of the error cases we hit can be recovered from
    also, just in general it would be good to unwind the recursion
    """
    trees = []
    token = next(token_iterator, None)
    while token:
        if token is OPEN_PAREN:
            next_tree = recursive_open_tree(token_iterator, at_root=True, broken_ok=broken_ok)
            if next_tree is None:
                raise ValueError("Tree reader somehow created a None tree!  Line number %d" % token_iterator.line_num)
            trees.append(next_tree)
            token = next(token_iterator, None)
            continue

        if token is CLOSE_PAREN:
            raise ValueError("Tree document had too many close parens!  Line number %d" % token_iterator.line_num)
        else:
            raise ValueError("Tree document had text between trees!  Line number %d" % token_iterator.line_num)

    return trees

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
            if self.line_num >= len(self.lines):
                next(self.line_iterator, "")
                raise StopIteration

            line = next(self.line_iterator, "").strip()
            if not line:
                continue

            pieces = []
            open_pieces = line.split(OPEN_PAREN)
            for o_idx, open_piece in enumerate(open_pieces):
                if open_piece:
                    close_pieces = open_piece.split(CLOSE_PAREN)
                    for c_idx, close_piece in enumerate(close_pieces):
                        close_piece = close_piece.strip()
                        if close_piece:
                            pieces.append(close_piece)
                        if c_idx != len(close_pieces) - 1:
                            pieces.append(CLOSE_PAREN)
                if o_idx != len(open_pieces) - 1:
                    pieces.append(OPEN_PAREN)
            self.token_iterator = iter(pieces)
            n = next(self.token_iterator, None)

        return n

def read_trees(text, broken_ok=False):
    """
    Reads multiple trees from the text
    """
    token_iterator = TokenIterator(text)
    trees = recursive_read_trees(token_iterator, broken_ok=broken_ok)
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
