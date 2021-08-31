"""
Reads ParseTree objects from a file, string, or similar input

Works by first splitting the input into (, ), and all other tokens,
then recursively processing those tokens into trees.
"""

from stanza.models.common import utils
from stanza.models.constituency.parse_tree import Tree

tqdm = utils.get_tqdm()

OPEN_PAREN = "("
CLOSE_PAREN = ")"

def recursive_open_tree(token_iterator, at_root):
    # TODO: unwind the recursion
    text = []
    children = []

    token = next(token_iterator, None)
    while token != None:
        if token is OPEN_PAREN:
            children.append(recursive_open_tree(token_iterator, at_root=False))
        elif token is CLOSE_PAREN:
            if len(text) == 0:
                if at_root:
                    return Tree(label="ROOT", children=children)
                raise ValueError("Found a tree with no label on a node")

            pieces = " ".join(text).split()
            if len(pieces) == 1:
                return Tree(label=pieces[0], children=children)
            if len(children) > 0:
                raise ValueError("Found a tree with both text children and bracketed children")
            label = pieces[0]
            child_label = " ".join(pieces[1:])
            return Tree(label=label, children=Tree(label=child_label))
        else:
            text.append(token)
        token = next(token_iterator, None)

def recursive_read_trees(token_iterator):
    """
    TODO: some of the error cases we hit can be recovered from
    also, the recursive call can throw errors and we should be able to report where
    also, just in general it would be good to unwind the recursion
    """
    trees = []
    token = next(token_iterator, None)
    while token:
        if token is OPEN_PAREN:
            trees.append(recursive_open_tree(token_iterator, at_root=True))
            token = next(token_iterator, None)
            continue

        if len(trees) > 0:
            error = "Most recent tree: {}".format(trees[-1])
        else:
            error = "Error occurred at start of document"

        if token is CLOSE_PAREN:
            raise ValueError("Tree document had too many close parens!  " + error)
        else:
            raise ValueError("Tree document had text between trees!  " + error)

    return trees

def read_trees(text):
    """
    Reads multiple trees from the text
    """
    lines = text.split("\n")
    pieces = []
    for line in lines:
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

    if len(pieces) > 10000:
        pieces = tqdm(pieces)
    token_iterator = iter(pieces)
    trees = recursive_read_trees(token_iterator)
    return trees

def read_tree_file(filename):
    with open(filename) as fin:
        trees = read_trees(fin.read())
    return trees

if __name__ == '__main__':
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = read_trees(text)
    print(trees)
