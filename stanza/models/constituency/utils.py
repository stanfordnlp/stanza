"""
Collects a few of the conparser utility methods which don't belong elsewhere
"""

from collections import deque
import copy

from stanza.models.common.doc import TEXT, Document

def replace_tags(tree, tags):
    if tree.is_leaf():
        raise ValueError("Must call replace_tags with non-leaf")

    tag_iterator = iter(tags)

    new_tree = copy.deepcopy(tree)
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
            raise ValueError("Got a badly structured tree: {}".format(tree))
        else:
            queue.extend(reversed(next_node.children))

    if any(True for _ in tag_iterator):
        raise ValueError("Too many tags for the given tree")

    return new_tree


def retag_trees(trees, pipeline, xpos=True):
    """
    Retag all of the trees using the given processor

    Returns a list of new trees
    """
    sentences = []
    for tree in trees:
        tokens = [{TEXT: pt.children[0].label} for pt in tree.preterminals()]
        sentences.append(tokens)

    doc = Document(sentences)
    doc = pipeline(doc)
    if xpos:
        tag_lists = [[x.xpos for x in sentence.words] for sentence in doc.sentences]
    else:
        tag_lists = [[x.upos for x in sentence.words] for sentence in doc.sentences]

    new_trees = [replace_tags(tree, tags) for tree, tags in zip(trees, tag_lists)]
    return new_trees

