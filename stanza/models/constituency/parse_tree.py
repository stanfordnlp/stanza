"""
Tree datastructure
"""

from collections import deque
from io import StringIO

from stanza.models.common.doc import StanzaObject

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
                if node == ')' or node == ' ':
                    buf.write(node)
                    continue
                if not node.children:
                    buf.write(node.label)
                    continue
                buf.write("(")
                buf.write(node.label)
                stack.append(')')
                for child in reversed(node.children):
                    stack.append(child)
                    stack.append(' ')
            buf.seek(0)
            return buf.read()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Tree):
            return False
        if self.label != other.label:
            return False
        if self.children != other.children:
            return False
        return True

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(x.depth() for x in self.children)
