class TreeStack:
    """
    A stack which can branch in several directions, as long as you
    keep track of the branching heads

    We don't simply keep track of a single stack object using a deque
    because versions of the parser which use a beam will want to be
    able to branch in different directions from the same base stack
    """
    def __init__(self, value, parent=None, value_to_str=str):
        # value could be as transition, a word, or a partially built constituent
        # value_to_str allows for an alternate value -> str for the case
        # of more complicated values, such as values which include lstm tensors
        self.value = value
        # parent == None will represent the end of the stack
        self.parent = parent
        self.value_to_str = value_to_str
        if parent is None:
            self._len = 1
        else:
            self._len = len(parent) + 1

    def pop(self):
        return self.parent

    def push(self, value):
        # returns a new StackNode which points to this
        return TreeStack(value, parent=self, value_to_str=self.value_to_str)

    def __iter__(self):
        stack = self
        while stack.parent is not None:
            yield stack.value
            stack = stack.parent
        yield stack.value

    def __str__(self):
        return "TreeStack(%s)" % ", ".join([self.value_to_str(x) for x in self])

    def __len__(self):
        return self._len

