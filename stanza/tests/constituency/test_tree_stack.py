import pytest

from stanza.models.constituency.tree_stack import TreeStack

from stanza.tests import *

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_simple():
    stack = TreeStack(value=5)
    stack = stack.push(3)
    stack = stack.push(1)

    expected_values = [1, 3, 5]
    for value in expected_values:
        assert stack.value == value
        stack = stack.pop()
    assert stack is None

def test_iter():
    stack = TreeStack(value=5)
    stack = stack.push(3)
    stack = stack.push(1)

    stack_list = list(stack)
    assert list(stack) == [1, 3, 5]

def test_str():
    stack = TreeStack(value=5)
    stack = stack.push(3)
    stack = stack.push(1)

    assert str(stack) == "TreeStack(1, 3, 5)"

def test_str_variant():
    stack = TreeStack(value=5, value_to_str=lambda x: str(x+1))
    stack = stack.push(3)
    stack = stack.push(1)

    assert str(stack) == "TreeStack(2, 4, 6)"

def test_len():
    stack = TreeStack(value=5)
    assert len(stack) == 1

    stack = stack.push(3)
    stack = stack.push(1)
    assert len(stack) == 3

def test_long_len():
    """
    Original stack had a bug where this took exponential time...
    """
    stack = TreeStack(value=0)
    for i in range(1, 40):
        stack = stack.push(i)
    assert len(stack) == 40
