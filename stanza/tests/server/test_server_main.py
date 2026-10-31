"""
Test the key=value property parsing of the stanza.server.main script
"""

import pytest

from stanza.server.main import dictstr

pytestmark = [pytest.mark.travis, pytest.mark.client]

def test_dictstr_bool():
    assert dictstr("ssplit.isOneSentence=true") == ("ssplit.isOneSentence", True)
    assert dictstr("ssplit.isOneSentence=True") == ("ssplit.isOneSentence", True)
    assert dictstr("ssplit.isOneSentence=false") == ("ssplit.isOneSentence", False)
    assert dictstr("ssplit.isOneSentence=FALSE") == ("ssplit.isOneSentence", False)

def test_dictstr_numbers():
    assert dictstr("parse.maxlen=100") == ("parse.maxlen", 100)
    assert dictstr("parse.maxlen=-1") == ("parse.maxlen", -1)
    assert dictstr("ner.threshold=0.5") == ("ner.threshold", 0.5)
    assert dictstr("ner.threshold=.25") == ("ner.threshold", 0.25)

def test_dictstr_strings():
    # values which only start with a number stay strings instead of crashing int()
    assert dictstr("x=3rd") == ("x", "3rd")
    assert dictstr("x=1.5.2") == ("x", "1.5.2")
    assert dictstr("tokenize.language=en") == ("tokenize.language", "en")

def test_dictstr_value_with_equals():
    assert dictstr("tokensregex.rules=a=b") == ("tokensregex.rules", "a=b")
