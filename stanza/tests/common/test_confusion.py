"""
Test a couple simple confusion matrices and output formats
"""

from collections import defaultdict
import pytest

from stanza.utils.confusion import format_confusion, confusion_to_f1, confusion_to_macro_f1, confusion_to_weighted_f1

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

@pytest.fixture
def simple_confusion():
    confusion = defaultdict(lambda: defaultdict(int))
    confusion["B-ORG"]["B-ORG"] = 1
    confusion["B-ORG"]["B-PER"] = 1
    confusion["E-ORG"]["E-ORG"] = 1
    confusion["E-ORG"]["E-PER"] = 1
    confusion["O"]["O"] = 4
    return confusion

@pytest.fixture
def short_confusion():
    """
    Same thing, but with a short name.  This should not be sorted by entity type
    """
    confusion = defaultdict(lambda: defaultdict(int))
    confusion["A"]["B-ORG"] = 1
    confusion["B-ORG"]["B-PER"] = 1
    confusion["E-ORG"]["E-ORG"] = 1
    confusion["E-ORG"]["E-PER"] = 1
    confusion["O"]["O"] = 4
    return confusion

EXPECTED_SIMPLE_OUTPUT = """
     t/p      O B-ORG E-ORG B-PER E-PER
        O     4     0     0     0     0
    B-ORG     0     1     0     1     0
    E-ORG     0     0     1     0     1
    B-PER     0     0     0     0     0
    E-PER     0     0     0     0     0
"""[1:-1]  # don't want to strip

EXPECTED_SHORT_OUTPUT = """
     t/p      O     A B-ORG B-PER E-ORG E-PER
        O     4     0     0     0     0     0
        A     0     0     1     0     0     0
    B-ORG     0     0     0     1     0     0
    B-PER     0     0     0     0     0     0
    E-ORG     0     0     0     0     1     1
    E-PER     0     0     0     0     0     0
"""[1:-1]

def test_output(simple_confusion, short_confusion):
    assert EXPECTED_SIMPLE_OUTPUT == format_confusion(simple_confusion)
    assert EXPECTED_SHORT_OUTPUT == format_confusion(short_confusion)

def test_macro_f1(simple_confusion, short_confusion):
    assert confusion_to_macro_f1(simple_confusion) == pytest.approx(0.466666666666)
    assert confusion_to_macro_f1(short_confusion) == pytest.approx(0.277777777777)

def test_weighted_f1(simple_confusion, short_confusion):
    assert confusion_to_weighted_f1(simple_confusion) == pytest.approx(0.83333333)
    assert confusion_to_weighted_f1(short_confusion) == pytest.approx(0.66666666)

    assert confusion_to_weighted_f1(simple_confusion, exclude=["O"]) == pytest.approx(0.66666666)
    assert confusion_to_weighted_f1(short_confusion, exclude=["O"]) == pytest.approx(0.33333333)

