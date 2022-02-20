import pytest

from stanza.tests import *

from stanza.models.ner import trainer

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_fix_singleton_tags():
    TESTS = [
        (["O"], ["O"]),
        (["B-PER"], ["S-PER"]),
        (["B-PER", "I-PER"], ["B-PER", "E-PER"]),
        (["B-PER", "O", "B-PER"], ["S-PER", "O", "S-PER"]),
        (["B-PER", "B-PER", "I-PER"], ["S-PER", "B-PER", "E-PER"]),
        (["B-PER", "I-PER", "O", "B-PER"], ["B-PER", "E-PER", "O", "S-PER"]),
        (["B-PER", "B-PER", "I-PER", "B-PER"], ["S-PER", "B-PER", "E-PER", "S-PER"]),
        (["B-PER", "I-ORG", "O", "B-PER"], ["S-PER", "S-ORG", "O", "S-PER"]),
        (["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"], ["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"]),
        (["S-PER", "B-PER", "E-PER"], ["S-PER", "B-PER", "E-PER"]),
        (["E-PER"], ["S-PER"]),
        (["E-PER", "O", "E-PER"], ["S-PER", "O", "S-PER"]),
        (["B-PER", "E-ORG", "O", "B-PER"], ["S-PER", "S-ORG", "O", "S-PER"]),
        (["I-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"], ["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"]),
        (["B-PER", "I-PER", "I-PER", "O", "B-PER", "E-PER"], ["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"]),
        (["B-PER", "I-PER", "E-PER", "O", "I-PER", "E-PER"], ["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"]),
        (["B-PER", "I-PER", "E-PER", "O", "B-PER", "I-PER"], ["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"]),
        (["I-PER", "I-PER", "I-PER", "O", "I-PER", "I-PER"], ["B-PER", "I-PER", "E-PER", "O", "B-PER", "E-PER"]),
    ]
             
    for unfixed, expected in TESTS:
        assert trainer.fix_singleton_tags(unfixed) == expected, "Error converting {} to {}".format(unfixed, expected)
