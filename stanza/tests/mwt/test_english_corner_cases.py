"""
Test a couple English MWT corner cases which might be more widely applicable to other MWT languages

- unknown English character doesn't result in bizarre splits
- Casing or CASING doesn't get lost in the dictionary lookup

In the English UD datasets, the MWT are composed exactly of the
subwords, so the MWT model should be chopping up the input text rather
than generating new text.

Furthermore, SHE'S and She's should be split "SHE 'S" and "She 's" respectively
"""

import pytest
import stanza

from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_mwt_unknown_char():
    pipeline = stanza.Pipeline(processors='tokenize,mwt', dir=TEST_MODELS_DIR, lang='en', download_method=None)

    mwt_trainer = pipeline.processors['mwt']._trainer

    assert mwt_trainer.args['force_exact_pieces']

    # find a letter 'i' which isn't in the training data
    # the MWT model should still recognize a possessive containing this letter
    assert "i" in mwt_trainer.vocab
    for letter in "ĩîíìī":
        if letter not in mwt_trainer.vocab:
            break
    else:
        raise AssertionError("Need to update the MWT test - all of the non-standard letters 'i' are now in the MWT vocab")

    word = "Jenn" + letter + "fer"
    possessive = word + "'s"
    text = "I wanna lick " + possessive + " antennae"
    doc = pipeline(text)
    assert doc.sentences[0].tokens[1].text == 'wanna'
    assert len(doc.sentences[0].tokens[1].words) == 2
    assert "".join(x.text for x in doc.sentences[0].tokens[1].words) == 'wanna'

    assert doc.sentences[0].tokens[3].text == possessive
    assert len(doc.sentences[0].tokens[3].words) == 2
    assert "".join(x.text for x in doc.sentences[0].tokens[3].words) == possessive


def test_english_mwt_casing():
    """
    Test that for a word where the lowercase split is known, the correct casing is still used

    Once upon a time, the logic used in the MWT expander would split
      SHE'S -> she 's

    which is a very surprising tokenization to people expecting
    the original text in the output document
    """
    pipeline = stanza.Pipeline(processors='tokenize,mwt', dir=TEST_MODELS_DIR, lang='en', download_method=None)

    mwt_trainer = pipeline.processors['mwt']._trainer
    for i in range(1, 20):
        # many test cases follow this pattern for some reason,
        # so we should proactively look for a test case which hasn't
        # made its way into the MWT dictionary
        unknown_name = "jennife" + "r" * i + "'s"
        if unknown_name not in mwt_trainer.expansion_dict and unknown_name.upper() not in mwt_trainer.expansion_dict:
            unknown_name = unknown_name.upper()
            break
    else:
        raise AssertionError("Need a new heuristic for the unknown word in the English MWT!")

    # this SHOULD show up in the expansion dict
    assert "she's" in mwt_trainer.expansion_dict, "Expected |she's| to be in the English MWT expansion dict... perhaps find a different test case"

    text = [x.text for x in pipeline("JENNIFER HAS NICE ANTENNAE").sentences[0].words]
    assert text == ['JENNIFER', 'HAS', 'NICE', 'ANTENNAE']

    text = [x.text for x in pipeline(unknown_name + " GOT NICE ANTENNAE").sentences[0].words]
    assert text == [unknown_name[:-2], "'S", 'GOT', 'NICE', 'ANTENNAE']

    text = [x.text for x in pipeline("SHE'S GOT NICE ANTENNAE").sentences[0].words]
    assert text == ['SHE', "'S", 'GOT', 'NICE', 'ANTENNAE']

    text = [x.text for x in pipeline("She's GOT NICE ANTENNAE").sentences[0].words]
    assert text == ['She', "'s", 'GOT', 'NICE', 'ANTENNAE']

