"""
Test that an unknown English character doesn't result in bizarre splits

In the English UD datasets, the MWT are composed exactly of the
subwords, so the MWT model should be chopping up the input text rather
than generating new text.
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
