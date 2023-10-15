"""
Small test of loading the Arabic pipeline

The main goal is to check that nothing goes wrong with RtL languages,
but incidentally this would have caught a bug where the xpos tags
were split into individual pieces instead of reassembled as expected
"""

import pytest
import stanza

from stanza.tests import TEST_MODELS_DIR

pytestmark = pytest.mark.pipeline

def test_arabic_pos_pipeline():
    pipe = stanza.Pipeline(**{'processors': 'tokenize,pos', 'dir': TEST_MODELS_DIR, 'download_method': None, 'lang': 'ar'})
    text = "ولم يتم اعتقال احد بحسب المتحدث باسم الشرطة."

    doc = pipe(text)
    # the first token translates to "and not", seems common enough
    # that we should be able to rely on it having a stable MWT and tag

    assert len(doc.sentences) == 1
    assert doc.sentences[0].tokens[0].text == "ولم"
    assert doc.sentences[0].words[0].xpos == "C---------"
    assert doc.sentences[0].words[1].xpos == "F---------"
