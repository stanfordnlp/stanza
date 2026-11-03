from stanza.models.common.doc import Document
from stanza.pipeline.coref_processor import extract_text


def test_extract_text_keeps_punctuation_spacing():
    """The spacing after a token is kept in Token.spaces_after, not in the misc field"""
    document = Document([
        [
            {"id": 1, "text": "He"},
            {"id": 2, "text": "was"},
            {"id": 3, "text": "born"},
            {"id": 4, "text": "on"},
            {"id": 5, "text": "July"},
            {"id": 6, "text": "4"},
            {"id": 7, "text": ","},
            {"id": 8, "text": "1976"},
        ],
    ])

    tokens = document.sentences[0].tokens
    # "July 4, 1976" - no space before the comma
    assert tokens[4].text == "July"
    assert tokens[5].text == "4"
    assert tokens[6].text == ","

    tokens[0].spaces_after = " "
    tokens[1].spaces_after = " "
    tokens[2].spaces_after = " "
    tokens[3].spaces_after = " "
    tokens[4].spaces_after = " "
    tokens[5].spaces_after = ""
    tokens[6].spaces_after = " "
    tokens[7].spaces_after = ""

    # words 5-8 are "July", "4", ",", "1976"
    assert extract_text(document, 0, 4, 8) == "July 4, 1976"


def test_extract_text_mwt_word_uses_misc():
    """A word taken from inside a multi-word token still uses its misc annotation"""
    document = Document([
        [
            {"id": (1, 2), "text": "cannot"},
            {"id": 1, "text": "can"},
            {"id": 2, "text": "not"},
            {"id": 3, "text": "stop"},
        ],
    ])

    words = document.sentences[0].words
    # word 1 is "can", which is inside the "cannot" token
    assert words[0].text == "can"

    document.sentences[0].tokens[0].spaces_after = " "
    document.sentences[0].tokens[1].spaces_after = " "

    assert extract_text(document, 0, 0, 2) == "cannot"
