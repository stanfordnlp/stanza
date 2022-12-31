import pytest

import stanza
from stanza.tests import *
from stanza.models.common.doc import Document, ID, TEXT, NER, CONSTITUENCY

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

@pytest.fixture
def sentences_dict():
    return [[{ID: 1, TEXT: "unban"},
             {ID: 2, TEXT: "mox"},
             {ID: 3, TEXT: "opal"}],
            [{ID: 4, TEXT: "ban"},
             {ID: 5, TEXT: "Lurrus"}]]

@pytest.fixture
def doc(sentences_dict):
    doc = Document(sentences_dict)
    return doc

def test_basic_values(doc, sentences_dict):
    """
    Test that sentences & token text are properly set when constructing a doc
    """
    assert len(doc.sentences) == len(sentences_dict)

    for sentence, raw_sentence in zip(doc.sentences, sentences_dict):
        assert sentence.doc == doc
        assert len(sentence.tokens) == len(raw_sentence)
        for token, raw_token in zip(sentence.tokens, raw_sentence):
            assert token.text == raw_token[TEXT]

def test_set_sentence(doc):
    """
    Test setting a field on the sentences themselves
    """
    doc.set(fields="sentiment",
            contents=["4", "0"],
            to_sentence=True)

    assert doc.sentences[0].sentiment == "4"
    assert doc.sentences[1].sentiment == "0"

def test_set_tokens(doc):
    """
    Test setting values on tokens
    """
    ner_contents = ["O", "ARTIFACT", "ARTIFACT", "O", "CAT"]
    doc.set(fields=NER,
            contents=ner_contents,
            to_token=True)

    result = doc.get(NER, from_token=True)
    assert result == ner_contents


def test_constituency_comment(doc):
    """
    Test that setting the constituency tree on a doc sets the constituency comment
    """
    for sentence in doc.sentences:
        assert len([x for x in sentence.comments if x.startswith("# constituency")]) == 0

    # currently nothing is checking that the items are actually trees
    trees = ["asdf", "zzzz"]
    doc.set(fields=CONSTITUENCY,
            contents=trees,
            to_sentence=True)

    for sentence, expected in zip(doc.sentences, trees):
        constituency_comments = [x for x in sentence.comments if x.startswith("# constituency")]
        assert len(constituency_comments) == 1
        assert constituency_comments[0].endswith(expected)

    # Test that if we replace the trees with an updated tree, the comment is also replaced
    trees = ["zzzz", "asdf"]
    doc.set(fields=CONSTITUENCY,
            contents=trees,
            to_sentence=True)

    for sentence, expected in zip(doc.sentences, trees):
        constituency_comments = [x for x in sentence.comments if x.startswith("# constituency")]
        assert len(constituency_comments) == 1
        assert constituency_comments[0].endswith(expected)

