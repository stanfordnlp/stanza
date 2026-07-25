import json

import pytest

import stanza
from stanza.tests import *
from stanza.models.common.doc import (
    CONSTITUENCY,
    DEPREL,
    DEPS,
    END_CHAR,
    HEAD,
    ID,
    MORPHEMES,
    MULTI_NER,
    NER,
    SENTIMENT,
    START_CHAR,
    TEXT,
    Document,
    EmptyWordEntry,
    TokenEntry,
    _require_empty_word_entry,
    _require_token_entry,
)

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


def test_default_token_entry_validation_checks_field_types():
    entry = {TEXT: "valid", HEAD: 0, MULTI_NER: ["O"]}
    assert _require_token_entry(entry, "test") == entry

    with pytest.raises(ValueError, match="valid default token dictionary"):
        _require_token_entry({TEXT: "invalid", HEAD: "root"}, "test")


def test_empty_word_entries_require_two_component_ids():
    with pytest.raises(ValueError, match="two-component empty-word ID"):
        _require_empty_word_entry(
            {ID: (1,), TEXT: "not-empty"},
            "test",
        )


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

def test_sentiment_comment(doc):
    """
    Test that setting the sentiment on a doc sets the sentiment comment
    """
    for sentence in doc.sentences:
        assert len([x for x in sentence.comments if x.startswith("# sentiment")]) == 0

    # currently nothing is checking that the items are actually trees
    sentiments = ["1", "2"]
    doc.set(fields=SENTIMENT,
            contents=sentiments,
            to_sentence=True)

    for sentence, expected in zip(doc.sentences, sentiments):
        sentiment_comments = [x for x in sentence.comments if x.startswith("# sentiment")]
        assert len(sentiment_comments) == 1
        assert sentiment_comments[0].endswith(expected)

    # Test that if we replace the trees with an updated tree, the comment is also replaced
    sentiments = ["3", "4"]
    doc.set(fields=SENTIMENT,
            contents=sentiments,
            to_sentence=True)

    for sentence, expected in zip(doc.sentences, sentiments):
        sentiment_comments = [x for x in sentence.comments if x.startswith("# sentiment")]
        assert len(sentiment_comments) == 1
        assert sentiment_comments[0].endswith(expected)

def test_sent_id_comment(doc):
    """
    Test that setting the sent_id on a sentence sets the sentiment comment
    """
    for sent_idx, sentence in enumerate(doc.sentences):
        assert len([x for x in sentence.comments if x.startswith("# sent_id")]) == 1
        assert sentence.sent_id == "%d" % sent_idx
    doc.sentences[0].sent_id = "foo"
    assert doc.sentences[0].sent_id == "foo"
    assert len([x for x in doc.sentences[0].comments if x.startswith("# sent_id")]) == 1
    assert "# sent_id = foo" in doc.sentences[0].comments

    doc.reindex_sentences(10)
    for sent_idx, sentence in enumerate(doc.sentences):
        assert sentence.sent_id == "%d" % (sent_idx + 10)
        assert len([x for x in doc.sentences[0].comments if x.startswith("# sent_id")]) == 1
        assert "# sent_id = %d" % (sent_idx + 10) in sentence.comments

    doc.sentences[0].add_comment("# sent_id = bar")
    assert doc.sentences[0].sent_id == "bar"
    assert "# sent_id = bar" in doc.sentences[0].comments
    assert len([x for x in doc.sentences[0].comments if x.startswith("# sent_id")]) == 1

def test_doc_id_comment(doc):
    """
    Test that setting the doc_id on a sentence sets the document comment
    """
    assert doc.sentences[0].doc_id is None
    assert len([x for x in doc.sentences[0].comments if x.startswith("# doc_id")]) == 0

    doc.sentences[0].doc_id = "foo"
    assert len([x for x in doc.sentences[0].comments if x.startswith("# doc_id")]) == 1
    assert "# doc_id = foo" in doc.sentences[0].comments
    assert doc.sentences[0].doc_id == "foo"

    doc.sentences[0].add_comment("# doc_id = bar")
    assert len([x for x in doc.sentences[0].comments if x.startswith("# doc_id")]) == 1
    assert doc.sentences[0].doc_id == "bar"

@pytest.fixture(scope="module")
def pipeline():
    return stanza.Pipeline(dir=TEST_MODELS_DIR, download_method=None)

def test_serialized(pipeline):
    """
    Brief test of the serialized format

    Checks that NER entities are correctly set.
    Also checks that constituency & sentiment are set on the sentences.
    """
    text = "John Bauer works at Stanford"
    doc = pipeline(text)
    assert len(doc.ents) == 2
    serialized = doc.to_serialized()
    doc2 = Document.from_serialized(serialized)
    assert len(doc2.sentences) == 1
    assert len(doc2.ents) == 2
    assert doc.sentences[0].constituency == doc2.sentences[0].constituency
    assert doc.sentences[0].sentiment == doc2.sentences[0].sentiment


def test_json_serialization_schema_and_round_trip():
    sentences: list[list[TokenEntry]] = [[
        {
            ID: 1,
            TEXT: "A",
            HEAD: 0,
            DEPREL: "root",
            DEPS: "0:root",
            START_CHAR: 0,
            END_CHAR: 1,
            MULTI_NER: ["O", "O"],
        },
        {
            ID: 2,
            TEXT: "test",
            HEAD: 1,
            DEPREL: "dep",
            DEPS: [("1", "dep")],
            START_CHAR: 2,
            END_CHAR: 6,
            MORPHEMES: ["test"],
        },
    ]]
    empty_sentences: list[list[EmptyWordEntry]] = [[{
        ID: (1, 1),
        TEXT: "pro",
        DEPS: [("1", "dep")],
    }]]
    doc = Document(
        sentences,
        text="A test",
        comments=[["# sent_id = serialization-test"]],
        empty_sentences=empty_sentences,
    )

    serialized = doc.to_serialized()
    payload = json.loads(serialized)

    assert set(payload) == {
        "text",
        "sentences",
        "comments",
        "empty_sentences",
    }
    assert payload["text"] == "A test"
    assert payload["sentences"][0][0][ID] == 1
    assert payload["sentences"][0][0][DEPS] == "0:root"
    assert payload["sentences"][0][0][MULTI_NER] == ["O", "O"]
    assert payload["sentences"][0][1][MORPHEMES] == ["test"]
    assert payload["empty_sentences"][0][0][ID] == [1, 1]
    assert payload["empty_sentences"][0][0][DEPS] == "1:dep"

    restored = Document.from_serialized(serialized)
    assert restored.to_dict() == doc.to_dict()
    assert restored.sentences[0].dependencies_string() == (
        "('A', 0, 'root')\n('test', 1, 'dep')"
    )
    assert restored.sentences[0].empty_words[0].deps == "1:dep"


def test_fake_dependencies_string(doc):
    sentence = doc.sentences[0]
    sentence.build_fake_dependencies()

    assert sentence.dependencies_string() == (
        "('unban', 0, 'root')\n"
        "('mox', 1, 'dep')\n"
        "('opal', 2, 'dep')"
    )


def test_sequence_field_names(doc):
    doc.set((NER,), ["O", "B-ITEM", "E-ITEM", "O", "S-CAT"], to_token=True)

    assert doc.sentences[0].tokens[0].to_dict((ID,)) == [{ID: 1}]
    assert doc.get((NER,), from_token=True) == [
        "O",
        "B-ITEM",
        "E-ITEM",
        "O",
        "S-CAT",
    ]
