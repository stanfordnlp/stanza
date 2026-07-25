import torch

from stanza.models.common.doc import Document
from stanza.models.coref.const import CorefResult
from stanza.pipeline.coref_processor import CorefProcessor, extract_text


def test_extract_coref_text_preserves_document_whitespace():
    document = Document([[
        {"text": "New"},
        {"text": "York"},
        {"text": "City"},
    ]])
    document.sentences[0].tokens[0].misc = "SpaceAfter=No"

    assert extract_text(document, 0, 0, 2) == "NewYork"
    assert extract_text(document, 0, 1, 3) == "York City"


def test_handle_zero_anaphora_creates_typed_empty_word_id():
    document = Document([[{"text": "Runs"}, {"text": "fast"}]])
    results = CorefResult(
        word_clusters=[[0, 1]],
        span_clusters=[[(0, 1), (1, 2)]],
        zero_scores=torch.tensor([[1.0], [-1.0]]),
    )
    processor = CorefProcessor.__new__(CorefProcessor)
    processor._use_zeros = True

    zero_nodes = processor._handle_zero_anaphora(
        document,
        results,
        sent_ids=[0, 0],
        word_pos=[0, 1],
    )

    assert zero_nodes == {(0, 0): (0, (0, 1))}
    assert [word.id for word in document.sentences[0].empty_words] == [(0, 1)]
    assert document.sentences[0].empty_words[0].text == "_"


def test_handle_zero_anaphora_can_be_disabled():
    document = Document([[{"text": "Runs"}]])
    results = CorefResult(
        word_clusters=[[0]],
        span_clusters=[[(0, 1)]],
        zero_scores=torch.tensor([1.0]),
    )
    processor = CorefProcessor.__new__(CorefProcessor)
    processor._use_zeros = False

    assert processor._handle_zero_anaphora(
        document,
        results,
        sent_ids=[0],
        word_pos=[0],
    ) == {}
    assert document.sentences[0].empty_words == []
