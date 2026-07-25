import pytest

import stanza
from stanza.tests import TEST_MODELS_DIR


pytestmark = pytest.mark.pipeline


def test_process_many_strings():
    pipe = stanza.Pipeline(
        lang="en",
        dir=TEST_MODELS_DIR,
        processors="tokenize,pos",
        download_method=None,
    )

    texts = [
        "Barack Obama was born in Hawaii.",
        "He was elected president in 2008.",
    ]
    docs = pipe.process_many(texts)

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert all(doc.sentences for doc in docs)
    assert all(
        all(word.upos is not None for word in doc.sentences[0].words)
        for doc in docs
    )


def test_process_many_documents():
    base_pipe = stanza.Pipeline(
        lang="en",
        dir=TEST_MODELS_DIR,
        processors="tokenize",
        download_method=None,
    )
    docs = [base_pipe("This is a test."), base_pipe("Another one.")]

    pipe = stanza.Pipeline(
        lang="en",
        dir=TEST_MODELS_DIR,
        processors="tokenize,pos",
        download_method=None,
    )
    processed = pipe.process_many(docs)

    assert isinstance(processed, list)
    assert len(processed) == 2
    # Order should be preserved
    assert processed[0].sentences[0].text is not None
    assert processed[1].sentences[0].text is not None
    assert processed[0].sentences[0].text.startswith("This")
    assert processed[1].sentences[0].text.startswith("Another")
    assert all(
        all(word.upos is not None for word in doc.sentences[0].words)
        for doc in processed
    )


def test_stream_strings():
    pipe = stanza.Pipeline(
        lang="en",
        dir=TEST_MODELS_DIR,
        processors="tokenize,pos",
        download_method=None,
    )
    texts = iter([
        "This is a streamed document.",
        "This is another streamed document.",
    ])

    processed = list(pipe.stream(texts, batch_size=1))

    assert len(processed) == 2
    assert [sentence.sent_id for doc in processed for sentence in doc.sentences] == ["0", "1"]
    assert all(
        all(word.upos is not None for sentence in doc.sentences for word in sentence.words)
        for doc in processed
    )
