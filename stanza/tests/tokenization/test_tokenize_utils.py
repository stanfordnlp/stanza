"""
Very simple test of the sentence slicing by <PAD> tags

TODO: could add a bunch more simple tests for the tokenization utils
"""

import pytest
import stanza

from stanza import Pipeline
from stanza.tests import *
from stanza.models.common import doc
from stanza.models.tokenization import data
from stanza.models.tokenization import utils

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_find_spans():
    """
    Test various raw -> span manipulations
    """
    raw = ['u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(0, 14)]

    raw = ['u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l', '<PAD>']
    assert utils.find_spans(raw) == [(0, 14)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l', '<PAD>']
    assert utils.find_spans(raw) == [(1, 15)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', ' ', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(1, 15)]

    raw = ['<PAD>', 'u', 'n', 'b', 'a', 'n', '<PAD>', 'm', 'o', 'x', ' ', 'o', 'p', 'a', 'l']
    assert utils.find_spans(raw) == [(1, 6), (7, 15)]

def check_offsets(doc, expected_offsets):
    """
    Compare the start_char and end_char of the tokens in the doc with the given list of list of offsets
    """
    assert len(doc.sentences) == len(expected_offsets)
    for sentence, offsets in zip(doc.sentences, expected_offsets):
        assert len(sentence.tokens) == len(offsets)
        for token, offset in zip(sentence.tokens, offsets):
            assert token.start_char == offset[0]
            assert token.end_char == offset[1]

def test_match_tokens_with_text():
    """
    Test the conversion of pretokenized text to Document
    """
    doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisatest")
    expected_offsets = [[(0, 4), (4, 6), (6, 7), (7, 11)]]
    check_offsets(doc, expected_offsets)

    doc = utils.match_tokens_with_text([["This", "is", "a", "test"], ["unban", "mox", "opal", "!"]], "Thisisatest  unban mox  opal!")
    expected_offsets = [[(0, 4), (4, 6), (6, 7), (7, 11)],
                        [(13, 18), (19, 22), (24, 28), (28, 29)]]
    check_offsets(doc, expected_offsets)

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisatestttt")

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "is", "a", "test"]], "Thisisates")

    with pytest.raises(ValueError):
        doc = utils.match_tokens_with_text([["This", "iz", "a", "test"]], "Thisisatest")

def test_long_paragraph():
    """
    Test the tokenizer's capacity to break text up into smaller chunks
    """
    pipeline = Pipeline("en", dir=TEST_MODELS_DIR, processors="tokenize")
    tokenizer = pipeline.processors['tokenize']

    raw_text = "TIL not to ask a date to dress up as Smurfette on a first date.  " * 100

    # run a test to make sure the chunk operation is called
    # if not, the test isn't actually testing what we need to test
    batches = data.DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    batches.advance_old_batch = None
    with pytest.raises(TypeError):
        _, _, _, document = utils.output_predictions(None, tokenizer.trainer, batches, tokenizer.vocab, None, 3000,
                                                     orig_text=raw_text,
                                                     no_ssplit=tokenizer.config.get('no_ssplit', False))

    # a new DataLoader should not be crippled as the above one was
    batches = data.DataLoader(tokenizer.config, input_text=raw_text, vocab=tokenizer.vocab, evaluation=True, dictionary=tokenizer.trainer.dictionary)
    _, _, _, document = utils.output_predictions(None, tokenizer.trainer, batches, tokenizer.vocab, None, 3000,
                                                 orig_text=raw_text,
                                                 no_ssplit=tokenizer.config.get('no_ssplit', False))

    document = doc.Document(document, raw_text)
    assert len(document.sentences) == 100
