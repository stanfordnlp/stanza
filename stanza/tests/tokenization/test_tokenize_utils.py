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

def test_postprocessor_application():
    """
    Check that the postprocessor behaves correctly by applying the identity postprocessor and hoping that it does indeed return correctly.
    """

    good_tokenization = [['I', 'am', 'Joe.', '⭆⊱⇞', 'Hi', '.'], ["I'm", 'a', 'chicken', '.']]
    text = "I am Joe. ⭆⊱⇞ Hi. I'm a chicken."

    target_doc = [[{'id': (1,), 'text': 'I', 'start_char': 0, 'end_char': 1}, {'id': (2,), 'text': 'am', 'start_char': 2, 'end_char': 4}, {'id': (3,), 'text': 'Joe.', 'start_char': 5, 'end_char': 9}, {'id': (4,), 'text': '⭆⊱⇞', 'start_char': 10, 'end_char': 13}, {'id': (5,), 'text': 'Hi', 'start_char': 14, 'end_char': 16}, {'id': (6,), 'text': '.', 'start_char': 16, 'end_char': 17}], [{'id': (1,), 'text': "I'm", 'start_char': 18, 'end_char': 21}, {'id': (2,), 'text': 'a', 'start_char': 22, 'end_char': 23}, {'id': (3,), 'text': 'chicken', 'start_char': 24, 'end_char': 31}, {'id': (4,), 'text': '.', 'start_char': 31, 'end_char': 32}]]

    def postprocesor(_):
        return good_tokenization

    res = utils.postprocess_doc(target_doc, postprocesor, text)

    assert res == target_doc

def test_reassembly_indexing():
    """
    Check that the reassembly code counts the indicies correctly, and including OOV chars.
    """

    good_tokenization = [['I', 'am', 'Joe.', '⭆⊱⇞', 'Hi', '.'], ["I'm", 'a', 'chicken', '.']]
    good_mwts = [[False for _ in range(len(i))] for i in good_tokenization]

    text = "I am Joe. ⭆⊱⇞ Hi. I'm a chicken."

    target_doc = [[{'id': (1,), 'text': 'I', 'start_char': 0, 'end_char': 1}, {'id': (2,), 'text': 'am', 'start_char': 2, 'end_char': 4}, {'id': (3,), 'text': 'Joe.', 'start_char': 5, 'end_char': 9}, {'id': (4,), 'text': '⭆⊱⇞', 'start_char': 10, 'end_char': 13}, {'id': (5,), 'text': 'Hi', 'start_char': 14, 'end_char': 16}, {'id': (6,), 'text': '.', 'start_char': 16, 'end_char': 17}], [{'id': (1,), 'text': "I'm", 'start_char': 18, 'end_char': 21}, {'id': (2,), 'text': 'a', 'start_char': 22, 'end_char': 23}, {'id': (3,), 'text': 'chicken', 'start_char': 24, 'end_char': 31}, {'id': (4,), 'text': '.', 'start_char': 31, 'end_char': 32}]]

    res = utils.reassemble_doc_from_tokens(good_tokenization, good_mwts, text)

    assert res == target_doc

def test_reassembly_reference_failures():
    """
    Check that the reassembly code complains correctly when the user adds tokens that doesn't exist
    """

    bad_addition_tokenization = [['Joe', 'Smith', 'lives', 'in', 'Southern', 'California', '.']]
    bad_addition_mwts = [[False for _ in range(len(bad_addition_tokenization[0]))]]

    bad_inline_tokenization = [['Joe', 'Smith', 'lives', 'in', 'Californiaa', '.']]
    bad_inline_mwts = [[False for _ in range(len(bad_inline_tokenization[0]))]]

    good_tokenization = [['Joe', 'Smith', 'lives', 'in', 'California', '.']]
    good_mwts = [[False for _ in range(len(good_tokenization[0]))]]

    text = "Joe Smith lives in California."

    with pytest.raises(ValueError):
        utils.reassemble_doc_from_tokens(bad_addition_tokenization, bad_addition_mwts, text)

    with pytest.raises(ValueError):
        utils.reassemble_doc_from_tokens(bad_inline_tokenization, bad_inline_mwts, text)

    utils.reassemble_doc_from_tokens(good_tokenization, good_mwts, text)


