"""
Tests for the speaker-tagging helpers in tokenize_processor.

These tests cover the pure-Python logic (parse_speaker_segments,
_check_empty_segments, _apply_speaker_to_sentences) and the
_set_up_model config parsing for speaker_delim.  They do not require a
loaded neural model or a real Pipeline.

Run with:
    python -m pytest tests/pipeline/test_tokenize_speaker.py -v
"""

import logging
import pytest

from stanza.pipeline.tokenize_processor import (
    TokenizeProcessor,
    _apply_speaker_to_sentences,
    _check_empty_segments,
    parse_speaker_segments,
)

pytestmark = pytest.mark.pipeline


# ---------------------------------------------------------------------------
# Minimal stubs so we can build lightweight Sentence/Token objects
# without loading model weights.
# ---------------------------------------------------------------------------

class _FakeToken:
    def __init__(self, start, end):
        self._start_char = start
        self._end_char = end
        self.spaces_after = ""
        self.spaces_before = ""
        self.words = []

    @property
    def start_char(self):
        return self._start_char

    @property
    def end_char(self):
        return self._end_char


class _FakeSentence:
    def __init__(self, tokens):
        self.tokens = tokens
        self.speaker = None


class _FakeDocument:
    def __init__(self, sentences):
        self.sentences = sentences
        self._text = None


def _make_doc(sentence_token_ranges):
    """
    Build a _FakeDocument whose sentences have tokens at the given
    (start, end) positions in stripped-text space.
    sentence_token_ranges: list of list of (start, end) pairs.
    """
    return _FakeDocument([
        _FakeSentence([_FakeToken(s, e) for s, e in token_ranges])
        for token_ranges in sentence_token_ranges
    ])


def _run_apply(result_doc, segments, original_text):
    """
    Call _apply_speaker_to_sentences and return (speakers, offsets) where
    offsets is a flat list of (start, end) for every token across all sentences.
    """
    _apply_speaker_to_sentences(result_doc, segments, original_text)
    speakers = [s.speaker for s in result_doc.sentences]
    offsets = [
        (tok._start_char, tok._end_char)
        for sent in result_doc.sentences
        for tok in sent.tokens
    ]
    return speakers, offsets


# ---------------------------------------------------------------------------
# parse_speaker_segments
# ---------------------------------------------------------------------------

def test_parse_no_tags_returns_single_unlabeled_segment():
    result = parse_speaker_segments("Hello world.", "<", ">")
    assert result == [(None, "Hello world.", 0)]

def test_parse_single_tag_at_start():
    result = parse_speaker_segments("<A>Hello world.", "<", ">")
    assert result == [
        (None, "", 0),
        ("A", "Hello world.", 3),
    ]

def test_parse_two_speakers():
    result = parse_speaker_segments("<A>No, you're going. <B>No, you are going.", "<", ">")
    assert result == [
        (None, "", 0),
        ("A", "No, you're going. ", 3),
        ("B", "No, you are going.", 24),
    ]

def test_parse_unlabeled_prefix():
    result = parse_speaker_segments("Narrator speaks. <A>Then Alice.", "<", ">")
    assert result == [
        (None, "Narrator speaks. ", 0),
        ("A", "Then Alice.", 20),
    ]

def test_parse_multi_char_label():
    result = parse_speaker_segments("<john>Hello. <amir>Hi there.", "<", ">")
    assert result == [
        (None, "", 0),
        ("john", "Hello. ", 6),
        ("amir", "Hi there.", 19),
    ]

def test_parse_curly_brace_delimiter():
    result = parse_speaker_segments("{A}Hello. {B}World.", "{", "}")
    assert result == [
        (None, "", 0),
        ("A", "Hello. ", 3),
        ("B", "World.", 13),
    ]

def test_parse_same_opener_closer_raises():
    with pytest.raises(ValueError):
        parse_speaker_segments("text", "|", "|")

def test_parse_consecutive_tags_empty_middle():
    # <A> produces an empty segment before <B> takes over
    result = parse_speaker_segments("<A><B>Hello.", "<", ">")
    assert result == [
        (None, "", 0),
        ("A", "", 3),
        ("B", "Hello.", 6),
    ]

def test_parse_trailing_tag_empty():
    result = parse_speaker_segments("<A>Hello.<B>", "<", ">")
    assert result == [
        (None, "", 0),
        ("A", "Hello.", 3),
        ("B", "", 12),
    ]

def test_parse_original_start_offsets():
    # Verify the third element of each tuple tracks the original string position.
    # <X> is [0,3), "abc" is [3,6), <Y> is [6,9), "defgh" is [9,14)
    result = parse_speaker_segments("<X>abc<Y>defgh", "<", ">")
    assert result == [
        (None, "", 0),
        ("X", "abc", 3),
        ("Y", "defgh", 9),
    ]

def test_parse_regex_special_chars_in_delimiter():
    # opener/closer must go through re.escape
    result = parse_speaker_segments("(A)Hello. (B)World.", "(", ")")
    assert result == [
        (None, "", 0),
        ("A", "Hello. ", 3),
        ("B", "World.", 13),
    ]


# ---------------------------------------------------------------------------
# _check_empty_segments
# ---------------------------------------------------------------------------

def test_check_no_warning_when_all_segments_non_empty(caplog):
    segments = [
        (None, "Intro.", 0),
        ("A", "Hello.", 7),
        ("B", "World.", 15),
    ]
    with caplog.at_level(logging.WARNING, logger="stanza"):
        _check_empty_segments(segments)
    assert not caplog.records

def test_check_warns_on_consecutive_tags(caplog):
    # <A><B> means A has an empty segment
    segments = [
        (None, "", 0),
        ("A", "", 3),
        ("B", "Hello.", 6),
    ]
    with caplog.at_level(logging.WARNING, logger="stanza"):
        _check_empty_segments(segments)
    assert any("A" in r.message for r in caplog.records)

def test_check_warns_on_trailing_empty_tag(caplog):
    segments = [
        (None, "", 0),
        ("A", "Hello.", 3),
        ("B", "", 12),
    ]
    with caplog.at_level(logging.WARNING, logger="stanza"):
        _check_empty_segments(segments)
    assert any("B" in r.message for r in caplog.records)

def test_check_no_warning_for_unlabeled_empty_prefix(caplog):
    # The unlabeled prefix is empty when the string starts with a tag — not a problem.
    segments = [
        (None, "", 0),
        ("A", "Hello.", 3),
    ]
    with caplog.at_level(logging.WARNING, logger="stanza"):
        _check_empty_segments(segments)
    assert not caplog.records


# ---------------------------------------------------------------------------
# _apply_speaker_to_sentences
# ---------------------------------------------------------------------------

def test_apply_single_segment_no_tags():
    # No tags: one unlabeled segment, offsets unchanged.
    original = "Hello world"
    segments = [(None, "Hello world", 0)]
    doc = _make_doc([[(0, 5), (6, 11)]])
    speakers, offsets = _run_apply(doc, segments, original)
    assert speakers == [None]
    assert offsets == [(0, 5), (6, 11)]

def test_apply_two_speakers_offset_correction():
    # Original:  "<A>No, you're going. <B>No, you are going."
    # Stripped:  "No, you're going. \n\nNo, you are going."
    # Segment A: original_start=3, stripped_start=0  → delta=+3
    # Segment B: original_start=24, stripped_start=20 → delta=+4
    original = "<A>No, you're going. <B>No, you are going."
    segments = [
        ("A", "No, you're going. ", 3),
        ("B", "No, you are going.", 24),
    ]
    doc = _make_doc([[(0, 18)], [(20, 38)]])
    speakers, offsets = _run_apply(doc, segments, original)
    assert speakers == ["A", "B"]
    assert offsets[0] == (3, 21)   # 0+3, 18+3
    assert offsets[1] == (24, 42)  # 20+4, 38+4

def test_apply_unlabeled_prefix_gets_none_speaker():
    # Original: "Intro. <A>Alice speaks."
    # Stripped: "Intro. \n\nAlice speaks."
    # Segment None: original_start=0, stripped_start=0  → delta=0
    # Segment A:    original_start=11, stripped_start=9  → delta=+2
    original = "Intro. <A>Alice speaks."
    segments = [
        (None, "Intro. ", 0),
        ("A", "Alice speaks.", 11),
    ]
    doc = _make_doc([[(0, 7)], [(9, 22)]])
    speakers, offsets = _run_apply(doc, segments, original)
    assert speakers == [None, "A"]
    assert offsets[0] == (0, 7)
    assert offsets[1] == (11, 24)  # 9+2, 22+2

def test_apply_multiple_sentences_per_segment():
    # Speaker A produces two sentences; speaker B produces one.
    original = "<A>Hello. Goodbye. <B>Hi."
    segments = [
        ("A", "Hello. Goodbye. ", 3),
        ("B", "Hi.", 22),
    ]
    doc = _make_doc([[(0, 7)], [(8, 16)], [(18, 21)]])
    speakers, _ = _run_apply(doc, segments, original)
    assert speakers == ["A", "A", "B"]

def test_apply_spaces_after_reflect_original_text():
    # spaces_after between sentences should contain the tag text from the
    # original string, not the "\n\n" from the stripped-and-joined text.
    # original = "<A>Hi. <B>Bye."
    #   <A> at [0,3), "Hi. " at [3,7), <B> at [7,10), "Bye." at [10,14)
    # Stripped: "Hi. \n\nBye."  →  seg A [0,4), seg B [6,10)
    # After correction: tok_A=(3,7), tok_B=(10,14)
    # spaces_after of tok_A = original[7:10] = "<B>"
    # spaces_before of tok_A = original[:3]  = "<A>"
    original = "<A>Hi. <B>Bye."
    segments = [("A", "Hi. ", 3), ("B", "Bye.", 10)]
    tok_A = _FakeToken(0, 4)
    tok_B = _FakeToken(6, 10)
    doc = _FakeDocument([_FakeSentence([tok_A]), _FakeSentence([tok_B])])
    _apply_speaker_to_sentences(doc, segments, original)
    assert tok_A._start_char == 3
    assert tok_A._end_char == 7
    assert tok_B._start_char == 10
    assert tok_B._end_char == 14
    assert tok_A.spaces_after == "<B>"
    assert tok_A.spaces_before == "<A>"
    assert tok_B.spaces_after == ""
    assert tok_B.spaces_before == ""


# ---------------------------------------------------------------------------
# Integration: parse + apply round-trip (no model needed)
# ---------------------------------------------------------------------------

def test_round_trip_two_speakers():
    """
    Simulate _process_speaker_tagged_text end-to-end without a model by
    manually constructing the post-tokenisation Document and verifying
    speakers and char offsets.
    """
    original = "<A>No, you're going. <B>No, you are going."
    segments = parse_speaker_segments(original, "<", ">")
    non_empty = [(spk, seg, orig) for spk, seg, orig in segments if seg]

    joined = "\n\n".join(seg for _, seg, _ in non_empty)
    assert joined == "No, you're going. \n\nNo, you are going."

    # Simulate tokeniser output: one token spanning each full segment.
    # Segment A occupies [0,18) in joined; segment B occupies [20,38).
    tok0 = _FakeToken(0, 18)
    tok1 = _FakeToken(20, 38)
    doc = _FakeDocument([_FakeSentence([tok0]), _FakeSentence([tok1])])

    _apply_speaker_to_sentences(doc, non_empty, original)

    assert doc.sentences[0].speaker == "A"
    assert doc.sentences[1].speaker == "B"
    # Segment A: delta = original_start(3) - stripped_start(0) = +3
    assert tok0._start_char == 3
    assert tok0._end_char == 21
    # Segment B: delta = original_start(24) - stripped_start(20) = +4
    assert tok1._start_char == 24
    assert tok1._end_char == 42
    # spaces_after of tok0 = original[21:24] = "<B>"
    # ("<A>"=3 chars, "No, you're going. "=18 chars → tok0 ends at 21;
    #  "<B>" occupies [21,24); tok1 starts at 24)
    assert tok0.spaces_after == "<B>"
    assert tok1.spaces_after == ""


# ---------------------------------------------------------------------------
# _set_up_model: speaker_delim config parsing
# ---------------------------------------------------------------------------

def _make_processor(config):
    """
    Instantiate a TokenizeProcessor without going through UDProcessor.__init__
    (which would try to load a model), then call _set_up_model directly with
    the given config.  pretokenized=True short-circuits trainer setup so only
    the speaker_delim block runs.
    """
    proc = TokenizeProcessor.__new__(TokenizeProcessor)
    proc._set_up_model(config, pipeline=None, device="cpu")
    return proc

def test_speaker_delim_angle_brackets():
    proc = _make_processor({"pretokenized": True, "speaker_delim": "<>"})
    assert proc._speaker_opener == "<"
    assert proc._speaker_closer == ">"

def test_speaker_delim_curly_braces():
    proc = _make_processor({"pretokenized": True, "speaker_delim": "{}"})
    assert proc._speaker_opener == "{"
    assert proc._speaker_closer == "}"

def test_speaker_delim_absent_gives_none():
    proc = _make_processor({"pretokenized": True})
    assert proc._speaker_opener is None
    assert proc._speaker_closer is None

def test_speaker_delim_wrong_length_raises():
    with pytest.raises(ValueError):
        _make_processor({"pretokenized": True, "speaker_delim": "<->"})  # 3 chars

def test_speaker_delim_single_char_raises():
    with pytest.raises(ValueError):
        _make_processor({"pretokenized": True, "speaker_delim": "<"})  # 1 char

def test_speaker_delim_same_opener_closer_does_not_raise_at_init():
    # The opener==closer check happens in parse_speaker_segments at process()
    # time, not at _set_up_model time.  Verify that init succeeds so that
    # callers get a clear error at the point of use rather than at pipeline build.
    proc = _make_processor({"pretokenized": True, "speaker_delim": "||"})
    assert proc._speaker_opener == "|"
    assert proc._speaker_closer == "|"
