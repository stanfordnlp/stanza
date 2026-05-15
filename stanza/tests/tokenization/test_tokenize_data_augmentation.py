"""
Tests for the augmentation and pre-counting functions in tokenization/data.py

Stochastic augmentation methods are tested by running many independent trials
and asserting that the expected outcome occurs at least once (or always, when
the property must hold for every non-None result).  With 200 trials the
probability of any one of the four spacing styles being missed is below
1 in 10^24, which is acceptable.

The other augmentation probs all default to 0.0 in FAKE_PROPERTIES, so only
the augmentation under test is active in each test.

Label encoding reminder (see data.py module docstring):
  0  continuation      – character is inside a token
  1  word end          – last character of a token
  2  sentence end      – last character of the final token in a sentence
  3  MWT end           – last character of a multi-word token
  4  MWT + sentence end
"""

import pytest
import random
import os
import tempfile

from stanza.tests import TEST_WORKING_DIR
from stanza.models.tokenization.data import (
    DataLoader,
    MID_SENT_AUGMENT_PAIRS,
    build_move_punct_set,
    build_known_mwt,
)
from stanza.models.tokenization.vocab import Vocab

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_PROPERTIES = {
    "lang": "en",
    'feat_funcs': ("space_before", "capitalized"),
    'max_seqlen': 300,
    'use_dictionary': False,
    # All augmentation probs default to 0.0; individual tests set the one
    # they are exercising to 1.0 via extra_args so that only that augmentation
    # is active, preventing interference between augmentations.
}

# "Hello, world."
#  H e l l o ,   w o r l d  .
#  0 0 0 0 1 1 0 0 0 0 0 1  2
HELLO_TEXT   = "Hello, world."
HELLO_LABELS = "0000110000012"

# "Hello , world."  (space before comma — eligible for move_punct_back)
#  H e l l o   ,   w o r l d  .
#  0 0 0 0 1 0 1 0 0 0 0 0 1  2
SPACED_COMMA_TEXT   = "Hello , world."
SPACED_COMMA_LABELS = "00001010000012"


def write_and_load(raw_text, labels, extra_args=None):
    """Write text+labels to temp files and return a DataLoader."""
    args = dict(FAKE_PROPERTIES)
    if extra_args:
        args.update(extra_args)
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        txt_path = os.path.join(test_dir, "text.txt")
        lbl_path = os.path.join(test_dir, "labels.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(labels)
        data = DataLoader(args=args, input_files={'txt': txt_path, 'label': lbl_path})
    return data


def run_trials(fn, n=200):
    """
    Call fn() n times, collecting all non-None results.
    Returns the list of results.
    """
    return [r for _ in range(n) if (r := fn()) is not None]


# ---------------------------------------------------------------------------
# augment_vocab
# ---------------------------------------------------------------------------

class TestAugmentVocab:

    def _make(self, sentences):
        vocab = Vocab(sentences, "en")
        return vocab, sentences

    def test_final_existing_absent_replacement(self):
        """source present at sentence end, target absent -> returns True and adds to vocab."""
        data = [[('H', 0), ('i', 1), ('?', 2)]]
        vocab, data = self._make(data)
        assert '?' in vocab
        assert '！' not in vocab
        assert DataLoader.augment_vocab(vocab, data, '?', '！', final=True) is True
        assert '！' in vocab

    def test_final_missing_source(self):
        """source absent from vocab -> returns False."""
        data = [[('H', 0), ('i', 1), ('?', 2)]]
        vocab, data = self._make(data)
        assert DataLoader.augment_vocab(vocab, data, '！', '?', final=True) is False

    def test_final_replacement_already_present(self):
        """target already in data -> returns False."""
        data = [[('H', 0), ('i', 1), ('?', 2)],
                [('B', 0), ('y', 1), ('！', 2)]]
        vocab, data = self._make(data)
        assert DataLoader.augment_vocab(vocab, data, '?', '！', final=True) is False

    def test_final_source_not_at_end(self):
        """source present mid-sentence only, never finally -> returns False."""
        data = [[('H',0),('i',0),(',',1),(' ',0),('y',0),('o',0),('u',1),('.',2)]]
        vocab, data = self._make(data)
        assert DataLoader.augment_vocab(vocab, data, ',', '\u2013', final=True) is False

    def test_not_final_finds_mid_sentence(self):
        """final=False counts mid-sentence occurrences."""
        data = [[('H',0),('i',0),(',',1),(' ',0),('y',0),('o',0),('u',1),('.',2)]]
        vocab, data = self._make(data)
        assert DataLoader.augment_vocab(vocab, data, ',', '\u2013', final=False) is True
        assert '\u2013' in vocab

    def test_not_final_includes_all_positions(self):
        """final=False counts all positions including the final character."""
        # '.' appears only as the final character; final=False still finds it
        data = [[('H', 0), ('i', 1), ('.', 2)]]
        vocab, data = self._make(data)
        assert DataLoader.augment_vocab(vocab, data, '.', '\u2014', final=False) is True


# ---------------------------------------------------------------------------
# build_move_punct_set
# ---------------------------------------------------------------------------

class TestBuildMovePunctSet:
    def test_comma_eligible_when_space_separated(self):
        """A space-separated comma following a non-digit word should be eligible."""
        chunk = [('H',0),('e',0),('l',0),('l',0),('o',1),(' ',0),(',',1),(' ',0),
                 ('w',0),('o',0),('r',0),('l',0),('d',1),(' ',0),('.',2)]
        result = build_move_punct_set([chunk], move_back_prob=0.02)
        assert ',' in result

    def test_comma_ineligible_when_already_attached(self):
        """A comma already attached to the preceding word should be removed from the set."""
        chunk = [('H',0),('e',0),('l',0),('l',0),('o',0),(',',1),(' ',0),
                 ('w',0),('o',0),('r',0),('l',0),('d',1),('.',2)]
        result = build_move_punct_set([chunk], move_back_prob=0.02)
        assert ',' not in result

# ---------------------------------------------------------------------------
# build_known_mwt
# ---------------------------------------------------------------------------

class TestBuildKnownMwt:

    def test_finds_known_mwt(self):
        """An MWT present in mwt_expansions labelled 3 should be found."""
        chunk = [('a',0),('l',3),(' ',0),('B',0),('a',0),('n',0),('c',0),('o',2)]
        result = build_known_mwt([chunk], {"al": ["a", "el"]})
        assert "al" in result

    def test_ignores_mwt_not_in_expansions(self):
        """An MWT label with no expansion entry should be ignored."""
        chunk = [('d',0),('e',0),('l',3),(' ',0),('B',0),('a',0),('n',0),('c',0),('o',2)]
        result = build_known_mwt([chunk], {})
        assert "del" not in result

    def test_ignores_three_way_mwt(self):
        """MWTs expanding to more than 2 words are not supported and should be ignored."""
        chunk = [('x',0),('y',0),('z',3),(' ',0),('f',0),('o',0),('o',2)]
        result = build_known_mwt([chunk], {"xyz": ["x", "y", "z"]})
        assert "xyz" not in result


# ---------------------------------------------------------------------------
# build_mid_sent_augmentations
# ---------------------------------------------------------------------------

class TestBuildMidSentAugmentations:

    def test_comma_to_dash_activated(self):
        """Comma present mid-sentence, en dash absent -> substitution activated."""
        data = [[('H',0),('e',0),('l',0),('l',0),('o',0),(',',1),(' ',0),
                 ('w',0),('o',0),('r',0),('l',0),('d',1),('.',2)]]
        vocab = Vocab(data, "en")
        result = DataLoader.build_mid_sent_augmentations(vocab, data, MID_SENT_AUGMENT_PAIRS)
        assert ',' in result
        assert '\u2013' in result[','] or '\u2014' in result[',']

    def test_dash_present_blocks_activation(self):
        """En dash already in data -> comma->en dash substitution should not activate,
        even when commas are also present."""
        data = [[('H',0),('e',0),('l',0),('l',0),('o',0),(',',1),(' ',0),
                 ('w',0),('o',0),('r',0),('l',0),('d',1),(' ',0),('\u2013',1),
                 ('f',0),('o',0),('o',1),('.',2)]]
        vocab = Vocab(data, "en")
        result = DataLoader.build_mid_sent_augmentations(vocab, data, [(',', '\u2013')])
        assert '\u2013' not in result.get(',', [])

    def test_empty_when_no_comma(self):
        """No comma in data -> nothing to augment."""
        data = [[('H',0),('i',1),('.',2)]]
        vocab = Vocab(data, "en")
        result = DataLoader.build_mid_sent_augmentations(vocab, data, MID_SENT_AUGMENT_PAIRS)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# augment_final_punct
# ---------------------------------------------------------------------------

class TestAugmentFinalPunct:

    def _loader(self):
        # "Hi?" -> H i ?  labels 0 1 2
        # augment_final_punct_prob=1.0 activates the vocab check in __init__;
        # all other augmentation probs remain at 0.0
        return write_and_load("Hi?", "012", extra_args={'augment_final_punct_prob': 1.0})

    def test_replaces_final_punct(self):
        """augment_final_punct should always swap '?' for a fullwidth variant."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.augment_final_punct(sentence))
        assert len(results) > 0, "augment_final_punct never returned a result"
        fullwidth = {'？', '︖', '﹖', '⁇'}
        for result in results:
            assert result[0][3][-1] in fullwidth, (
                f"unexpected final character: {result[0][3][-1]!r}"
            )

    def test_no_eligible_punct_returns_none(self):
        """augment_final_punct returns None when the augmentations dict is empty."""
        loader = write_and_load("Hi.", "012", extra_args={'augment_final_punct_prob': 1.0})
        loader.augmentations = {}
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.augment_final_punct(sentence))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# augment_mid_sent_punct
# ---------------------------------------------------------------------------

class TestAugmentMidSentPunct:

    def _loader(self):
        # augment_mid_punct_prob=1.0 activates vocab check; other probs at 0.0
        loader = write_and_load(HELLO_TEXT, HELLO_LABELS,
                                extra_args={'augment_mid_punct_prob': 1.0})
        loader.mid_sent_augmentations = DataLoader.build_mid_sent_augmentations(
            loader.vocab, loader.data, MID_SENT_AUGMENT_PAIRS)
        return loader

    def test_comma_replaced_by_dash(self):
        """The comma should always be replaced by a dash (never left as a comma)."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.augment_mid_sent_punct(sentence))
        assert len(results) > 0, "augment_mid_sent_punct never returned a result"
        for result in results:
            chars = result[0][3]
            assert ',' not in chars
            assert '\u2013' in chars or '\u2014' in chars

    def test_dash_is_own_token(self):
        """The replacement dash should always have a non-zero label (own token)."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.augment_mid_sent_punct(sentence))
        assert len(results) > 0
        for result in results:
            new_sentence = result[0]
            for char, label in zip(new_sentence[3], new_sentence[1]):
                if char in ('\u2013', '\u2014'):
                    assert label != 0, "dash should not have continuation label"

    def test_comma_in_number_not_replaced(self):
        """A comma with label 0 (inside a number token) should never be replaced."""
        # "1,000." -> 1 , 0 0 0 .  labels  0 0 0 0 1 2
        loader = write_and_load("1,000.", "000012",
                                extra_args={'augment_mid_punct_prob': 1.0})
        loader.mid_sent_augmentations = DataLoader.build_mid_sent_augmentations(
            loader.vocab, loader.data, MID_SENT_AUGMENT_PAIRS)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.augment_mid_sent_punct(sentence))
        assert len(results) == 0, "should not augment a comma inside a number"

    def test_no_mid_sent_augmentations_returns_none(self):
        """If mid_sent_augmentations is empty, the method always returns None."""
        loader = write_and_load(HELLO_TEXT, HELLO_LABELS)
        loader.mid_sent_augmentations = {}
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.augment_mid_sent_punct(sentence))
        assert len(results) == 0

    def test_all_spacing_styles_reachable(self):
        """All four spacing styles (spaced both, left, right, neither) must be reachable."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        seen_styles = set()
        for _ in range(200):
            result = loader.augment_mid_sent_punct(sentence)
            if result is None:
                continue
            chars = result[0][3]
            try:
                dash_idx = next(i for i, c in enumerate(chars) if c in ('\u2013', '\u2014'))
            except StopIteration:
                continue
            space_before = dash_idx > 0 and chars[dash_idx - 1] == ' '
            space_after  = dash_idx < len(chars) - 1 and chars[dash_idx + 1] == ' '
            seen_styles.add((space_before, space_after))
        assert len(seen_styles) == 4, f"Only saw spacing styles: {seen_styles}"


# ---------------------------------------------------------------------------
# move_punct_back
# ---------------------------------------------------------------------------

class TestMovePunctBack:

    def test_moves_space_separated_comma(self):
        """A space-separated comma should always be moved to attach to the preceding word."""
        loader = write_and_load(SPACED_COMMA_TEXT, SPACED_COMMA_LABELS,
                                extra_args={'punct_move_back_prob': 1.0})
        loader.move_punct = build_move_punct_set(loader.data, move_back_prob=0.02)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.move_punct_back(sentence), n=1)
        assert len(results) > 0
        for result in results:
            chars = result[0][3]
            comma_idx = chars.index(',')
            assert chars[comma_idx - 1] != ' ', "comma should be attached to preceding word"

    def test_does_not_move_attached_comma(self):
        """A comma already attached to its word should never trigger move_punct_back."""
        loader = write_and_load(HELLO_TEXT, HELLO_LABELS,
                                extra_args={'punct_move_back_prob': 1.0})
        loader.move_punct = build_move_punct_set(loader.data, move_back_prob=0.02)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.move_punct_back(sentence))
        assert len(results) == 0

    def test_does_not_move_comma_after_digit(self):
        """
        '1 ,' should not be moved — the digit guard in move_punct_back prevents it.

        "1 , 000."  — comma is space-separated so build_move_punct_set includes it,
        but move_punct_back should skip it because idx-2 is a digit
        """
        text   = "1 , 000."
        labels = "01010012"
        loader = write_and_load(text, labels, extra_args={'punct_move_back_prob': 1.0})
        loader.move_punct = build_move_punct_set(loader.data, move_back_prob=0.02)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.move_punct_back(sentence))
        assert len(results) == 0

