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
import re
import os
import tempfile
import inspect

from stanza.tests import TEST_WORKING_DIR
from stanza.models.tokenization import data as data_module
from stanza.models.tokenization.data import (
    DataLoader,
    MID_SENT_AUGMENT_PAIRS,
    build_move_punct_set,
    build_known_mwt,
)
from stanza.models.tokenization.vocab import Vocab

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]


def discover_augmentation_probs():
    """
    Find every '*_prob' DataLoader argument referenced in data.py, by
    scanning its source for args.get('xxx_prob', ...) / args['xxx_prob'].

    This drives the next()-integration smoke test below. Deriving the list
    from the source (instead of hand-maintaining it here) means a newly
    added augmentation is automatically picked up the next time the tests
    run -- no separate edit to this test file is required when someone adds
    a new `whatever_prob` argument to DataLoader.
    """
    source = inspect.getsource(data_module)
    return sorted(set(re.findall(r"args(?:\.get)?\(?\[?'(\w+_prob)'", source)))


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

# "Hello,world."  (comma already attached, no space anywhere near it —
# used to confirm comma_typo leaves already-glued commas alone, since
# there is no trailing space for it to relocate)
#  H e l l o ,  w o r l d  .
#  0 0 0 0 0 1  0 0 0 0 1  2
ATTACHED_COMMA_TEXT   = "Hello,world."
ATTACHED_COMMA_LABELS = "000001000012"


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


# ---------------------------------------------------------------------------
# comma_typo
# ---------------------------------------------------------------------------

class TestCommaTypo:

    def _loader(self, text=HELLO_TEXT, labels=HELLO_LABELS):
        # comma_typo_prob=1.0 activates the vocab check in __init__;
        # all other augmentation probs remain at 0.0
        return write_and_load(text, labels, extra_args={'comma_typo_prob': 1.0})

    def test_eligible_when_comma_present(self):
        """A comma anywhere in the training data marks comma_typo as eligible."""
        loader = self._loader()
        assert loader.comma_typo_eligible is True

    def test_ineligible_when_no_comma(self):
        """No comma anywhere in the training data -> comma_typo never activates."""
        # "Hi there."  H i   t h e r e  .   labels 0 1 0 0 0 0 0 1 2
        loader = self._loader(text="Hi there.", labels="010000012")
        assert loader.comma_typo_eligible is False
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) == 0

    def test_moves_space_to_before_comma(self):
        """'Hello, world.' should always become 'Hello ,world.'."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) > 0, "comma_typo never returned a result"
        for result in results:
            chars = result[0][3]
            assert ''.join(chars) == "Hello ,world."

    def test_comma_remains_own_token(self):
        """After the typo, the comma must still carry a non-zero (own-token) label."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) > 0
        for result in results:
            new_sentence = result[0]
            comma_idx = new_sentence[3].index(',')
            assert new_sentence[1][comma_idx] != 0, "comma should not be a continuation"

    def test_preceding_word_end_label_unchanged(self):
        """
        w1's final character must KEEP its word-end label (1) after the typo.

        This mirrors the natural "w1 , w2" pattern (space before comma)
        already in the corpus, where the space is a continuation (0) and
        the letter before it carries the word-end label. Demoting w1's
        final character to a continuation would incorrectly teach the
        tokenizer that "w1 ," is a single token spanning the space.
        """
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) > 0
        for result in results:
            new_sentence = result[0]
            chars = new_sentence[3]
            o_idx = chars.index('o')  # last letter of "Hello"
            assert new_sentence[1][o_idx] == 1, "'o' should remain a word end, not become a continuation"

    def test_inserted_space_is_continuation(self):
        """The newly inserted space (before the comma) must carry label 0."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) > 0
        for result in results:
            new_sentence = result[0]
            chars = new_sentence[3]
            comma_idx = chars.index(',')
            assert chars[comma_idx - 1] == ' ', "expected a space immediately before the comma"
            assert new_sentence[1][comma_idx - 1] == 0, "inserted space should be a continuation"

    def test_matches_natural_spaced_comma_labels(self):
        """
        The augmented "Hello ,world." should carry the same labels, at every
        shared position, as the naturally occurring "Hello , world." -- i.e.
        comma_typo should produce a label sequence consistent with how the
        corpus already encodes a word followed by a space-separated comma.
        """
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) > 0

        gold_loader = self._loader(text="Hello , world.", labels="00001010000012")
        gold_sentence = gold_loader.sentences[0][0]
        gold_chars = gold_sentence[3]
        gold_labels = [int(l) for l in gold_sentence[1]]

        for result in results:
            new_sentence = result[0]
            chars = new_sentence[3]
            labels = [int(l) for l in new_sentence[1]]
            # "Hello ,world." == "Hello , world." with the space after the
            # comma removed -- so every position up to and including the
            # comma should match gold exactly, label for label.
            comma_idx = chars.index(',')
            assert chars[:comma_idx + 1] == gold_chars[:comma_idx + 1]
            assert labels[:comma_idx + 1] == gold_labels[:comma_idx + 1]

    def test_does_not_move_already_attached_comma(self):
        """A comma with no following space (already glued to w2) should never trigger."""
        loader = self._loader(text=ATTACHED_COMMA_TEXT, labels=ATTACHED_COMMA_LABELS)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) == 0

    def test_comma_in_number_not_moved(self):
        """A comma with label 0 (inside a number token, e.g. '1,000') is never moved."""
        # "1,000 here."  1(0) ,(0) 0(0)0(0)0(1) space(0) h(0)e(0)r(0)e(1) .(2)
        text   = "1,000 here."
        labels = "00001" "0" "0001" "2"
        loader = self._loader(text=text, labels=labels)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) == 0, "should not augment a comma inside a number"

    def test_no_op_when_not_eligible(self):
        """Even with the per-call gate active, an explicitly disabled loader stays inert."""
        loader = self._loader()
        loader.comma_typo_eligible = False
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_typo(sentence))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# comma_glue
# ---------------------------------------------------------------------------

class TestCommaGlue:

    def _loader(self, text=HELLO_TEXT, labels=HELLO_LABELS):
        # comma_glue_prob=1.0 activates the vocab check in __init__;
        # all other augmentation probs remain at 0.0
        return write_and_load(text, labels, extra_args={'comma_glue_prob': 1.0})

    def test_eligible_when_comma_present(self):
        """A comma anywhere in the training data marks comma_glue as eligible."""
        loader = self._loader()
        assert loader.comma_glue_eligible is True

    def test_ineligible_when_no_comma(self):
        """No comma anywhere in the training data -> comma_glue never activates."""
        # "Hi there."  H i   t h e r e  .   labels 0 1 0 0 0 0 0 1 2
        loader = self._loader(text="Hi there.", labels="010000012")
        assert loader.comma_glue_eligible is False
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) == 0

    def test_removes_space_after_comma(self):
        """'Hello, world.' should always become 'Hello,world.'."""
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) > 0, "comma_glue never returned a result"
        for result in results:
            chars = result[0][3]
            assert ''.join(chars) == "Hello,world."

    def test_matches_naturally_glued_comma_labels(self):
        """
        The augmented "Hello,world." should be character-for-character and
        label-for-label identical to the naturally occurring "Hello,world."
        -- i.e. comma_glue's pure-deletion transform should produce a label
        sequence consistent with how the corpus already encodes a comma
        attached on both sides.
        """
        loader = self._loader()
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) > 0

        # "Hello,world."  H e l l o , w o r l d .
        #                 0 0 0 0 1 1 0 0 0 0 1 2
        gold_loader = self._loader(text="Hello,world.", labels="000011000012")
        gold_sentence = gold_loader.sentences[0][0]
        gold_chars = list(gold_sentence[3])
        gold_labels = [int(l) for l in gold_sentence[1]]

        for result in results:
            new_sentence = result[0]
            chars = list(new_sentence[3])
            labels = [int(l) for l in new_sentence[1]]
            assert chars == gold_chars
            assert labels == gold_labels

    def test_does_not_glue_already_attached_comma(self):
        """A comma with no following space (already glued to w2) should never trigger."""
        loader = self._loader(text=ATTACHED_COMMA_TEXT, labels=ATTACHED_COMMA_LABELS)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) == 0

    def test_comma_in_number_not_glued(self):
        """A comma with label 0 (inside a number token, e.g. '1,000') is never glued."""
        # "1,000 here."  1(0) ,(0) 0(0)0(0)0(1) space(0) h(0)e(0)r(0)e(1) .(2)
        text   = "1,000 here."
        labels = "00001" "0" "0001" "2"
        loader = self._loader(text=text, labels=labels)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) == 0, "should not glue a comma already inside a number"

    def test_blocked_when_digit_adjacent_on_both_sides(self):
        """
        '3, 000 things.' must never be glued to '3,000 things.', since that
        would be indistinguishable from a genuine European-style thousands
        separator and must not be taught as a place to split.
        """
        # "3, 000 things."  3(1) ,(1) space(0) 0(0)0(0)0(1) space(0) t..s(0..1) .(2)
        text   = "3, 000 things."
        labels = "11" "0" "001" "0" "000001" "2"
        loader = self._loader(text=text, labels=labels)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) == 0, "should not glue a comma between two digit runs"

    def test_allowed_when_digit_only_before_comma(self):
        """'3, bar.' (digit before, non-digit after) is safe to glue -> '3,bar.'."""
        # "3, bar."  3(1) ,(1) space(0) b(0)a(0)r(1) .(2)
        text   = "3, bar."
        labels = "11" "0" "001" "2"
        loader = self._loader(text=text, labels=labels)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) > 0, "should be allowed when only the left side is a digit"
        for result in results:
            assert ''.join(result[0][3]) == "3,bar."

    def test_allowed_when_digit_only_after_comma(self):
        """'foo, 5 bar.' (non-digit before, digit after) is safe to glue -> 'foo,5 bar.'."""
        # "foo, 5 bar."  f(0)o(0)o(1) ,(1) space(0) 5(1) space(0) b(0)a(0)r(1) .(2)
        text   = "foo, 5 bar."
        labels = "001" "1" "0" "1" "0" "001" "2"
        loader = self._loader(text=text, labels=labels)
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) > 0, "should be allowed when only the right side is a digit"
        for result in results:
            assert ''.join(result[0][3]) == "foo,5 bar."

    def test_no_op_when_not_eligible(self):
        """Even with the per-call gate active, an explicitly disabled loader stays inert."""
        loader = self._loader()
        loader.comma_glue_eligible = False
        sentence = loader.sentences[0][0]
        results = run_trials(lambda: loader.comma_glue(sentence))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Integration: DataLoader.next()
# ---------------------------------------------------------------------------
#
# Every test above calls an augmentation method directly on a `sentence`
# tuple. That's the right level for testing the augmentation logic itself,
# but it never exercises next() / strings_starting(), which is the code
# path actually used during training. A bug that only manifests there
# (e.g. a method definition accidentally lost or misplaced during an edit,
# or a missing total_len adjustment for a length-changing augmentation)
# would not be caught by any of the method-level tests above. These tests
# close that gap by running the real batch-fetching path end to end.
#
# The set of augmentation probs exercised here is discovered automatically
# from data.py's source (see discover_augmentation_probs() above), rather
# than hand-listed -- so adding a new `whatever_prob` augmentation to
# DataLoader does NOT require touching this test file; the next pytest run
# will pick it up and smoke-test it through next() automatically.

# A small multi-sentence/multi-paragraph corpus with several commas, long
# enough that batch_size=2 has more than one paragraph to sample from.
# Paragraphs are separated by a blank line, matching the corpus format
# used elsewhere in this file.
#   "Hello, world."   0000110000012
#   "foo, bar."       001100012
#   "This, that."     00011000012
#   "A, b."           11012
INTEGRATION_TEXT = "Hello, world.\n\nfoo, bar.\n\nThis, that.\n\nA, b.\n"
INTEGRATION_LABELS = "0000110000012" + "\n\n" + "001100012" + "\n\n" + "00011000012" + "\n\n" + "11012" + "\n"


class TestDataLoaderNextIntegration:

    def test_discovery_finds_known_probs(self):
        """
        Guard against the discovery regex silently breaking (e.g. if the
        '*_prob' naming convention or the args.get(...) call style in
        data.py ever changes): if discover_augmentation_probs() starts
        returning an empty or suspiciously short list, the parametrized
        test below would silently run zero/fewer cases instead of failing.
        This pins a minimum expected set so that kind of breakage is loud.
        """
        found = discover_augmentation_probs()
        expected_minimum = {
            'comma_typo_prob', 'comma_glue_prob', 'punct_move_back_prob',
            'last_char_move_prob', 'last_char_drop_prob', 'split_mwt_prob',
            'augment_mid_punct_prob', 'augment_final_punct_prob',
        }
        missing = expected_minimum - set(found)
        assert not missing, f"discover_augmentation_probs() stopped finding: {missing}"

    def _loader(self, extra_args=None):
        args = {'batch_size': 2}
        if extra_args:
            args.update(extra_args)
        return write_and_load(INTEGRATION_TEXT, INTEGRATION_LABELS, extra_args=args)

    @pytest.mark.parametrize("prob_arg", discover_augmentation_probs())
    def test_next_runs_with_augmentation_active(self, prob_arg):
        """
        DataLoader.next() must run without error, repeatedly, with each
        augmentation probability set to 1.0 individually. This is a smoke
        test: it does not check label correctness (that's covered by the
        method-level tests above), only that the full batch-fetching path
        -- including whichever augmentation is active -- does not crash
        and returns correctly-shaped output.
        """
        loader = self._loader(extra_args={prob_arg: 1.0})
        for _ in range(10):
            units, labels, feats, raw_units = loader.next()
            assert units.shape == labels.shape
            assert units.shape[0] == feats.shape[0] == len(raw_units)

    def test_next_runs_with_all_augmentations_active(self):
        """Sanity check that the augmentations don't crash when combined."""
        loader = self._loader(extra_args={
            prob_arg: 0.5 for prob_arg in discover_augmentation_probs()
        })
        for _ in range(20):
            units, labels, feats, raw_units = loader.next()
            assert units.shape == labels.shape

    def test_comma_glue_observable_in_real_batches(self):
        """
        With comma_glue_prob=1.0, repeatedly drawn batches should eventually
        contain a comma with no space immediately after it -- i.e. confirm
        the augmentation actually reaches the data returned by next(), not
        just that next() avoids crashing.
        """
        loader = self._loader(extra_args={'comma_glue_prob': 1.0})
        assert loader.comma_glue_eligible is True

        saw_glued_comma = False
        for _ in range(50):
            _, _, _, raw_units = loader.next()
            for sent_raw in raw_units:
                chars = [c for c in sent_raw if c != '<PAD>']
                for j, c in enumerate(chars[:-1]):
                    if c == ',' and chars[j + 1] != ' ':
                        saw_glued_comma = True
        assert saw_glued_comma, "comma_glue never appeared to fire across 50 batches"
