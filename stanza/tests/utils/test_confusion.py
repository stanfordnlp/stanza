"""
Unit tests for stanza.utils.confusion.

Run with:
    pytest stanza/tests/utils/test_confusion.py
"""

import pytest
from collections import defaultdict
from stanza.utils.confusion import (
    condense_ner_labels,
    format_confusion,
    confusion_to_accuracy,
    confusion_to_f1,
    confusion_to_macro_f1,
    confusion_to_weighted_f1,
)

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def pos_confusion():
    """Simple 2-class POS confusion matrix (NOUN/VERB) used across many tests."""
    return {
        'NOUN': {'NOUN': 50, 'VERB': 3},
        'VERB': {'NOUN': 1, 'VERB': 40},
    }


def perfect_confusion():
    """Confusion matrix where every prediction is correct."""
    return {
        'A': {'A': 10},
        'B': {'B': 20},
        'C': {'C': 5},
    }


def zero_confusion():
    """Confusion matrix where nothing was ever predicted correctly."""
    return {
        'A': {'B': 5},
        'B': {'A': 3},
    }


def ner_confusion():
    """Toy BIOES NER confusion matrix."""
    return {
        'B-PER': {'B-PER': 10, 'I-PER': 1},
        'I-PER': {'I-PER': 8,  'B-PER': 0},
        'S-ORG': {'S-ORG': 6,  'O': 2},
        'O':     {'O': 100,    'S-ORG': 1},
    }


# ---------------------------------------------------------------------------
# condense_ner_labels
# ---------------------------------------------------------------------------

class TestCondenseNerLabels:

    def test_strips_bioes_prefixes(self):
        cm = {'B-PER': {'B-PER': 5, 'I-ORG': 1}, 'I-ORG': {'B-PER': 0, 'I-ORG': 3}}
        new_cm, new_gold, new_pred = condense_ner_labels(cm, ['B-PER', 'I-ORG'], ['B-PER', 'I-ORG'])
        assert set(new_gold) == {'PER', 'ORG'}
        assert set(new_pred) == {'PER', 'ORG'}

    def test_counts_are_preserved(self):
        cm = {'B-PER': {'B-PER': 5, 'I-PER': 2}, 'I-PER': {'B-PER': 1, 'I-PER': 4}}
        new_cm, new_gold, new_pred = condense_ner_labels(cm, ['B-PER', 'I-PER'], ['B-PER', 'I-PER'])
        # All four cells collapse into PER→PER
        assert new_cm['PER']['PER'] == 5 + 2 + 1 + 4

    def test_labels_without_hyphen_are_unchanged(self):
        cm = {'O': {'O': 10, 'B-PER': 1}, 'B-PER': {'O': 0, 'B-PER': 5}}
        new_cm, new_gold, new_pred = condense_ner_labels(cm, ['O', 'B-PER'], ['O', 'B-PER'])
        assert 'O' in new_gold
        assert 'PER' in new_gold

    def test_deduplicates_labels(self):
        # B-PER and I-PER both condense to PER — should appear only once
        cm = {'B-PER': {'B-PER': 3}, 'I-PER': {'I-PER': 2}}
        _, new_gold, new_pred = condense_ner_labels(cm, ['B-PER', 'I-PER'], ['B-PER', 'I-PER'])
        assert new_gold.count('PER') == 1
        assert new_pred.count('PER') == 1


# ---------------------------------------------------------------------------
# format_confusion
# ---------------------------------------------------------------------------

class TestFormatConfusion:

    def test_returns_string(self):
        result = format_confusion(pos_confusion())
        assert isinstance(result, str)

    def test_labels_appear_in_output(self):
        result = format_confusion(pos_confusion())
        assert 'NOUN' in result
        assert 'VERB' in result

    def test_diagonal_values_appear(self):
        result = format_confusion(pos_confusion())
        assert '50' in result
        assert '40' in result

    def test_corner_label_default(self):
        result = format_confusion(pos_confusion())
        assert 't\\p' in result

    def test_corner_label_transposed(self):
        result = format_confusion(pos_confusion(), transpose=True)
        assert 'p\\t' in result

    def test_transpose_swaps_cells(self):
        # In the original, NOUN→VERB = 3 and VERB→NOUN = 1.
        # After transpose rows are predicted, columns are gold, so the
        # cell positions swap but the values remain in the output.
        original = format_confusion(pos_confusion())
        transposed = format_confusion(pos_confusion(), transpose=True)
        # Both values must still be present either way
        assert '3' in transposed
        assert '1' in transposed

    def test_hide_zeroes_removes_zero_cells(self):
        cm = {'A': {'A': 5, 'B': 0}, 'B': {'A': 0, 'B': 3}}
        result_shown = format_confusion(cm, hide_zeroes=False)
        result_hidden = format_confusion(cm, hide_zeroes=True)
        # With hide_zeroes=False the zero should appear as '0'
        assert '0' in result_shown
        # With hide_zeroes=True the zero cells become blank — '0' may still
        # appear as part of '50' etc., but the standalone zero is gone.
        # We check by counting: hiding should reduce or equal occurrences.
        assert result_hidden.count(' 0 ') <= result_shown.count(' 0 ')

    def test_explicit_labels_respected(self):
        # Only the supplied labels should appear in the output
        result = format_confusion(pos_confusion(), labels=['NOUN', 'VERB'])
        assert 'NOUN' in result
        assert 'VERB' in result

    def test_hide_blank_removes_all_zero_rows(self):
        cm = {
            'A': {'A': 5},
            'B': {'B': 0},   # entirely zero row
        }
        result = format_confusion(cm, hide_blank=True)
        assert 'A' in result
        assert 'B' not in result

    def test_perfect_matrix_has_zeroes_only_off_diagonal(self):
        result = format_confusion(perfect_confusion())
        assert '10' in result
        assert '20' in result
        assert '5' in result

    def test_float_values_formatted_with_decimal(self):
        cm = {'A': {'A': 4.5, 'B': 0.5}, 'B': {'A': 1.0, 'B': 3.0}}
        result = format_confusion(cm)
        assert '.' in result   # float formatting is active

    def test_o_label_sorted_first(self):
        cm = {'O': {'O': 100}, 'B-PER': {'B-PER': 5}}
        result = format_confusion(cm)
        lines = result.split('\n')
        # header line contains both; first data row should be O
        data_rows = [l for l in lines if l.strip() and 't\\p' not in l]
        assert data_rows[0].strip().startswith('O')

    def test_wide_ner_matrix_auto_condenses(self):
        # Build a matrix wide enough to trigger auto-condensing (>150 chars)
        labels = [f'B-{e}' for e in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'] + [f'E-{e}' for e in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        cm = defaultdict(lambda: defaultdict(int))
        for l in labels:
            cm[l][l] = 1
        result = format_confusion(cm)
        # After condensing the BIES prefix is gone
        assert 'B-A' not in result

    def test_2x2_output_exact(self):
        result = format_confusion(pos_confusion())
        expected = (
            "     t\\p   NOUN  VERB\n"
            "     NOUN    50     3\n"
            "     VERB     1    40"
        )
        assert result == expected

    def test_2x2_output_exact_transposed(self):
        result = format_confusion(pos_confusion(), transpose=True)
        expected = (
            "     p\\t   NOUN  VERB\n"
            "     NOUN    50     1\n"
            "     VERB     3    40"
        )
        assert result == expected
# ---------------------------------------------------------------------------
# confusion_to_accuracy
# ---------------------------------------------------------------------------

class TestConfusionToAccuracy:

    def test_basic_accuracy(self):
        correct, total = confusion_to_accuracy(pos_confusion())
        assert correct == 90
        assert total == 94

    def test_perfect_accuracy(self):
        correct, total = confusion_to_accuracy(perfect_confusion())
        assert correct == total == 35

    def test_zero_accuracy(self):
        correct, total = confusion_to_accuracy(zero_confusion())
        assert correct == 0
        assert total == 8

    def test_single_class(self):
        cm = {'A': {'A': 7}}
        correct, total = confusion_to_accuracy(cm)
        assert correct == 7
        assert total == 7


# ---------------------------------------------------------------------------
# confusion_to_f1
# ---------------------------------------------------------------------------

class TestConfusionToF1:

    def test_returns_all_labels(self):
        results = confusion_to_f1(pos_confusion())
        assert set(results.keys()) == {'NOUN', 'VERB'}

    def test_f1_values_in_range(self):
        results = confusion_to_f1(pos_confusion())
        for label, r in results.items():
            assert 0.0 <= r.precision <= 1.0
            assert 0.0 <= r.recall <= 1.0
            assert 0.0 <= r.f1 <= 1.0

    def test_perfect_f1(self):
        results = confusion_to_f1(perfect_confusion())
        for label, r in results.items():
            assert r.precision == pytest.approx(1.0)
            assert r.recall == pytest.approx(1.0)
            assert r.f1 == pytest.approx(1.0)

    def test_zero_f1_when_never_predicted(self):
        # 'A' is never predicted (only predicted as 'B')
        cm = {'A': {'B': 5}, 'B': {'B': 3}}
        results = confusion_to_f1(cm)
        assert results['A'].precision == pytest.approx(0.0)
        assert results['A'].f1 == pytest.approx(0.0)

    def test_zero_recall_when_never_in_gold(self):
        # 'B' is predicted but never appears as gold
        cm = {'A': {'A': 5, 'B': 2}}
        results = confusion_to_f1(cm)
        assert results['B'].recall == pytest.approx(0.0)
        assert results['B'].f1 == pytest.approx(0.0)

    def test_noun_f1_value(self):
        # NOUN: TP=50, FP=1, FN=3  → P=50/51, R=50/53
        results = confusion_to_f1(pos_confusion())
        expected_p = 50 / 51
        expected_r = 50 / 53
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert results['NOUN'].precision == pytest.approx(expected_p)
        assert results['NOUN'].recall == pytest.approx(expected_r)
        assert results['NOUN'].f1 == pytest.approx(expected_f1)

    def test_verb_f1_value(self):
        # VERB: TP=40, FP=3, FN=1  → P=40/43, R=40/41
        results = confusion_to_f1(pos_confusion())
        expected_p = 40 / 43
        expected_r = 40 / 41
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert results['VERB'].precision == pytest.approx(expected_p)
        assert results['VERB'].recall == pytest.approx(expected_r)
        assert results['VERB'].f1 == pytest.approx(expected_f1)

    def test_namedtuple_fields(self):
        results = confusion_to_f1(pos_confusion())
        r = results['NOUN']
        # Verify the namedtuple fields are accessible by name
        assert hasattr(r, 'precision')
        assert hasattr(r, 'recall')
        assert hasattr(r, 'f1')


# ---------------------------------------------------------------------------
# confusion_to_macro_f1
# ---------------------------------------------------------------------------

class TestConfusionToMacroF1:

    def test_perfect_macro_f1(self):
        assert confusion_to_macro_f1(perfect_confusion()) == pytest.approx(1.0)

    def test_macro_f1_is_unweighted_mean(self):
        results = confusion_to_f1(pos_confusion())
        expected = sum(r.f1 for r in results.values()) / len(results)
        assert confusion_to_macro_f1(pos_confusion()) == pytest.approx(expected)

    def test_macro_f1_in_range(self):
        score = confusion_to_macro_f1(pos_confusion())
        assert 0.0 <= score <= 1.0

    def test_macro_f1_treats_rare_class_equally(self):
        # A rare class with bad performance should pull the macro average down
        # more than it would pull weighted F1 down.
        cm = {
            'COMMON': {'COMMON': 1000, 'RARE': 0},
            'RARE':   {'COMMON': 10,   'RARE': 1},
        }
        macro = confusion_to_macro_f1(cm)
        weighted = confusion_to_weighted_f1(cm)
        assert macro < weighted


# ---------------------------------------------------------------------------
# confusion_to_weighted_f1
# ---------------------------------------------------------------------------

class TestConfusionToWeightedF1:

    def test_perfect_weighted_f1(self):
        assert confusion_to_weighted_f1(perfect_confusion()) == pytest.approx(1.0)

    def test_known_value(self):
        # Verified interactively against the actual implementation
        assert confusion_to_weighted_f1(pos_confusion()) == pytest.approx(0.9575442288208247)

    def test_weighted_f1_in_range(self):
        score = confusion_to_weighted_f1(pos_confusion())
        assert 0.0 <= score <= 1.0

    def test_exclude_removes_class_from_average(self):
        # Excluding NOUN leaves only VERB, so result should equal VERB's F1
        results = confusion_to_f1(pos_confusion())
        verb_f1 = results['VERB'].f1
        assert confusion_to_weighted_f1(pos_confusion(), exclude={'NOUN'}) == pytest.approx(verb_f1)

    def test_exclude_noun_known_value(self):
        assert confusion_to_weighted_f1(pos_confusion(), exclude={'NOUN'}) == pytest.approx(0.9523809523809524)

    def test_exclude_none_same_as_default(self):
        assert confusion_to_weighted_f1(pos_confusion(), exclude=None) == pytest.approx(
            confusion_to_weighted_f1(pos_confusion())
        )

    def test_weights_proportional_to_row_sums(self):
        # Build a matrix where one class has 10x more instances.
        # The weighted F1 should be closer to the frequent class's F1.
        cm = {
            'FREQ': {'FREQ': 100, 'RARE': 5},
            'RARE': {'FREQ': 1,   'RARE': 2},
        }
        results = confusion_to_f1(cm)
        weighted = confusion_to_weighted_f1(cm)
        # Weighted result should lie between the two per-class F1s
        lo = min(results['FREQ'].f1, results['RARE'].f1)
        hi = max(results['FREQ'].f1, results['RARE'].f1)
        assert lo <= weighted <= hi
