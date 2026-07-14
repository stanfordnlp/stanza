"""
Tests for the structural/tabular text features added to tokenization/data.py
for GitHub #1640 (English tokenizer over-splitting at [digit][capitalized
word], eg "Pope Leo X used" -> "Pope Leo | X used").

These features (labeled_field, phone_id, date_pattern, currency) exist to let
the tokenizer distinguish document-structural text (news datelines, email
signature blocks, tabular listings -- where a sentence boundary right after
a bare digit really is correct) from narrative prose (where it usually isn't).

Two layers of coverage here:
  - regex-only tests, isolated from DataLoader/vocab machinery, mirroring the
    style of test_numeric_re in test_tokenize_data.py
  - DataLoader integration tests, confirming the features actually reach the
    per-character feature vectors with the right values at the right
    positions, and don't fire on ordinary prose (including the exact
    sentences from the #1640 bug report)

No pipeline download is required: like test_has_mwt in test_tokenize_data.py,
DataLoader builds its own vocab from the input data when none is passed.
"""

import pytest
import tempfile
import numpy as np

from stanza.tests import *
from stanza.models.tokenization.data import DataLoader, STRUCTURAL_FEATURES, KNOWN_FEAT_FUNCS

pytestmark = [pytest.mark.travis]

# Column order matches insertion order of STRUCTURAL_FEATURES in data.py.
# If that dict's definition order ever changes, these indices need updating.
STRUCTURAL_FEAT_PROPERTIES = {
    "lang": "en",
    "feat_funcs": ("space_before", "capitalized", "numeric",
                   "labeled_field", "phone_id", "date_pattern", "currency"),
    "max_seqlen": 1000,
    "use_dictionary": False,
}
LABELED_FIELD_COL = 3
PHONE_ID_COL = 4
DATE_PATTERN_COL = 5
CURRENCY_COL = 6
STRUCTURAL_COLS = [LABELED_FIELD_COL, PHONE_ID_COL, DATE_PATTERN_COL, CURRENCY_COL]


def write_tokenizer_input(test_dir, raw_text, labels):
    """
    Same helper as in test_tokenize_data.py -- writes raw_text and labels to
    randomly named files in test_dir. Tempfiles are not auto-cleaned; put
    them in a tempdir.
    """
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=test_dir, delete=False) as fout:
        txt_file = fout.name
        fout.write(raw_text)

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=test_dir, delete=False) as fout:
        label_file = fout.name
        fout.write(labels)

    return txt_file, label_file


def get_feats(test_dir, raw_text):
    """
    Builds a DataLoader in evaluation mode (so the whole input is treated as
    a single "sentence", per TokenizationDataset's docstring) using
    STRUCTURAL_FEAT_PROPERTIES, and returns the (seq_len, num_feats) feature
    array along with the raw character list, for inspecting specific
    positions in a test.

    Labels are irrelevant in evaluation mode (para_to_sentences doesn't cut
    on label==2/4 when self.eval is True), so we just supply all zeros.
    """
    labels = "0" * len(raw_text)
    txt_file, label_file = write_tokenizer_input(test_dir, raw_text, labels)
    data = DataLoader(args=STRUCTURAL_FEAT_PROPERTIES,
                       input_files={'txt': txt_file, 'label': label_file},
                       evaluation=True)
    assert len(data.sentences) == 1
    assert len(data.sentences[0]) == 1
    _, _, feats, raw_units = data.sentences[0][0]
    return np.array(feats), raw_units


# ---------------------------------------------------------------------------
# Regex-only tests (no DataLoader/vocab needed)
# ---------------------------------------------------------------------------

LABELED_FIELD_MATCHES = [
    "Fax: 713-654-1281",
    "Cell: 281-435-0295",
    "Job Group: Specialist",
    "Notice Regarding Entry of Orders:",
]
LABELED_FIELD_NON_MATCHES = [
    "Pope Leo X used his influence to reshape the papal court.",
    "The 5th Amendment protects against compelled self-incrimination.",
]

def test_labeled_field_regex():
    for text in LABELED_FIELD_MATCHES:
        assert STRUCTURAL_FEATURES['labeled_field'].search(text) is not None, text
    for text in LABELED_FIELD_NON_MATCHES:
        assert STRUCTURAL_FEATURES['labeled_field'].search(text) is None, text


PHONE_ID_MATCHES = [
    "(713) 571-9571",
    "713-654-0365",
    "x365",
]
PHONE_ID_NON_MATCHES = [
    "Chapter 7 for details on installation requirements.",
    "Apollo 11 landed on the moon in July of 1969.",
]

def test_phone_id_regex():
    for text in PHONE_ID_MATCHES:
        assert STRUCTURAL_FEATURES['phone_id'].search(text) is not None, text
    for text in PHONE_ID_NON_MATCHES:
        assert STRUCTURAL_FEATURES['phone_id'].search(text) is None, text


DATE_PATTERN_MATCHES = [
    "Washington Times 28/10/2004",
    "Asia Times Online 16/11/2004",
    "November 5, 1999",
]
DATE_PATTERN_NON_MATCHES = [
    "Luther's 95 Theses remain a foundational document of the Reformation.",
    "Team 7 Alpha advanced to the semifinal round.",
]

def test_date_pattern_regex():
    for text in DATE_PATTERN_MATCHES:
        assert STRUCTURAL_FEATURES['date_pattern'].search(text) is not None, text
    for text in DATE_PATTERN_NON_MATCHES:
        assert STRUCTURAL_FEATURES['date_pattern'].search(text) is None, text


CURRENCY_MATCHES = [
    "Maria Valdes superior $62,500",
    "Current Salary: $47,500",
    "$47,500.00",
]
CURRENCY_NON_MATCHES = [
    "Emperor Franz Joseph I ruled for nearly seven decades.",
    "Case 12 Doe v. Smith was cited repeatedly in the appeal.",
]

def test_currency_regex():
    for text in CURRENCY_MATCHES:
        assert STRUCTURAL_FEATURES['currency'].search(text) is not None, text
    for text in CURRENCY_NON_MATCHES:
        assert STRUCTURAL_FEATURES['currency'].search(text) is None, text


# ---------------------------------------------------------------------------
# DataLoader integration tests: confirm the regexes actually reach the
# per-character feature vectors at the right columns and positions
# ---------------------------------------------------------------------------

def test_feat_dim_matches_feat_funcs():
    """
    Sanity check on the plumbing between args['feat_funcs'] and the actual
    per-character feature vector width -- this is the invariant tokenizer.py
    relies on when it sets args['feat_dim'] = len(args['feat_funcs']).
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        feats, _ = get_feats(test_dir, "Pope Leo X used his influence.")
        assert feats.shape[1] == len(STRUCTURAL_FEAT_PROPERTIES['feat_funcs'])


def test_labeled_field_feature_positions():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        text = "Fax: 7136541281"
        feats, raw_units = get_feats(test_dir, text)
        # "Fax:" (indices 0-3, inclusive of the colon) should be flagged
        assert all(feats[i, LABELED_FIELD_COL] == 1 for i in range(0, 4))
        # a plain digit run with no label shouldn't be
        assert all(feats[i, LABELED_FIELD_COL] == 0 for i in range(5, len(text)))


def test_phone_id_feature_positions():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        text = "Call (713) 571-9571 today"
        feats, raw_units = get_feats(test_dir, text)
        phone_start = text.index("(713)")
        phone_end = text.index("9571") + len("9571")
        assert all(feats[i, PHONE_ID_COL] == 1 for i in range(phone_start, phone_end))
        # "Call" and "today" shouldn't be flagged
        assert all(feats[i, PHONE_ID_COL] == 0 for i in range(0, phone_start))
        assert all(feats[i, PHONE_ID_COL] == 0 for i in range(phone_end + 1, len(text)))


def test_date_pattern_feature_positions():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        text = "Filed on 28/10/2004 today"
        feats, raw_units = get_feats(test_dir, text)
        date_start = text.index("28/10/2004")
        date_end = date_start + len("28/10/2004")
        assert all(feats[i, DATE_PATTERN_COL] == 1 for i in range(date_start, date_end))
        assert all(feats[i, DATE_PATTERN_COL] == 0 for i in range(0, date_start))
        assert all(feats[i, DATE_PATTERN_COL] == 0 for i in range(date_end, len(text)))


def test_currency_feature_positions():
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        text = "Salary was $62,500 last year"
        feats, raw_units = get_feats(test_dir, text)
        cur_start = text.index("$62,500")
        cur_end = cur_start + len("$62,500")
        assert all(feats[i, CURRENCY_COL] == 1 for i in range(cur_start, cur_end))
        assert all(feats[i, CURRENCY_COL] == 0 for i in range(0, cur_start))
        assert all(feats[i, CURRENCY_COL] == 0 for i in range(cur_end, len(text)))


def test_all_four_features_combined():
    """
    All four patterns in one paragraph, checking each fires independently
    without interfering with the others.
    """
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        text = "Fax: 713-654-1281 filed 28/10/2004 for $62,500 total"
        feats, raw_units = get_feats(test_dir, text)
        assert feats[:, LABELED_FIELD_COL].sum() > 0
        assert feats[:, PHONE_ID_COL].sum() > 0
        assert feats[:, DATE_PATTERN_COL].sum() > 0
        assert feats[:, CURRENCY_COL].sum() > 0


# ---------------------------------------------------------------------------
# No false positives on the #1640 bug report sentences, or on ordinary prose
# from the corrective training examples
# ---------------------------------------------------------------------------

BUG_REPORT_SENTENCES = [
    "Pope Leo X used his position to influence the papal court.",
    "Luther's 95 Theses remain a foundational document of the Reformation.",
    "Henry VIII had six wives over the course of his reign.",
    "Chapter 7 for details on installation requirements.",
    "Apollo 11 landed on the moon in July of 1969.",
    "Emperor Franz Joseph I ruled for nearly seven decades.",
]

PROSE_SENTENCES = [
    "The 5th Amendment protects against compelled self-incrimination.",
    "Team 7 Alpha advanced to the semifinal round.",
    "Case 12 Doe v. Smith was cited repeatedly in the appeal.",
    "Chapter 9 Methodology explains how the samples were gathered.",
]

@pytest.mark.parametrize("text", BUG_REPORT_SENTENCES + PROSE_SENTENCES)
def test_no_false_positives_on_prose(text):
    with tempfile.TemporaryDirectory(dir=TEST_WORKING_DIR) as test_dir:
        feats, _ = get_feats(test_dir, text)
        for col in STRUCTURAL_COLS:
            assert feats[:, col].sum() == 0, \
                "structural column {} unexpectedly fired on {!r}".format(col, text)


# ---------------------------------------------------------------------------
# Guard tests for run_tokenizer.py's EXTRA_FEAT_FUNCS staying in sync with
# what data.py actually recognizes and with tokenizer.py's base defaults.
# Without these, a rename/removal in data.py or a change to
# tokenizer.DEFAULT_FEAT_FUNCS could go stale in run_tokenizer.py silently
# (or only surface as a ValueError deep into an actual training run).
# ---------------------------------------------------------------------------

from stanza.models import tokenizer as tokenizer_module
from stanza.utils.training import run_tokenizer

def test_extra_feat_funcs_are_recognized():
    """
    Every name in run_tokenizer.EXTRA_FEAT_FUNCS must be something data.py's
    para_to_sentences dispatch actually recognizes (else it raises
    ValueError at training time, not at test time).
    """
    for lang, extras in run_tokenizer.EXTRA_FEAT_FUNCS.items():
        for name in extras:
            assert name in KNOWN_FEAT_FUNCS, \
                "{!r} (added for {!r}) is not a recognized feat_func name".format(name, lang)


def test_extra_feat_funcs_do_not_duplicate_base_defaults():
    """
    A name listed in both tokenizer.DEFAULT_FEAT_FUNCS and
    run_tokenizer.EXTRA_FEAT_FUNCS would silently duplicate that feature
    column in the model's input vector.
    """
    base = set(tokenizer_module.DEFAULT_FEAT_FUNCS)
    for lang, extras in run_tokenizer.EXTRA_FEAT_FUNCS.items():
        overlap = base & set(extras)
        assert not overlap, \
            "{} in EXTRA_FEAT_FUNCS[{!r}] duplicates a base default".format(overlap, lang)


def test_default_feat_funcs_composition():
    """
    default_feat_funcs(lang) should compose tokenizer.py's base defaults
    with the language-specific additions -- this is the actual mechanism
    that replaces re-typing the full list in run_tokenizer.py, so it's
    worth pinning down directly rather than only via the two tests above.
    """
    # a language with no additions falls back to None, letting
    # tokenizer.py's own default apply unchanged
    assert run_tokenizer.default_feat_funcs("de") is None

    en_funcs = run_tokenizer.default_feat_funcs("en")
    assert en_funcs is not None
    assert en_funcs[:len(tokenizer_module.DEFAULT_FEAT_FUNCS)] == list(tokenizer_module.DEFAULT_FEAT_FUNCS)
    assert set(en_funcs) == set(tokenizer_module.DEFAULT_FEAT_FUNCS) | set(run_tokenizer.EXTRA_FEAT_FUNCS["en"])
