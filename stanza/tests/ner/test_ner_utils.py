import pytest

from stanza.tests import *

from stanza.models.ner import utils

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

WORDS       = [["Unban",   "Mox",  "Opal"], ["Ragavan",  "is",     "red"], ["Urza",   "Lord",  "High", "Artificer", "goes", "infinite", "with",  "Thopter",    "Sword"]]
BIO_TAGS    = [["O",     "B-ART", "I-ART"], ["B-MONKEY", "O",  "B-COLOR"], ["B-PER", "I-PER", "I-PER", "I-PER",        "O",        "O",    "O", "B-WEAPON", "B-WEAPON"]]
BIO_U_TAGS  = [["O",     "B_ART", "I_ART"], ["B_MONKEY", "O",  "B_COLOR"], ["B_PER", "I_PER", "I_PER", "I_PER",        "O",        "O",    "O", "B_WEAPON", "B_WEAPON"]]
BIOES_TAGS  = [["O",     "B-ART", "E-ART"], ["S-MONKEY", "O",  "S-COLOR"], ["B-PER", "I-PER", "I-PER", "E-PER",        "O",        "O",    "O", "S-WEAPON", "S-WEAPON"]]
# note the problem with not using BIO tags - the consecutive tags for thopter/sword get treated as one item
BASIC_TAGS  = [["O",       "ART",   "ART"], ["MONKEY",   "O",    "COLOR"], [  "PER",   "PER",   "PER",   "PER",        "O",        "O",    "O",   "WEAPON",   "WEAPON"]]
BASIC_BIOES = [["O",     "B-ART", "E-ART"], ["S-MONKEY", "O",  "S-COLOR"], ["B-PER", "I-PER", "I-PER", "E-PER",        "O",        "O",    "O", "B-WEAPON", "E-WEAPON"]]

def check_reprocessed_tags(words, input_tags, expected_tags):
    sentences = [list(zip(x, y)) for x, y in zip(words, input_tags)]
    retagged = utils.process_tags(sentences=sentences, scheme="bioes")
    expected_retagged = [list(zip(x, y)) for x, y in zip(words, expected_tags)]
    assert retagged == expected_retagged

def test_process_tags_bio():
    check_reprocessed_tags(WORDS, BIO_TAGS, BIOES_TAGS)

def test_process_tags_basic():
    check_reprocessed_tags(WORDS, BASIC_TAGS, BASIC_BIOES)

def test_process_tags_bioes():
    """
    This one should not change, naturally
    """
    check_reprocessed_tags(WORDS, BIOES_TAGS, BIOES_TAGS)
    check_reprocessed_tags(WORDS, BASIC_BIOES, BASIC_BIOES)

def run_flattened(fn, tags):
    return fn([x for x in y for y in tags])

def test_check_bio():
    assert     utils.is_bio_scheme([x for y in BIO_TAGS for x in y])
    assert not utils.is_bio_scheme([x for y in BIOES_TAGS for x in y])
    assert not utils.is_bio_scheme([x for y in BASIC_TAGS for x in y])
    assert not utils.is_bio_scheme([x for y in BASIC_BIOES for x in y])

def test_check_basic():
    assert not utils.is_basic_scheme([x for y in BIO_TAGS for x in y])
    assert not utils.is_basic_scheme([x for y in BIOES_TAGS for x in y])
    assert     utils.is_basic_scheme([x for y in BASIC_TAGS for x in y])
    assert not utils.is_basic_scheme([x for y in BASIC_BIOES for x in y])

def test_underscores():
    """
    Check that the methods work if the inputs are underscores instead of dashes
    """
    assert not utils.is_basic_scheme([x for y in BIO_U_TAGS for x in y])
    check_reprocessed_tags(WORDS, BIO_U_TAGS, BIOES_TAGS)

def test_merge_tags():
    """
    Check a few versions of the tag sequence merging
    """
    seq1     = [     "O",     "O",     "O", "B-FOO", "E-FOO",     "O"]
    seq2     = [ "S-FOO",     "O", "B-FOO", "E-FOO",     "O",     "O"]
    seq3     = [ "B-FOO", "E-FOO", "B-FOO", "E-FOO",     "O",     "O"]
    seq_err  = [     "O", "B-FOO",     "O", "B-FOO", "E-FOO",     "O"]
    seq_err2 = [     "O", "B-FOO",     "O", "B-FOO", "B-FOO",     "O"]
    seq_err3 = [     "O", "B-FOO",     "O", "B-FOO", "I-FOO",     "O"]
    seq_err4 = [     "O", "B-FOO",     "O", "B-FOO", "I-FOO", "I-FOO"]

    result = utils.merge_tags(seq1, seq2)
    expected = [ "S-FOO",     "O",     "O", "B-FOO", "E-FOO",     "O"]
    assert result == expected

    result = utils.merge_tags(seq2, seq1)
    expected = [ "S-FOO",     "O", "B-FOO", "E-FOO",     "O",     "O"]
    assert result == expected

    result = utils.merge_tags(seq1, seq3)
    expected = [ "B-FOO", "E-FOO",     "O", "B-FOO", "E-FOO",     "O"]
    assert result == expected

    with pytest.raises(ValueError):
        result = utils.merge_tags(seq1, seq_err)

    with pytest.raises(ValueError):
        result = utils.merge_tags(seq1, seq_err2)

    with pytest.raises(ValueError):
        result = utils.merge_tags(seq1, seq_err3)

    with pytest.raises(ValueError):
        result = utils.merge_tags(seq1, seq_err4)

