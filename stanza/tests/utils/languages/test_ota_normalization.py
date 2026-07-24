"""
Tests for stanza.utils.languages.ota.transliterate.ota_converter

The intent is to pin the behavior of every substitution rule so that
refactoring (e.g. compiling the regexes) can be verified to produce
identical output.

One subtlety worth noting: the quote normalization rule converts various
quote-like characters to U+2019 RIGHT SINGLE QUOTATION MARK (\u2019),
NOT to ASCII apostrophe U+0027 (').  The tests below reflect actual
behavior.
"""

import pytest
from stanza.utils.languages.ota.transliterate import ota_converter

# ---------------------------------------------------------------------------
# Deletion rules (fire before lowercasing)
# ---------------------------------------------------------------------------

def test_deletion_caron():
    assert ota_converter('ˇ') == ''

def test_deletion_breve():
    assert ota_converter('˘') == ''

def test_deletion_modifier_w():
    assert ota_converter('ʷ') == ''

def test_deletion_mid_word():
    """ Deleted characters are removed, not replaced, leaving surrounding chars intact """
    assert ota_converter('aˇb') == 'ab'

# ---------------------------------------------------------------------------
# Pre-lowercase uppercase substitutions
# These rules must fire before .lower() because they match specific
# uppercase codepoints that would be lost after lowercasing.
# ---------------------------------------------------------------------------

def test_pre_lower_capital_a_grave():
    """ À (U+00C0) -> â.  Must fire before lower() since à would remain à after lower(). """
    assert ota_converter('À') == 'â'

def test_pre_lower_ae():
    assert ota_converter('æ') == 's'

def test_pre_lower_dotless_capital_i():
    """ İ (U+0130) -> i.  Turkish dotless capital I. """
    assert ota_converter('İ') == 'i'

def test_pre_lower_capital_i_macron():
    """ Ī (U+012A) -> î """
    assert ota_converter('Ī') == 'î'

def test_pre_lower_capital_c_cedilla():
    assert ota_converter('Ç') == 'ç'

def test_pre_lower_capital_s_cedilla():
    assert ota_converter('Ş') == 'ş'

# ---------------------------------------------------------------------------
# Lowercasing
# ---------------------------------------------------------------------------

def test_lowercasing_ascii():
    assert ota_converter('HELLO') == 'hello'

def test_lowercasing_combined_with_macron_a():
    """ Pādişāh: uppercase P, macron-a -> exercises lowercasing + ā->â rule """
    assert ota_converter('Pādişāh') == 'pâdişâh'

# ---------------------------------------------------------------------------
# Post-lowercase rules
# ---------------------------------------------------------------------------

def test_pilcrow_i_deletion():
    """ ¶i -> empty string """
    assert ota_converter('¶i') == ''

def test_pilcrow_i_deletion_mid_word():
    assert ota_converter('a¶ib') == 'ab'

# ---------------------------------------------------------------------------
# Quote normalization -> U+2019 RIGHT SINGLE QUOTATION MARK
# Note: target is \u2019, not ASCII apostrophe \u0027
# ---------------------------------------------------------------------------

def test_quote_left_single():
    assert ota_converter('\u2018') == '\u2019'

def test_quote_e_grave():
    """ è (U+00E8) -> \u2019 in OTA convention """
    assert ota_converter('è') == '\u2019'

def test_quote_e_circumflex():
    assert ota_converter('ê') == '\u2019'

def test_quote_modifier_ain():
    assert ota_converter('ʿ') == '\u2019'

def test_quote_greek_rough_breathing():
    assert ota_converter('῾') == '\u2019'

def test_quote_modifier_glottal_stop():
    assert ota_converter('ˀ') == '\u2019'

def test_quote_greek_koronis():
    assert ota_converter('᾽') == '\u2019'

def test_quote_right_single_passthrough():
    """ \u2019 is already the target form, should survive unchanged """
    assert ota_converter('\u2019') == '\u2019'

# ---------------------------------------------------------------------------
# Letter substitutions
# ---------------------------------------------------------------------------

def test_d_underdot():
    assert ota_converter('ḍ') == 'd'

def test_e_macron_to_d():
    """ ē (U+0113) -> d in OTA convention """
    assert ota_converter('ē') == 'd'

def test_g_dot():
    assert ota_converter('ġ') == 'g'

def test_k_underdot():
    assert ota_converter('ḳ') == 'k'

def test_k_cedilla():
    assert ota_converter('ķ') == 'k'

def test_u_acute_to_k():
    """ ú -> k in OTA convention """
    assert ota_converter('ú') == 'k'

def test_h_loop():
    assert ota_converter('ẖ') == 'h'

def test_g_cedilla_to_h():
    assert ota_converter('ģ') == 'h'

def test_h_stroke():
    assert ota_converter('ħ') == 'h'

def test_h_circumflex():
    assert ota_converter('ĥ') == 'h'

def test_h_underdot():
    assert ota_converter('ḥ') == 'h'

def test_o_grave_to_h():
    assert ota_converter('ò') == 'h'

def test_h_subring():
    assert ota_converter('ḫ') == 'h'

def test_o_acute_to_h():
    assert ota_converter('ó') == 'h'

def test_a_macron():
    assert ota_converter('ā') == 'â'

def test_a_grave():
    assert ota_converter('à') == 'â'

def test_i_grave():
    assert ota_converter('ì') == 'î'

def test_i_macron():
    assert ota_converter('ī') == 'î'

def test_o_macron():
    assert ota_converter('ō') == 'ô'

def test_s_circumflex():
    assert ota_converter('ŝ') == 's'

def test_s_acute():
    assert ota_converter('ś') == 's'

def test_t_macron_below():
    """ ṯ (U+1E6F) -> s in OTA convention """
    assert ota_converter('ṯ') == 's'

def test_a_ring():
    assert ota_converter('å') == 's'

def test_a_tilde():
    assert ota_converter('ã') == 's'

def test_a_umlaut():
    assert ota_converter('ä') == 's'

def test_s_underdot():
    assert ota_converter('ṣ') == 's'

def test_u_grave_to_t():
    assert ota_converter('ù') == 't'

def test_w_umlaut_to_t():
    assert ota_converter('ẅ') == 't'

def test_t_underdot():
    assert ota_converter('ṭ') == 't'

def test_s_caron_to_t():
    assert ota_converter('š') == 't'

def test_u_macron():
    assert ota_converter('ū') == 'û'

def test_y_umlaut():
    assert ota_converter('ÿ') == 'û'

def test_z_underdot():
    assert ota_converter('ẓ') == 'z'

def test_z_acute():
    assert ota_converter('ź') == 'z'

def test_o_stroke():
    assert ota_converter('ø') == 'z'

def test_z_dot():
    assert ota_converter('ż') == 'z'

def test_z_line_below():
    assert ota_converter('ẕ') == 'z'

def test_z_caron():
    assert ota_converter('ž') == 'z'

def test_o_tilde_to_z():
    assert ota_converter('õ') == 'z'

def test_e_dot():
    assert ota_converter('ė') == 'e'

def test_n_caron():
    assert ota_converter('ň') == 'n'

def test_n_eng():
    assert ota_converter('ŋ') == 'n'

# ---------------------------------------------------------------------------
# Passthrough: characters that should not be modified
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", [
    'pâdişâh',   # already in normalized OTA form
    'gazel',
    'kitâb',
    'unban',
    '123',
    'hello',
    'â', 'î', 'ô', 'û',   # target forms should survive unchanged
    'ç', 'ş',              # cedilla forms that are the target, not the input
])
def test_passthrough(word):
    assert ota_converter(word) == word

# ---------------------------------------------------------------------------
# Multi-character / word-level integration cases
# ---------------------------------------------------------------------------

def test_word_gazel():
    """ ġazel: dotted-g word, common in OTA texts """
    assert ota_converter('ġazel') == 'gazel'

def test_word_kalem():
    """ ḳalem: underdot-k word """
    assert ota_converter('ḳalem') == 'kalem'

def test_word_padisah():
    """ Full word combining uppercase, macron-a, cedilla-s """
    assert ota_converter('Pādişāh') == 'pâdişâh'

def test_word_kitab():
    assert ota_converter('kitāb') == 'kitâb'
