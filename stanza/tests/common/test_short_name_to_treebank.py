import pytest

import stanza
from stanza.models.common import short_name_to_treebank

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def test_short_name():
    assert short_name_to_treebank.short_name_to_treebank("en_ewt") == "UD_English-EWT"

def test_canonical_name():
    assert short_name_to_treebank.canonical_treebank_name("UD_URDU-UDTB") == "UD_Urdu-UDTB"
    assert short_name_to_treebank.canonical_treebank_name("ur_udtb") == "UD_Urdu-UDTB"
    assert short_name_to_treebank.canonical_treebank_name("Unban_Mox_Opal") == "Unban_Mox_Opal"
