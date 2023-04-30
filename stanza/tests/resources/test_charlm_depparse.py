import pytest

from stanza.resources.print_charlm_depparse import list_depparse
from stanza.resources.prepare_resources import default_charlms, depparse_charlms

def test_list_depparse():
    models = list_depparse()

    # check that it's picking up the models which don't have specific charlms
    # first, make sure the default assumption of the test is still true...
    # if this test fails, find a different language which isn't in depparse_charlms
    assert "af" not in depparse_charlms
    assert "af" in default_charlms
    assert "af_afribooms" in models

    # assert that it's picking up the models which do have specific charlms that aren't None
    # again, first make sure the default assumptions are true
    # if one of these next few tests fail, just update the test
    assert "en" in depparse_charlms
    assert "en" in default_charlms
    assert "ewt" not in depparse_charlms["en"]
    assert "craft" in depparse_charlms["en"]
    assert "mimic" in depparse_charlms["en"]
    # now, check the results
    assert "en_ewt" in models
    assert "en_mimic" in models
    assert "en_craft" not in models
