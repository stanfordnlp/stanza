"""
Test the tag column descriptions used to configure the tagger's output layers
"""

import pytest

from stanza.models.common.doc import FEATS, MISC, UPOS, XPOS
from stanza.models.pos.tag_columns import (DEFAULT_TAG_COLUMNS, TagKind, build_tag_columns,
                                           extract_misc_value, parse_extra_tag_columns,
                                           tag_columns_from_config, tag_columns_to_config)

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_default_columns():
    assert [x.name for x in DEFAULT_TAG_COLUMNS] == ['upos', 'xpos', 'feats']
    assert [x.field for x in DEFAULT_TAG_COLUMNS] == [UPOS, XPOS, FEATS]
    assert all(x.output for x in DEFAULT_TAG_COLUMNS)
    assert all(x.misc_key is None for x in DEFAULT_TAG_COLUMNS)

def test_parse_empty():
    assert parse_extra_tag_columns(None) == ()
    assert parse_extra_tag_columns("") == ()

def test_parse_bare_name():
    columns = parse_extra_tag_columns("bis")
    assert len(columns) == 1
    assert columns[0].name == 'bis'
    assert columns[0].misc_key == 'bis'
    assert columns[0].field == MISC
    assert columns[0].kind is TagKind.AUTO
    # an extra tagset has nowhere in conllu to be written
    assert not columns[0].output

def test_parse_misc_key():
    columns = parse_extra_tag_columns("bis=BIS")
    assert columns[0].name == 'bis'
    assert columns[0].misc_key == 'BIS'

def test_parse_several():
    columns = parse_extra_tag_columns("lines_xpos=LinesXPOS;partut_xpos=PartUTXPOS")
    assert [x.name for x in columns] == ['lines_xpos', 'partut_xpos']
    assert [x.misc_key for x in columns] == ['LinesXPOS', 'PartUTXPOS']

def test_parse_rejects_collisions():
    with pytest.raises(ValueError):
        parse_extra_tag_columns("xpos=Whatever")
    with pytest.raises(ValueError):
        parse_extra_tag_columns("word")
    with pytest.raises(ValueError):
        parse_extra_tag_columns("bis=BIS;bis=OtherBIS")

def test_build_tag_columns():
    columns = build_tag_columns("bis=BIS")
    assert [x.name for x in columns] == ['upos', 'xpos', 'feats', 'bis']
    assert build_tag_columns(None) == DEFAULT_TAG_COLUMNS

def test_config_round_trip():
    """
    The config has to survive torch.load(weights_only=True), so it can
    only contain builtins
    """
    columns = build_tag_columns("bis=BIS")
    config = tag_columns_to_config(columns)
    assert all(isinstance(x, (str, bool, type(None))) for row in config for x in row)
    assert tag_columns_from_config(config) == columns

def test_config_default_for_old_models():
    """A model file with no tag_columns entry gets the default three"""
    assert tag_columns_from_config(None) == DEFAULT_TAG_COLUMNS
    assert tag_columns_from_config([]) == DEFAULT_TAG_COLUMNS

def test_extract_misc_value():
    assert extract_misc_value("BIS=NN", "BIS") == "NN"
    assert extract_misc_value("start_char=0|end_char=8|BIS=NNP", "BIS") == "NNP"
    assert extract_misc_value("start_char=0|end_char=8", "BIS") == "_"
    assert extract_misc_value("_", "BIS") == "_"
    assert extract_misc_value(None, "BIS") == "_"
    # SpaceAfter=No shouldn't be mistaken for a value of a key it merely contains
    assert extract_misc_value("SpaceAfter=No", "Space") == "_"
