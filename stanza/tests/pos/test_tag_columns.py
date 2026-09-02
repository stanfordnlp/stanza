"""
Test the tag column descriptions used to configure the tagger's output layers
"""

import pytest

from stanza.models.common.doc import FEATS, MISC, UPOS, XPOS
from stanza.models.pos.tag_columns import (DEFAULT_TAG_COLUMNS, TagKind, build_tag_columns,
                                           extract_misc_value, parse_extra_tag_columns,
                                           parse_tag_column_parents, set_misc_value,
                                           tag_column_eval_order,
                                           tag_columns_from_config, tag_columns_to_config,
                                           validate_tag_columns)

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
    assert all(isinstance(x, (str, bool, list, type(None))) for row in config for x in row)
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


def test_default_parents():
    """upos is the root and everything else hangs off it"""
    columns = build_tag_columns("bis=BIS")
    assert {x.name: x.parents for x in columns} == {
        'upos': (), 'xpos': ('upos',), 'feats': ('upos',), 'bis': ('upos',)}

def test_parents_spec():
    columns = build_tag_columns("bis=BIS", "bis=upos;xpos=bis")
    parents = {x.name: x.parents for x in columns}
    assert parents['bis'] == ('upos',)
    assert parents['xpos'] == ('bis',)
    assert parents['feats'] == ('upos',)

def test_multiple_parents():
    columns = build_tag_columns("bis=BIS", "xpos=upos,bis")
    assert {x.name: x.parents for x in columns}['xpos'] == ('upos', 'bis')

def test_eval_order_follows_parents():
    """
    A column may be conditioned on one declared after it

    The declared order lines the tags up with the predictions; only the
    computation has to follow the parents.
    """
    columns = build_tag_columns("bis=BIS", "xpos=bis")
    assert [x.name for x in columns] == ['upos', 'xpos', 'feats', 'bis']
    order = tag_column_eval_order(columns)
    assert order.index('bis') < order.index('xpos')
    assert order.index('upos') < order.index('bis')

def test_parents_rejects_bad_specs():
    with pytest.raises(ValueError):   # no such column
        build_tag_columns(None, "xpos=nope")
    with pytest.raises(ValueError):   # upos is the root
        build_tag_columns(None, "upos=xpos")
    with pytest.raises(ValueError):   # cycle
        build_tag_columns("bis=BIS", "xpos=bis;bis=xpos")
    with pytest.raises(ValueError):   # its own parent
        build_tag_columns(None, "xpos=xpos")
    with pytest.raises(ValueError):   # nothing to hang off
        build_tag_columns(None, "xpos=")
    with pytest.raises(ValueError):   # not a child=parent pair
        build_tag_columns(None, "xpos")
    with pytest.raises(ValueError):   # listed twice
        build_tag_columns(None, "xpos=upos;xpos=feats")

def test_parents_survive_the_config():
    columns = build_tag_columns("bis=BIS", "bis=upos;xpos=bis")
    config = tag_columns_to_config(columns)
    assert all(isinstance(x, (str, bool, list, type(None))) for row in config for x in row)
    assert tag_columns_from_config(config) == columns

def test_config_without_parents():
    """A model file written before the columns had parents gets the default arrangement"""
    columns = build_tag_columns("bis=BIS")
    config = [row[:5] for row in tag_columns_to_config(columns)]
    assert tag_columns_from_config(config) == columns


def test_set_misc_value():
    assert set_misc_value("_", "BIS", "NN") == "BIS=NN"
    assert set_misc_value(None, "BIS", "NN") == "BIS=NN"
    assert set_misc_value("SpaceAfter=No", "BIS", "NN") == "SpaceAfter=No|BIS=NN"
    # an existing value for the key is replaced, not repeated
    assert set_misc_value("BIS=VM", "BIS", "NN") == "BIS=NN"
    assert set_misc_value("SpaceAfter=No|BIS=VM", "BIS", "NN") == "SpaceAfter=No|BIS=NN"
    # nothing to say means the key is left out entirely
    assert set_misc_value("SpaceAfter=No", "BIS", "_") == "SpaceAfter=No"
    assert set_misc_value("BIS=VM", "BIS", "_") == "_"
    # a key which merely starts the same is left alone
    assert set_misc_value("BISON=1", "BIS", "NN") == "BISON=1|BIS=NN"

def test_misc_round_trip():
    misc = set_misc_value("start_char=0|end_char=4", "BIS", "NNP")
    assert extract_misc_value(misc, "BIS") == "NNP"
    assert extract_misc_value(misc, "start_char") == "0"
