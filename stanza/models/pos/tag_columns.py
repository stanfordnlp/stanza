"""
Descriptions of the tag columns a POS model predicts.

A TagColumn names one thing the model predicts and says where to find
it in a Document.  UPOS, XPOS, and UFeats come first, in that order,
and UPOS is required to be first of all, since the other columns can
be conditioned on it (see the chain inference in model.py).
Everything after those is an extra column: an additional tagset which
some, but generally not all, of the training files supply.

Extra columns are read from the MISC field of a CoNLL-U file, since
there is no other place in the format to put them:

  1  ghar  _  _  _  _  _  _  _  BIS=NN

and are declared on the command line as

  --extra_tag_columns bis=BIS

or, for several of them,

  --extra_tag_columns lines_xpos=LinesXPOS;partut_xpos=PartUTXPOS

The name on the left becomes the vocab key and the model's head name;
the key on the right is what to look for in MISC, and defaults to the
name if left off.
"""

from collections import namedtuple
from enum import Enum

from stanza.models.common.doc import UPOS, XPOS, FEATS, MISC

class TagKind(Enum):
    """
    Which kind of vocab to build for a column.

    WORD:     one flat label per word.  UPOS.
    AUTO:     inspect the data and use either a flat vocab or a
              CompositeVocab with a separator.  XPOS, and any extra
              column, since an unknown tagset is exactly the case
              choose_simplest_factory was written for.
    FEATURES: keyed, '|' separated, multiple values per word.  UFeats.
    """
    WORD     = 1
    AUTO     = 2
    FEATURES = 3

# name:     vocab key, model head name, and how the user refers to it
# field:    which Document field to read
# misc_key: if set, the column lives in MISC under this key
# kind:     which vocab to build
# output:   whether predictions for this column are written back to
#           the Document and scored.  Only the three native CoNLL-U
#           columns have a place to be written to, so extra columns
#           are trained but not emitted.
TagColumn = namedtuple("TagColumn", ["name", "field", "misc_key", "kind", "output"])

UPOS_COLUMN  = TagColumn("upos",  UPOS,  None, TagKind.WORD,     True)
XPOS_COLUMN  = TagColumn("xpos",  XPOS,  None, TagKind.AUTO,     True)
FEATS_COLUMN = TagColumn("feats", FEATS, None, TagKind.FEATURES, True)

DEFAULT_TAG_COLUMNS = (UPOS_COLUMN, XPOS_COLUMN, FEATS_COLUMN)

RESERVED_NAMES = frozenset(["char", "word"] + [x.name for x in DEFAULT_TAG_COLUMNS])

def parse_extra_tag_columns(spec):
    """
    Turn "bis=BIS;lines_xpos=LinesXPOS" into a tuple of TagColumn

    A bare "bis" is shorthand for "bis=bis".
    """
    if not spec:
        return ()

    columns = []
    seen = set()
    for piece in spec.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        name, _, misc_key = piece.partition("=")
        name = name.strip()
        misc_key = misc_key.strip() or name
        if not name:
            raise ValueError("Empty tag column name in --extra_tag_columns: %s" % spec)
        if name in RESERVED_NAMES:
            raise ValueError("Cannot use '%s' as an extra tag column: that name is already taken" % name)
        if name in seen:
            raise ValueError("Tag column '%s' listed twice in --extra_tag_columns" % spec)
        seen.add(name)
        columns.append(TagColumn(name, MISC, misc_key, TagKind.AUTO, False))
    return tuple(columns)

def build_tag_columns(extra_spec=None):
    """The three native columns, plus whatever extras were asked for"""
    return DEFAULT_TAG_COLUMNS + parse_extra_tag_columns(extra_spec)

def tag_columns_to_config(columns):
    """
    Flatten to plain strings so the columns survive a round trip through
    the model file.

    Models are loaded with weights_only=True, which will not unpickle a
    namedtuple or an Enum, so the config can only hold builtins.
    """
    return [[x.name, x.field, x.misc_key, x.kind.name, x.output] for x in columns]

def tag_columns_from_config(config):
    """
    Rebuild the columns from what tag_columns_to_config wrote.

    A model file with no such entry predates extra columns, and gets
    the default three.  Columns which are already TagColumn are passed
    through, so this is safe to call on an args dict whether or not it
    has been through a model file.
    """
    if not config:
        return DEFAULT_TAG_COLUMNS
    if isinstance(config[0], TagColumn):
        return tuple(config)
    return tuple(TagColumn(name, field, misc_key, TagKind[kind], output)
                 for name, field, misc_key, kind, output in config)

def tag_columns_from_args(args):
    """Pull the columns out of an args dict, defaulting to the native three"""
    if args is None:
        return DEFAULT_TAG_COLUMNS
    return tag_columns_from_config(args.get('tag_columns'))

def extract_misc_value(misc, key):
    """
    Read one key=value pair out of a MISC field, '_' if it isn't there

    '_' rather than None so that it lines up with resolve_none and with
    the way an empty native column is represented.
    """
    if not misc or misc == '_':
        return '_'
    for piece in misc.split("|"):
        piece_key, sep, piece_value = piece.partition("=")
        if sep and piece_key == key:
            return piece_value if piece_value else '_'
    return '_'
