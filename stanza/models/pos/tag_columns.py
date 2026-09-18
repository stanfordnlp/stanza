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
# parents:  which columns this one is conditioned on.  upos has none,
#           as it is the root; everything else defaults to upos.
TagColumn = namedtuple("TagColumn", ["name", "field", "misc_key", "kind", "output", "parents"])

UPOS_COLUMN  = TagColumn("upos",  UPOS,  None, TagKind.WORD,     True,  ())
XPOS_COLUMN  = TagColumn("xpos",  XPOS,  None, TagKind.AUTO,     True,  ("upos",))
FEATS_COLUMN = TagColumn("feats", FEATS, None, TagKind.FEATURES, True,  ("upos",))

DEFAULT_TAG_COLUMNS = (UPOS_COLUMN, XPOS_COLUMN, FEATS_COLUMN)

# how a column is told about its parents.  TAG_EMB embeds the parent's
# predicted (or gold, while training) tag; HIDDEN feeds the parent's
# hidden layer, which is wider and carries a gradient back into the
# parent's head.
TAG_LINK_EMB = "tag_emb"
TAG_LINK_HIDDEN = "hidden"
TAG_LINKS = (TAG_LINK_EMB, TAG_LINK_HIDDEN)

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
        columns.append(TagColumn(name, MISC, misc_key, TagKind.AUTO, False, ("upos",)))
    return tuple(columns)

def parse_tag_column_parents(spec, columns):
    """
    Apply a "child=parent,parent;child=parent" spec to a list of columns

    Every column is conditioned on upos unless it says otherwise.  A
    column may be conditioned on any other column instead, or on
    several, as long as the result is acyclic - see model.py for what
    the conditioning actually feeds forward.
    """
    if not spec:
        return tuple(columns)

    by_name = {x.name: x for x in columns}
    parents = {}
    for piece in spec.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        child, sep, names = piece.partition("=")
        child = child.strip()
        if not sep:
            raise ValueError("--tag_column_parents needs child=parent, got '%s'" % piece)
        if child not in by_name:
            raise ValueError("Unknown tag column '%s' in --tag_column_parents" % child)
        if child in parents:
            raise ValueError("Tag column '%s' listed twice in --tag_column_parents" % child)
        if child == UPOS_COLUMN.name:
            raise ValueError("Cannot give '%s' a parent: it is the root of the tag columns" % child)
        parents[child] = tuple(x.strip() for x in names.split(",") if x.strip())

    columns = tuple(x._replace(parents=parents.get(x.name, x.parents)) for x in columns)
    validate_tag_columns(columns)
    return columns

def validate_tag_columns(columns):
    """Check that the columns name a sensible acyclic set of connections"""
    names = [x.name for x in columns]
    if not names or names[0] != UPOS_COLUMN.name:
        raise ValueError("The first tag column must be %s, got %s" % (UPOS_COLUMN.name, names))
    known = set(names)
    for column in columns:
        if column.name == UPOS_COLUMN.name:
            if column.parents:
                raise ValueError("'%s' is the root and cannot have parents" % column.name)
            continue
        if not column.parents:
            raise ValueError("Tag column '%s' has no parents.  Use %s to hang it off the root" %
                             (column.name, UPOS_COLUMN.name))
        for parent in column.parents:
            if parent not in known:
                raise ValueError("Tag column '%s' has an unknown parent '%s'" % (column.name, parent))
            if parent == column.name:
                raise ValueError("Tag column '%s' cannot be its own parent" % column.name)
    # a cycle shows up here as a name which can never be emitted
    tag_column_eval_order(columns)
    return columns

def tag_column_eval_order(columns):
    """
    The order the heads have to be computed in, so a parent is always ready

    The declared order of the columns is what lines the tags and the
    predictions up with each other; this is only about computation, so
    a column may be conditioned on one declared after it.
    """
    parents = {x.name: x.parents for x in columns}
    order = []
    done = set()
    visiting = set()

    def visit(name, path):
        if name in done:
            return
        if name in visiting:
            raise ValueError("Tag columns have a cycle in their parents: %s" % " -> ".join(path + [name]))
        visiting.add(name)
        for parent in parents[name]:
            visit(parent, path + [name])
        visiting.discard(name)
        done.add(name)
        order.append(name)

    for column in columns:
        visit(column.name, [])
    return order

def build_tag_columns(extra_spec=None, parent_spec=None):
    """The three native columns, plus whatever extras were asked for"""
    columns = DEFAULT_TAG_COLUMNS + parse_extra_tag_columns(extra_spec)
    return parse_tag_column_parents(parent_spec, columns)

def tag_columns_to_config(columns):
    """
    Flatten to plain strings so the columns survive a round trip through
    the model file.

    Models are loaded with weights_only=True, which will not unpickle a
    namedtuple or an Enum, so the config can only hold builtins.
    """
    return [[x.name, x.field, x.misc_key, x.kind.name, x.output, list(x.parents)] for x in columns]

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
    columns = []
    for row in config:
        name, field, misc_key, kind, output = row[:5]
        parents = tuple(row[5]) if len(row) > 5 else (() if name == UPOS_COLUMN.name else (UPOS_COLUMN.name,))
        columns.append(TagColumn(name, field, misc_key, TagKind[kind], output, parents))
    return tuple(columns)

def tag_columns_from_args(args):
    """Pull the columns out of an args dict, defaulting to the native three"""
    if args is None:
        return DEFAULT_TAG_COLUMNS
    return tag_columns_from_config(args.get('tag_columns'))

def set_misc_value(misc, key, value):
    """
    Put key=value into a MISC field, leaving everything else in it alone

    An existing entry for the key is replaced rather than repeated, and
    a value of '_' means there is nothing to say, so the key is left out.
    """
    pieces = [] if not misc or misc == '_' else misc.split("|")
    pieces = [x for x in pieces if not x.startswith(key + "=")]
    if value and value != '_':
        pieces.append("%s=%s" % (key, value))
    return "|".join(pieces) if pieces else "_"

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
