"""
Convert CoNLL-U dependency trees into tikz-dependency LaTeX for papers.

The tikz-dependency documentation is at
https://ctan.org/pkg/tikz-dependency

The reader here is deliberately more lenient than stanza.utils.conll,
so that it can handle both a full treebank sentence and a hand-trimmed
excerpt pasted out of one:

  - token IDs do not need to be 1..N or contiguous
  - heads which point outside the selected tokens are skipped
    (or drawn as root arrows with --external-heads root)
  - multi-word token ranges (3-4) and empty nodes (5.1) are ignored
  - columns may be separated by spaces instead of tabs

Typical uses:

  # every sentence in a file, all tokens
  python3 -m stanza.utils.visualization.conll_to_tikz sents.conllu

  # one sentence, tokens 3 through 7 plus 10
  python3 -m stanza.utils.visualization.conll_to_tikz train.conllu --sent-id sd_001 --indices 3-7,10

  # an excerpt with arbitrary IDs, Sindhi forms drawn right to left
  python3 -m stanza.utils.visualization.conll_to_tikz excerpt.conllu --language sindhi

--language wraps each form in \\foreignlanguage, and the drawing
direction is guessed from it using Stanza's list of right to left
languages.  --rtl or --ltr overrides the guess, such as for a babel
language name Stanza doesn't recognize.

Glosses for the second row come from Gloss= in the MISC column, or from
--glosses, which lists them in sentence order for the selected tokens.
The second row falls back to just the UPOS when no gloss is available.

Arcs into the tokens listed in a '# highlight tokens = X Y Z' comment
are drawn in a different color.

Output can be a bare fragment (default) or a compilable document with
--standalone.  With --language, the document uses babel and needs
xelatex or lualatex.

Trees too wide for the text column can go in a float:

  # figure with caption and label, shrunk to the column width if needed
  python3 -m stanza.utils.visualization.conll_to_tikz train.conllu --sent-id sd_001 --float figure \\
      --caption 'Example parse of {text}' --label 'fig:{sent_id}' --fit-width

--float figure* spans both columns of a two-column paper, and
--float sidewaysfigure (from the rotating package) turns the tree
sideways on its own page.  --fit-width uses adjustbox to shrink a tree
to \\linewidth, and leaves narrower trees at their natural size.

--caption, --label and --second-row are templates.  Only the {field}
names they list are substituted, so other LaTeX braces pass through
unchanged, eg --second-row '\\textit{{gloss}} / {upos}'.
"""

import argparse
import re
import sys

from stanza.models.common.constant import is_right_to_left, UnknownLanguageError

DEFAULT_EDGE_STYLE = "blue!60!black,ultra thick"
DEFAULT_LABEL_STYLE = "fill=green!60,font=\\bfseries,text=black"
DEFAULT_HIGHLIGHT_EDGE_STYLE = "red!70!black,ultra thick"
DEFAULT_HIGHLIGHT_LABEL_STYLE = "fill=red!30,font=\\bfseries,text=black"

PREAMBLE = r"""% This is the package for the dependency drawings
% documentation is at
% https://ctan.math.illinois.edu/graphics/pgf/contrib/tikz-dependency/tikz-dependency-doc.pdf
\usepackage{tikz}
\usepackage{tikz-dependency}
\tikzset{/depgraph/.cd,/depgraph/.search also = {/tikz},
    baseline=-0.6ex, inner sep=-0.1cm, edge horizontal padding=3pt, edge unit distance=1.5ex}
"""


class Token:
    def __init__(self, idx, form, lemma, upos, xpos, feats, head, deprel, misc, line_num):
        self.id = idx
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.misc = misc
        self.line_num = line_num

    def misc_value(self, key):
        if self.misc in ("_", ""):
            return None
        for piece in self.misc.split("|"):
            if "=" in piece:
                k, v = piece.split("=", 1)
                if k == key:
                    return v
        return None


class Sentence:
    def __init__(self):
        self.comments = []
        self.tokens = []

    def comment_value(self, key):
        """Value of a '# key = value' comment, or None"""
        for comment in self.comments:
            m = re.match(r"#\s*" + re.escape(key) + r"\s*=\s*(.*)$", comment)
            if m:
                return m.group(1).strip()
        return None

    @property
    def sent_id(self):
        return self.comment_value("sent_id")

    def highlighted(self):
        """IDs listed in '# highlight tokens = X Y Z'"""
        value = self.comment_value("highlight tokens")
        if not value:
            return set()
        return {int(x) for x in value.replace(",", " ").split()}


def read_conllu(lines):
    sentences = []
    current = Sentence()
    for line_num, line in enumerate(lines, start=1):
        line = line.rstrip("\r\n")
        if not line.strip():
            if current.tokens or current.comments:
                sentences.append(current)
            current = Sentence()
            continue
        if line.startswith("#"):
            current.comments.append(line)
            continue
        fields = line.split("\t")
        if len(fields) < 8:
            # tolerate space-separated hand-typed excerpts
            fields = line.split()
        if len(fields) < 8:
            raise ValueError("Line %d has %d columns, expected at least 8: %s" % (line_num, len(fields), line))
        fields = fields + ["_"] * (10 - len(fields))
        if "-" in fields[0] or "." in fields[0]:
            # MWT range or empty node
            continue
        try:
            idx = int(fields[0])
        except ValueError:
            raise ValueError("Line %d: cannot read token id %r" % (line_num, fields[0]))
        head = None if fields[6] == "_" else int(fields[6])
        current.tokens.append(Token(idx, fields[1], fields[2], fields[3], fields[4], fields[5],
                                    head, fields[7], fields[9], line_num))
    if current.tokens or current.comments:
        sentences.append(current)
    # a file of only comments shouldn't produce empty drawings
    return [s for s in sentences if s.tokens]


def parse_indices(text):
    """'3-7,10 12' -> {3,4,5,6,7,10,12}"""
    result = set()
    for piece in re.split(r"[,\s]+", text.strip()):
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            start, end = int(start), int(end)
            if end < start:
                raise ValueError("Backwards range %s" % piece)
            result.update(range(start, end + 1))
        else:
            result.add(int(piece))
    return result


LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"{\char38}",   # \& is the deptext column separator
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text):
    return "".join(LATEX_SPECIALS.get(c, c) for c in text)


def fill_template(template, values):
    """
    Replace {name} for each name in values, leaving all other braces alone

    str.format would require doubling every LaTeX brace in the template
    """
    pattern = re.compile(r"\{(" + "|".join(re.escape(k) for k in values) + r")\}")
    return pattern.sub(lambda m: values[m.group(1)], template)


def second_row_text(token, gloss, fmt):
    if fmt is None:
        if gloss:
            return "%s / %s" % (latex_escape(gloss), latex_escape(token.upos))
        return latex_escape(token.upos)
    values = {
        "id": str(token.id),
        "form": token.form,
        "lemma": token.lemma,
        "upos": token.upos,
        "xpos": token.xpos,
        "feats": token.feats,
        "deprel": token.deprel,
        "gloss": gloss or "",
    }
    return fill_template(fmt, {k: latex_escape(v) for k, v in values.items()})


def guess_rtl(language):
    """
    Whether Stanza knows the language (name or code) to be right to left

    Unknown languages, such as babel names Stanza doesn't use, are left to right
    """
    if not language:
        return False
    try:
        return is_right_to_left(language)
    except UnknownLanguageError:
        return False


def wrap_foreign(text, language):
    text = latex_escape(text)
    if language:
        return "\\foreignlanguage{%s}{%s}" % (language, text)
    return text


def convert(sentence, args):
    tokens = sentence.tokens
    seen = set()
    for token in tokens:
        if token.id in seen:
            raise ValueError("Duplicate token id %d (line %d)" % (token.id, token.line_num))
        seen.add(token.id)

    if args.indices:
        wanted = parse_indices(args.indices)
        missing = wanted - seen
        if missing:
            raise ValueError("Requested indices not in sentence %s: %s" %
                             (sentence.sent_id or "", " ".join(str(x) for x in sorted(missing))))
        tokens = [t for t in tokens if t.id in wanted]
    tokens = sorted(tokens, key=lambda t: t.id)

    glosses = [None] * len(tokens)
    if args.glosses is not None:
        given = [g.strip() for g in args.glosses.split("|")]
        if len(given) != len(tokens):
            raise ValueError("--glosses has %d entries but %d tokens are selected" % (len(given), len(tokens)))
        glosses = given
    else:
        glosses = [t.misc_value(args.gloss_key) for t in tokens]

    language = args.language
    if args.rtl:
        rtl = True
    elif args.ltr:
        rtl = False
    else:
        rtl = guess_rtl(language)

    num = len(tokens)
    column = {}
    for position, token in enumerate(tokens, start=1):
        column[token.id] = num - position + 1 if rtl else position
    # the deptext columns are written left to right on the page
    display = sorted(zip(tokens, glosses), key=lambda tg: column[tg[0].id])

    highlighted = sentence.highlighted() if args.highlight else set()

    indent = "   "
    out = []
    out.append("\\begin{dependency}")
    out.append(indent + "\\begin{deptext}[column sep=%s]" % args.column_sep)
    out.append(indent * 2 + " \\& ".join(wrap_foreign(t.form, language) for t, _ in display) + " \\\\")
    if not args.no_second_row:
        cells = ["{%s %s}" % (args.second_row_size, second_row_text(t, g, args.second_row))
                 for t, g in display]
        out.append(indent * 2 + " \\& ".join(cells) + " \\\\")
    out.append(indent + "\\end{deptext}")

    def style_options(token):
        if token.id in highlighted:
            return args.highlight_edge_style, args.highlight_label_style
        return args.edge_style, args.label_style

    def options_text(token):
        edge_style, label_style = style_options(token)
        opts = []
        if edge_style:
            opts.append("edge style={%s}" % edge_style)
        if label_style:
            opts.append("label style={%s}" % label_style)
        if not opts:
            return ""
        return "[" + (",\n" + indent + " " * len("\\depedge[")).join(opts) + "]"

    # order edges by the dependent's position in the sentence
    for token in tokens:
        label = latex_escape(token.deprel)
        if token.head is None:
            continue
        if token.head == 0:
            if args.root:
                out.append(indent + "\\deproot%s{%d}{%s}" % (options_text(token), column[token.id], label))
            continue
        if token.head in column:
            out.append(indent + "\\depedge%s{%d}{%d}{%s}" %
                       (options_text(token), column[token.head], column[token.id], label))
        elif args.external_heads == "root":
            out.append(indent + "\\deproot%s{%d}{%s}" % (options_text(token), column[token.id], label))
    out.append("\\end{dependency}")
    return out


def wrap(dependency, sentence, number, args):
    """
    Put the dependency lines in a center environment or a float

    Inside a float, \\centering is used instead of center, which would
    add extra vertical space around the tree
    """
    if args.fit_width:
        dependency = (["\\begin{adjustbox}{max width=\\linewidth}"] + dependency +
                      ["\\end{adjustbox}"])
    if not args.float:
        if args.center:
            dependency = ["\\begin{center}"] + dependency + ["\\end{center}"]
        return "\n".join(dependency)

    sent_id = sentence.sent_id or str(number)
    caption_values = {
        "n": str(number),
        "sent_id": latex_escape(sent_id),
        "text": wrap_foreign(sentence.comment_value("text") or "", args.language),
    }
    # \label accepts _ and the like, and the label should match the sent_id exactly
    label_values = {"n": str(number), "sent_id": sent_id}
    if args.float == "sidewaysfigure":
        # rotating's sidewaysfigure always gets its own page, no placement
        out = ["\\begin{sidewaysfigure}"]
    else:
        out = ["\\begin{%s}[%s]" % (args.float, args.placement)]
    if args.center:
        out.append("\\centering")
    out.extend(dependency)
    if args.caption:
        out.append("\\caption{%s}" % fill_template(args.caption, caption_values))
    if args.label:
        out.append("\\label{%s}" % fill_template(args.label, label_values))
    out.append("\\end{%s}" % args.float)
    return "\n".join(out)


def standalone(body, args):
    lines = ["\\documentclass{article}"]
    if args.fit_width:
        lines.append("\\usepackage{adjustbox}")
    if args.float == "sidewaysfigure":
        lines.append("\\usepackage{rotating}")
    if args.language:
        # \foreignlanguage comes from babel.  babel's ini locales cover
        # Sindhi and friends, but non-Latin scripts need xelatex or lualatex
        lines.append("\\usepackage{fontspec}")
        lines.append("\\usepackage[english,bidi=default]{babel}")
        lines.append("\\babelprovide[import]{%s}" % args.language)
        if args.font:
            lines.append("\\babelfont[%s]{rm}{%s}" % (args.language, args.font))
    lines.append(PREAMBLE)
    lines.append("\\begin{document}")
    lines.append(body)
    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert CoNLL-U trees to tikz-dependency LaTeX")
    parser.add_argument("input", nargs="?", default="-", help="CoNLL-U file, or - for stdin (default)")
    parser.add_argument("-o", "--output", default="-", help="Output file, or - for stdout (default)")

    select = parser.add_argument_group("selection")
    select.add_argument("--sent-id", help="Only convert the sentence with this # sent_id")
    select.add_argument("--sentence", type=int, help="Only convert the Nth sentence in the file (1-based)")
    select.add_argument("--indices", help="Token IDs to keep, eg '3-7,10 12'.  Default: all tokens")

    text = parser.add_argument_group("text rows")
    text.add_argument("--language", help="Wrap each form in \\foreignlanguage{LANGUAGE}{...}")
    direction = text.add_mutually_exclusive_group()
    direction.add_argument("--rtl", action="store_true", help="Draw right to left.  Default: guessed from --language")
    direction.add_argument("--ltr", action="store_true", help="Draw left to right even for an RTL --language")
    text.add_argument("--glosses", help="'|'-separated glosses for the selected tokens, in sentence order")
    text.add_argument("--gloss-key", default="Gloss", help="MISC key to read glosses from (default: %(default)s)")
    text.add_argument("--second-row", default=None,
                      help="Format string for the second row, using {gloss} {upos} {xpos} {lemma} {form} {feats} {deprel} {id}."
                           "  Default: 'gloss / UPOS', or just UPOS without a gloss")
    text.add_argument("--second-row-size", default="\\small", help="Font size command for the second row (default: %(default)s)")
    text.add_argument("--no-second-row", action="store_true", help="Only draw the forms")
    text.add_argument("--column-sep", default="0.35cm", help="deptext column sep (default: %(default)s)")

    edges = parser.add_argument_group("edges")
    edges.add_argument("--root", action="store_true", help="Draw a \\deproot arrow for the sentence root")
    edges.add_argument("--external-heads", choices=["skip", "root"], default="skip",
                       help="What to do with tokens whose head was not selected (default: %(default)s)")
    edges.add_argument("--edge-style", default=DEFAULT_EDGE_STYLE, help="Edge style; empty string for the package default")
    edges.add_argument("--label-style", default=DEFAULT_LABEL_STYLE, help="Label style; empty string for the package default")
    edges.add_argument("--no-highlight", dest="highlight", action="store_false",
                       help="Ignore '# highlight tokens = ...' comments")
    edges.add_argument("--highlight-edge-style", default=DEFAULT_HIGHLIGHT_EDGE_STYLE,
                       help="Edge style for arcs into highlighted tokens")
    edges.add_argument("--highlight-label-style", default=DEFAULT_HIGHLIGHT_LABEL_STYLE,
                       help="Label style for arcs into highlighted tokens")

    output = parser.add_argument_group("output")
    output.add_argument("--no-center", dest="center", action="store_false",
                        help="Don't center the tree (no center environment, or no \\centering in a float)")
    output.add_argument("--float", choices=["figure", "figure*", "sidewaysfigure"],
                        help="Put each tree in a float.  sidewaysfigure needs \\usepackage{rotating}")
    output.add_argument("--placement", default="htbp", help="Float placement (default: %(default)s)")
    output.add_argument("--caption", help="Float caption template, using {text} {sent_id} {n}")
    output.add_argument("--label", help="Float label template, using {sent_id} {n}.  "
                                        "Needs a field when there are multiple sentences")
    output.add_argument("--fit-width", action="store_true",
                        help="Shrink trees wider than \\linewidth.  Needs \\usepackage{adjustbox}")
    output.add_argument("--standalone", action="store_true", help="Emit a complete document instead of a fragment")
    output.add_argument("--font", help="With --standalone and --language, the font for that language (eg 'Noto Naskh Arabic')")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.input == "-":
        lines = sys.stdin.readlines()
    else:
        with open(args.input, encoding="utf-8") as fin:
            lines = fin.readlines()
    sentences = read_conllu(lines)

    if args.sent_id is not None:
        sentences = [s for s in sentences if s.sent_id == args.sent_id]
        if not sentences:
            raise SystemExit("No sentence with sent_id %s" % args.sent_id)
    if args.sentence is not None:
        if not 1 <= args.sentence <= len(sentences):
            raise SystemExit("--sentence %d out of range: %d sentences" % (args.sentence, len(sentences)))
        sentences = [sentences[args.sentence - 1]]
    if not sentences:
        raise SystemExit("No sentences found")
    if args.glosses is not None and len(sentences) > 1:
        raise SystemExit("--glosses only makes sense with a single sentence; use --sent-id or --sentence")
    if args.label and len(sentences) > 1 and not re.search(r"\{(sent_id|n)\}", args.label):
        raise SystemExit("--label would be duplicated across %d sentences; include {sent_id} or {n}" % len(sentences))
    if (args.caption or args.label or args.placement != "htbp") and not args.float:
        raise SystemExit("--caption, --label and --placement only apply with --float")

    try:
        body = "\n\n".join(wrap(convert(s, args), s, n, args) for n, s in enumerate(sentences, start=1))
    except ValueError as e:
        raise SystemExit(str(e))
    result = standalone(body, args) if args.standalone else body + "\n"

    if args.output == "-":
        sys.stdout.write(result)
    else:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fout:
            fout.write(result)


if __name__ == "__main__":
    main()
