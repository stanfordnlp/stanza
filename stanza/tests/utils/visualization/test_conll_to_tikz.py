"""
Tests for the CoNLL-U to tikz-dependency converter

These check the generated LaTeX text.  Compiling it is left to LaTeX.
"""

import re

import pytest

from stanza.utils.visualization import conll_to_tikz
from stanza.utils.visualization.conll_to_tikz import convert, fill_template, parse_args, parse_indices, read_conllu, wrap

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

SINDHI = """
# sent_id = sd_test
# text = ھي ڇوڪرو آيو
1	ھي	ھي	DET	_	_	2	det	_	Gloss=This
2	ڇوڪرو	ڇوڪرو	NOUN	_	_	3	nsubj	_	Gloss=boy
3	آيو	اچڻ	VERB	_	_	0	root	_	Gloss=came
""".lstrip()

ENGLISH = """
# sent_id = en_test_1
# text = The dog barked loudly.
1	The	the	DET	DT	_	2	det	_	_
2	dog	dog	NOUN	NN	_	3	nsubj	_	_
3	barked	bark	VERB	VBD	_	0	root	_	_
4	loudly	loudly	ADV	RB	_	3	advmod	_	SpaceAfter=No
5	.	.	PUNCT	.	_	3	punct	_	_
""".lstrip()

# a hand-trimmed excerpt: IDs start at 12, spaces instead of tabs,
# an MWT line and an empty node, and a head (9) outside the excerpt
EXCERPT = """
# highlight tokens = 14
12 Smith Smith PROPN NNP _ 14 nsubj _ _
13 & & CCONJ CC _ 9 cc _ _
14-15 sawher _ _ _ _ _ _ _ _
14 saw see VERB VBD _ 0 root _ _
14.1 saw see VERB VBD _ _ _ _ _
15 her she PRON PRP _ 14 obj _ _
""".lstrip()


def to_tikz(text, *flags):
    """Convert the first sentence of text with the given command line flags"""
    args = parse_args(["-"] + list(flags))
    sentence = read_conllu(text.splitlines(keepends=True))[0]
    return wrap(convert(sentence, args), sentence, 1, args)


EDGE_RE = re.compile(r"\\(depedge|deproot)(?:\[[^\]]*\])?((?:\{[^}]*\})+)")

def edges(tikz):
    """The \\depedge and \\deproot commands in the output, without their styles"""
    return ["\\%s%s" % match for match in EDGE_RE.findall(tikz)]


def test_read_excerpt():
    sentences = read_conllu(EXCERPT.splitlines(keepends=True))
    assert len(sentences) == 1
    sentence = sentences[0]
    assert [t.id for t in sentence.tokens] == [12, 13, 14, 15]
    assert [t.head for t in sentence.tokens] == [14, 9, 0, 14]
    assert sentence.highlighted() == {14}


def test_read_multiple():
    sentences = read_conllu((SINDHI + "\n" + ENGLISH).splitlines(keepends=True))
    assert [s.sent_id for s in sentences] == ["sd_test", "en_test_1"]
    assert sentences[1].comment_value("text") == "The dog barked loudly."


def test_read_errors():
    with pytest.raises(ValueError):
        read_conllu(["1\tThe\tthe\n"])
    with pytest.raises(ValueError):
        read_conllu(["x\tThe\tthe\tDET\t_\t_\t0\troot\t_\t_\n"])


def test_parse_indices():
    assert parse_indices("3-5,10 12") == {3, 4, 5, 10, 12}
    assert parse_indices("7") == {7}
    with pytest.raises(ValueError):
        parse_indices("5-3")


def test_fill_template():
    # only the named fields change, so LaTeX braces survive
    assert fill_template("\\textit{{gloss}} / {upos} {other}", {"gloss": "dog", "upos": "NOUN"}) == "\\textit{dog} / NOUN {other}"


def test_ltr():
    tikz = to_tikz(ENGLISH)
    assert tikz.startswith("\\begin{center}\n\\begin{dependency}")
    assert tikz.endswith("\\end{dependency}\n\\end{center}")
    assert "The \\& dog \\& barked \\& loudly \\& . \\\\" in tikz
    assert "{\\small DET} \\& {\\small NOUN}" in tikz
    assert edges(tikz) == ["\\depedge{2}{1}{det}", "\\depedge{3}{2}{nsubj}",
                           "\\depedge{3}{4}{advmod}", "\\depedge{3}{5}{punct}"]
    assert edges(to_tikz(ENGLISH, "--root"))[2] == "\\deproot{3}{root}"


def test_rtl():
    """Sindhi is drawn right to left, so the columns are reversed"""
    tikz = to_tikz(SINDHI, "--language", "sindhi")
    assert "\\foreignlanguage{sindhi}{آيو} \\& \\foreignlanguage{sindhi}{ڇوڪرو} \\& \\foreignlanguage{sindhi}{ھي} \\\\" in tikz
    assert "{\\small came / VERB} \\& {\\small boy / NOUN} \\& {\\small This / DET} \\\\" in tikz
    assert edges(tikz) == ["\\depedge{2}{3}{det}", "\\depedge{1}{2}{nsubj}"]

    # the language code works too, and --ltr overrides the guess
    assert to_tikz(SINDHI, "--language", "sd") == tikz.replace("{sindhi}", "{sd}")
    assert edges(to_tikz(SINDHI, "--language", "sindhi", "--ltr")) == ["\\depedge{2}{1}{det}", "\\depedge{3}{2}{nsubj}"]


def test_guess_rtl():
    assert conll_to_tikz.guess_rtl("urdu")
    assert conll_to_tikz.guess_rtl("ur")
    assert not conll_to_tikz.guess_rtl("english")
    assert not conll_to_tikz.guess_rtl(None)
    assert not conll_to_tikz.guess_rtl("not a real language")


def test_excerpt_external_heads():
    tikz = to_tikz(EXCERPT, "--root")
    # & has to be escaped without using \&, the deptext column separator
    assert "Smith \\& {\\char38} \\& saw \\& her \\\\" in tikz
    # 13 has its head outside the excerpt, so it gets no arc
    assert edges(tikz) == ["\\depedge{3}{1}{nsubj}", "\\deproot{3}{root}", "\\depedge{3}{4}{obj}"]

    tikz = to_tikz(EXCERPT, "--external-heads", "root")
    assert edges(tikz) == ["\\depedge{3}{1}{nsubj}", "\\deproot{2}{cc}", "\\depedge{3}{4}{obj}"]


def test_highlight():
    tikz = to_tikz(EXCERPT, "--root")
    root_line = [x for x in tikz.split("\n") if "\\deproot" in x][0]
    assert conll_to_tikz.DEFAULT_HIGHLIGHT_EDGE_STYLE in root_line
    assert conll_to_tikz.DEFAULT_HIGHLIGHT_EDGE_STYLE not in to_tikz(EXCERPT, "--root", "--no-highlight")
    # without the comment, nothing is highlighted
    assert conll_to_tikz.DEFAULT_HIGHLIGHT_EDGE_STYLE not in to_tikz(ENGLISH, "--root")


def test_indices():
    tikz = to_tikz(ENGLISH, "--indices", "2-4")
    assert "dog \\& barked \\& loudly \\\\" in tikz
    assert edges(tikz) == ["\\depedge{2}{1}{nsubj}", "\\depedge{2}{3}{advmod}"]

    with pytest.raises(ValueError):
        to_tikz(ENGLISH, "--indices", "4-9")


def test_second_row():
    tikz = to_tikz(ENGLISH, "--second-row", "\\textit{{lemma}}/{xpos}")
    assert "{\\small \\textit{bark}/VBD}" in tikz
    tikz = to_tikz(ENGLISH, "--glosses", "the|dog|barked|loudly|.")
    assert "{\\small barked / VERB}" in tikz
    assert "\\small" not in to_tikz(ENGLISH, "--no-second-row")
    with pytest.raises(ValueError):
        to_tikz(ENGLISH, "--glosses", "the|dog")


def test_float():
    tikz = to_tikz(ENGLISH, "--float", "figure", "--caption", "Parse of \\textit{{text}} ({sent_id})", "--label", "fig:{sent_id}")
    lines = tikz.split("\n")
    assert lines[:3] == ["\\begin{figure}[htbp]", "\\centering", "\\begin{dependency}"]
    # the sent_id is escaped in the caption, but not in the label
    assert lines[-3:] == ["\\caption{Parse of \\textit{The dog barked loudly.} (en\\_test\\_1)}",
                          "\\label{fig:en_test_1}",
                          "\\end{figure}"]

    tikz = to_tikz(ENGLISH, "--float", "figure*", "--placement", "t", "--fit-width")
    lines = tikz.split("\n")
    assert lines[:3] == ["\\begin{figure*}[t]", "\\centering", "\\begin{adjustbox}{max width=\\linewidth}"]
    assert lines[-2:] == ["\\end{adjustbox}", "\\end{figure*}"]

    tikz = to_tikz(SINDHI, "--float", "sidewaysfigure", "--language", "sindhi", "--caption", "{text}")
    assert tikz.startswith("\\begin{sidewaysfigure}\n")
    assert "\\caption{\\foreignlanguage{sindhi}{ھي ڇوڪرو آيو}}" in tikz


def test_main(tmp_path):
    conll_file = tmp_path / "test.conllu"
    conll_file.write_text(SINDHI + "\n" + ENGLISH, encoding="utf-8")
    output_file = tmp_path / "test.tex"

    conll_to_tikz.main([str(conll_file), "-o", str(output_file), "--sent-id", "en_test_1", "--standalone", "--fit-width"])
    text = output_file.read_text(encoding="utf-8")
    assert text.startswith("\\documentclass{article}\n\\usepackage{adjustbox}\n")
    assert "\\usepackage{tikz-dependency}" in text
    assert text.count("\\begin{dependency}") == 1
    assert text.endswith("\\end{document}\n")

    conll_to_tikz.main([str(conll_file), "-o", str(output_file)])
    assert output_file.read_text(encoding="utf-8").count("\\begin{dependency}") == 2

    # a label used for every sentence would be duplicated
    with pytest.raises(SystemExit):
        conll_to_tikz.main([str(conll_file), "--float", "figure", "--label", "fig:tree"])
    with pytest.raises(SystemExit):
        conll_to_tikz.main([str(conll_file), "--caption", "no float"])
    with pytest.raises(SystemExit):
        conll_to_tikz.main([str(conll_file), "--sent-id", "missing"])
