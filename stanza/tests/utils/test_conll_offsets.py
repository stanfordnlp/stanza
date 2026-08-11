"""
Tests of rebuilding the text and the character offsets of a document read from CoNLL-U
"""

import warnings

import pytest

from stanza.utils.conll import CoNLL
from stanza.tests import *

pytestmark = pytest.mark.pipeline

# an MWT which splits into pieces of the surface form, a couple SpaceAfter=No,
# and a SpacesAfter at the end of the sentence
ENGLISH = """
# sent_id = 1
# text = She can't swim, sadly.
1\tShe\tshe\tPRON\tPRP\t_\t3\tnsubj\t_\t_
2-3\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
2\tca\tcan\tAUX\tMD\t_\t4\taux\t_\t_
3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\t_
4\tswim\tswim\tVERB\tVB\t_\t0\troot\t_\tSpaceAfter=No
5\t,\t,\tPUNCT\t,\t_\t6\tpunct\t_\t_
6\tsadly\tsadly\tADV\tRB\t_\t4\tadvmod\t_\tSpaceAfter=No
7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tSpacesAfter=\\n

# sent_id = 2
# text = Bye!
1\tBye\tbye\tINTJ\tUH\t_\t0\troot\t_\tSpaceAfter=No
2\t!\t!\tPUNCT\t.\t_\t1\tpunct\t_\t_
""".lstrip()

ENGLISH_TEXT = "She can't swim, sadly.\nBye!"

# `al` is an MWT which is not a substring split of its words
SPANISH = """
# sent_id = 1
# text = Vamos al mar.
1\tVamos\tir\tVERB\t_\t_\t0\troot\t_\t_
2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_
2\ta\ta\tADP\t_\t_\t4\tcase\t_\t_
3\tel\tel\tDET\t_\t_\t4\tdet\t_\t_
4\tmar\tmar\tNOUN\t_\t_\t1\tobl\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_
""".lstrip()

# the text comment says there are spaces around the quotes,
# the MISC says there are not
INCONSISTENT = """
# sent_id = 1
# text = Il a dit « bonjour ».
1\tIl\til\tPRON\t_\t_\t3\tnsubj\t_\t_
2\ta\tavoir\tAUX\t_\t_\t3\taux\t_\t_
3\tdit\tdire\tVERB\t_\t_\t0\troot\t_\t_
4\t«\t«\tPUNCT\t_\t_\t6\tpunct\t_\tSpaceAfter=No
5\tbonjour\tbonjour\tNOUN\t_\t_\t3\tobj\t_\tSpaceAfter=No
6\t»\t»\tPUNCT\t_\t_\t5\tpunct\t_\tSpaceAfter=No
7\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_
""".lstrip()

# the SpaceAfter=No which matters here is on the range line of the MWT,
# which is where UD puts it, rather than on the last word
MWT_SPACE_AFTER = """
# sent_id = 1
# text = Nadie puede representarme.
1\tNadie\tnadie\tPRON\t_\tNumber=Sing|PronType=Neg\t3\tnsubj\t_\t_
2\tpuede\tpoder\tAUX\t_\tVerbForm=Fin\t3\taux\t_\t_
3-4\trepresentarme\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
3\trepresentar\trepresentar\tVERB\t_\tVerbForm=Inf\t0\troot\t_\t_
4\tme\tyo\tPRON\t_\tNumber=Sing|Person=1\t3\tobj\t_\t_
5\t.\t.\tPUNCT\t_\tPunctType=Peri\t3\tpunct\t_\t_
""".lstrip()

# the words of the MWT have their own MISC, but none of it is about
# whitespace, so the SpaceAfter=No on the range line is still the one that counts
MWT_UNRELATED_WORD_MISC = """
# sent_id = 1
# text = Nadie puede representarme.
1\tNadie\tnadie\tPRON\t_\tNumber=Sing|PronType=Neg\t3\tnsubj\t_\t_
2\tpuede\tpoder\tAUX\t_\tVerbForm=Fin\t3\taux\t_\t_
3-4\trepresentarme\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
3\trepresentar\trepresentar\tVERB\t_\tVerbForm=Inf\t0\troot\t_\tNER=O
4\tme\tyo\tPRON\t_\tNumber=Sing|Person=1\t3\tobj\t_\tNER=B-PER|Translit=me
5\t.\t.\tPUNCT\t_\tPunctType=Peri\t3\tpunct\t_\t_
""".lstrip()

def check_offsets(doc):
    """Every token and every word with an offset should index back to itself"""
    for sentence in doc.sentences:
        for token in sentence.tokens:
            assert doc.text[token.start_char:token.end_char] == token.text
            for word in token.words:
                if word.start_char is not None:
                    assert doc.text[word.start_char:word.end_char] == word.text

def test_no_offsets_by_default():
    """Reading a document without the flag should not invent offsets"""
    doc = CoNLL.conll2doc(input_str=ENGLISH)
    assert doc.text is None
    for sentence in doc.sentences:
        for token in sentence.tokens:
            assert token.start_char is None
            assert token.end_char is None

def test_text_and_offsets():
    doc = CoNLL.conll2doc(input_str=ENGLISH, reconstruct_text=True)
    assert doc.text == ENGLISH_TEXT
    check_offsets(doc)
    tokens = doc.sentences[0].tokens
    assert (tokens[0].start_char, tokens[0].end_char) == (0, 3)
    # the MWT is split across its words
    assert [(word.start_char, word.end_char) for word in tokens[1].words] == [(4, 6), (6, 9)]

def test_sentence_text():
    doc = CoNLL.conll2doc(input_str=ENGLISH, reconstruct_text=True)
    assert [sentence.text for sentence in doc.sentences] == ["She can't swim, sadly.", "Bye!"]

def test_no_text_comments():
    """The offsets should come out the same when rebuilt from SpaceAfter alone"""
    without = "\n".join(x for x in ENGLISH.split("\n") if not x.startswith("# text"))
    doc = CoNLL.conll2doc(input_str=without, reconstruct_text=True)
    assert doc.text == ENGLISH_TEXT
    check_offsets(doc)

def test_unsplittable_mwt():
    """`al` is not made of its words, so the words get no offsets, but the token does"""
    doc = CoNLL.conll2doc(input_str=SPANISH, reconstruct_text=True)
    assert doc.text == "Vamos al mar."
    check_offsets(doc)
    token = doc.sentences[0].tokens[1]
    assert (token.start_char, token.end_char) == (6, 8)
    assert all(word.start_char is None for word in token.words)

def test_inconsistent_annotation():
    """When the text comment and the MISC disagree, the text wins, with a warning"""
    with pytest.warns(UserWarning):
        doc = CoNLL.conll2doc(input_str=INCONSISTENT, reconstruct_text=True)
    assert doc.text == "Il a dit « bonjour »."
    check_offsets(doc)

def test_mwt_space_after():
    """UD marks SpaceAfter on the range line of an MWT, not on its last word"""
    doc = CoNLL.conll2doc(input_str=MWT_SPACE_AFTER, reconstruct_text=True)
    assert doc.text == "Nadie puede representarme."
    check_offsets(doc)

def test_mwt_space_after_no_text_comment():
    """The same, with only the SpaceAfter annotations to go on"""
    without = "\n".join(x for x in MWT_SPACE_AFTER.split("\n") if not x.startswith("# text"))
    doc = CoNLL.conll2doc(input_str=without, reconstruct_text=True)
    assert doc.text == "Nadie puede representarme."
    check_offsets(doc)

def test_mwt_space_after_on_word():
    """Some treebanks put the annotation on the last word instead"""
    moved = MWT_SPACE_AFTER.replace("representarme\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No", "representarme\t_\t_\t_\t_\t_\t_\t_\t_")
    moved = moved.replace("Number=Sing|Person=1\t3\tobj\t_\t_", "Number=Sing|Person=1\t3\tobj\t_\tSpaceAfter=No")
    without = "\n".join(x for x in moved.split("\n") if not x.startswith("# text"))
    doc = CoNLL.conll2doc(input_str=without, reconstruct_text=True)
    assert doc.text == "Nadie puede representarme."
    check_offsets(doc)

def test_unrelated_word_misc():
    """MISC on a word which says nothing about whitespace should not mask the token's annotation

    Without the `# text` comment there is nothing else to fall back on, so
    reading the word's MISC as "no annotation means a space" would put a
    space in front of the period
    """
    without = "\n".join(x for x in MWT_UNRELATED_WORD_MISC.split("\n") if not x.startswith("# text"))
    doc = CoNLL.conll2doc(input_str=without, reconstruct_text=True)
    assert doc.text == "Nadie puede representarme."
    check_offsets(doc)

def test_unrelated_word_misc_no_warning():
    """The same document with its `# text` comment should agree, and so not warn"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        doc = CoNLL.conll2doc(input_str=MWT_UNRELATED_WORD_MISC, reconstruct_text=True)
    assert doc.text == "Nadie puede representarme."
    check_offsets(doc)

def test_misc_unchanged():
    """Rebuilding the text should not rewrite the MISC column"""
    plain = CoNLL.conll2doc(input_str=ENGLISH)
    offsets = CoNLL.conll2doc(input_str=ENGLISH, reconstruct_text=True)
    assert "{:C-o}".format(plain) == "{:C-o}".format(offsets)
