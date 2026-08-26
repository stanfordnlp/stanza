"""
Test the CoNLL-U comments attached to a Sentence

The comments are the storage for sent_id, doc_id, speaker, sentiment, and the
constituency tree, so these tests cover both the comment API itself and the
Sentence attributes which read and write it.
"""

import pytest

from stanza.models.common.doc import Comment, CommentList, Document, split_comment
from stanza.models.common.doc import CONSTITUENCY, DOC_ID, SENT_ID, SENTIMENT, SPEAKER, TEXT
from stanza.models.constituency import tree_reader
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

CONLLU = """
# sent_id = s1
# text = Hi there.
# speaker = Bob
# newpar
# constituency = (ROOT (S (INTJ (UH Hi)) (ADVP (RB there)) (. .)))
# sentiment = 2
1\tHi\thi\tINTJ\tUH\t_\t0\troot\t_\t_
2\tthere\tthere\tADV\tRB\t_\t1\tadvmod\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_

# sent_id = s2
# text = Bye.
# speaker = Alice
1\tBye\tbye\tINTJ\tUH\t_\t0\troot\t_\tSpaceAfter=No
2\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_
""".lstrip()

@pytest.fixture
def doc():
    return CoNLL.conll2doc(input_str=CONLLU)

@pytest.fixture
def sentence(doc):
    return doc.sentences[0]


def test_split_comment():
    """The spellings of `# key = value` which show up in treebanks"""
    assert split_comment("# sent_id = 3") == (SENT_ID, "3")
    assert split_comment("#sent_id=3") == (SENT_ID, "3")
    assert split_comment("# text = Hi there.") == (TEXT, "Hi there.")
    assert split_comment("#text=Hi there.") == (TEXT, "Hi there.")
    # a key with a space in it, which UD uses for newdoc & newpar
    assert split_comment("# newdoc id = foo") == ("newdoc id", "foo")
    # `# text_en` and `# text_ortho`, as used by Naija-NSC, are their own keys
    assert split_comment("# text_en = Hi") == ("text_en", "Hi")
    # a comment with no key at all
    assert split_comment("# newpar") == (None, None)


def test_values_from_comments(sentence):
    """Reading a file fills in the Sentence attributes the comments back"""
    assert sentence.sent_id == "s1"
    assert sentence.speaker == "Bob"
    assert sentence.text == "Hi there."
    assert sentence.sentiment == 2
    assert isinstance(sentence.sentiment, int)
    assert str(sentence.constituency) == "(ROOT (S (INTJ (UH Hi)) (ADVP (RB there)) (. .)))"
    assert sentence.doc_id is None


def test_round_trip(doc):
    """Comments come back out of the document unchanged"""
    assert "{:C}\n".format(doc) == CONLLU


def test_unusual_spelling_preserved():
    """A comment keeps the text it had in the file, rather than being reformatted"""
    doc = CoNLL.conll2doc(input_str="#sent_id=s1\n#text=Hi\n1\tHi\thi\tINTJ\tUH\t_\t0\troot\t_\t_\n")
    sentence = doc.sentences[0]
    assert sentence.sent_id == "s1"
    assert sentence.text == "Hi"
    assert list(sentence.comments) == ["#sent_id=s1", "#text=Hi"]


def test_remove_by_key(sentence):
    assert sentence.remove_comment("speaker") == 1
    assert sentence.speaker is None
    assert "# speaker = Bob" not in sentence.comments
    # removing a comment which isn't there is not an error
    assert sentence.remove_comment("speaker") == 0


def test_remove_by_line(sentence):
    """A whole comment line removes that key whatever value it holds"""
    assert sentence.remove_comment("# speaker = some other value") == 1
    assert sentence.speaker is None


def test_remove_keyless(sentence):
    """A comment with no key is matched on its text"""
    assert sentence.remove_comment("# newpar") == 1
    assert "# newpar" not in sentence.comments
    assert sentence.remove_comment("newpar") == 0


def test_remove_clears_attribute(sentence):
    """The comment is the storage, so removing it clears the attribute"""
    assert sentence.constituency is not None
    assert sentence.remove_comment(CONSTITUENCY) == 1
    assert sentence.constituency is None

    assert sentence.sentiment is not None
    assert sentence.remove_comment(SENTIMENT) == 1
    assert sentence.sentiment is None


def test_remove_across_document(doc):
    assert doc.remove_comments(SPEAKER) == 2
    assert [x.speaker for x in doc.sentences] == [None, None]
    assert doc.remove_comments(SPEAKER) == 0


def test_set_across_document(doc):
    doc.set_comments(DOC_ID, "corpus-1")
    assert [x.doc_id for x in doc.sentences] == ["corpus-1", "corpus-1"]
    assert "# doc_id = corpus-1" in doc.sentences[0].comments
    doc.set_comments(DOC_ID, None)
    assert [x.doc_id for x in doc.sentences] == [None, None]


def test_no_duplicate_annotations(sentence):
    """A key which backs an attribute can only appear once"""
    sentence.sent_id = "a"
    sentence.sent_id = "b"
    sentence.add_comment("# sent_id = c")
    assert [x for x in sentence.comments if x.startswith("# sent_id")] == ["# sent_id = c"]
    assert sentence.sent_id == "c"


def test_repeated_key_kept(sentence):
    """A key which does not back an attribute is not collapsed on the way in

    Losing a line of a treebank just by reading it would be worse than
    carrying a repeated key around.
    """
    sentence.add_comment("# text_en = one")
    sentence.add_comment("# text_en = two")
    assert [x for x in sentence.comments if x.startswith("# text_en")] == ["# text_en = one", "# text_en = two"]
    # setting it explicitly does collapse them
    sentence.set_comment("text_en", "three")
    assert [x for x in sentence.comments if x.startswith("# text_en")] == ["# text_en = three"]


def test_none_removes(sentence):
    """Setting an attribute to None drops the comment rather than writing None"""
    sentence.constituency = None
    sentence.sentiment = None
    sentence.speaker = None
    assert not sentence.has_comment(CONSTITUENCY)
    assert not sentence.has_comment(SENTIMENT)
    assert not sentence.has_comment(SPEAKER)
    assert "None" not in "\n".join(sentence.comments)


def test_blank_speaker(sentence):
    """A blank speaker removes the comment, same as None"""
    sentence.speaker = ""
    assert not sentence.has_comment(SPEAKER)


def test_get_set_has(sentence):
    assert sentence.has_comment(SPEAKER)
    assert not sentence.has_comment("translit")
    assert sentence.get_comment(SPEAKER) == "Bob"
    assert sentence.get_comment("translit") is None
    assert sentence.get_comment("translit", "n/a") == "n/a"

    sentence.set_comment("translit", "hi there.")
    assert sentence.comments[-1] == "# translit = hi there."
    assert sentence.get_comment("translit") == "hi there."


def test_list_interface(doc):
    """Comments still read like the list of strings they used to be"""
    comments = doc.sentences[1].comments
    assert len(comments) == 3
    assert comments[0] == "# sent_id = s2"
    assert comments[:2] == ["# sent_id = s2", "# text = Bye."]
    assert list(comments) == ["# sent_id = s2", "# text = Bye.", "# speaker = Alice"]
    assert comments == ["# sent_id = s2", "# text = Bye.", "# speaker = Alice"]
    assert "# speaker = Alice" in comments
    assert "\n".join(comments) == "# sent_id = s2\n# text = Bye.\n# speaker = Alice"


def test_list_mutation(doc):
    """Assigning or deleting by index goes through the same parsing"""
    comments = doc.sentences[1].comments
    comments[2] = "# speaker = Carol"
    assert doc.sentences[1].speaker == "Carol"
    del comments[2]
    assert doc.sentences[1].speaker is None


def test_assign_comments(sentence):
    """Replacing all of the comments replaces everything they carried"""
    sentence.comments = ["# sent_id = brand-new", "# speaker = Dave"]
    assert len(sentence.comments) == 2
    assert sentence.sent_id == "brand-new"
    assert sentence.speaker == "Dave"
    assert sentence.constituency is None
    assert sentence.sentiment is None


def test_add_comment_hash(sentence):
    """A comment without a leading # gets one"""
    sentence.add_comment("bare comment")
    assert sentence.comments[-1] == "# bare comment"


def test_comment_list_standalone():
    comments = CommentList(["# sent_id = 1", "bare"])
    assert list(comments) == ["# sent_id = 1", "# bare"]
    assert comments.keys() == [SENT_ID]
    assert comments.get(SENT_ID) == "1"
    assert comments.get_line(SENT_ID) == "# sent_id = 1"
    assert comments.has(SENT_ID)

    copied = comments.copy()
    copied.set(SPEAKER, "Bob")
    assert len(comments) == 2
    assert len(copied) == 3

    comments.clear()
    assert not comments
    assert len(comments) == 0


def test_constituency_newlines():
    """A tree with a newline in it survives a round trip through a comment"""
    tree = tree_reader.read_trees("(ROOT (S (INTJ (UH Hi))\n(ADVP (RB there)) (. .)))")[0]
    comment = Comment.from_value(CONSTITUENCY, tree)
    assert "\n" not in str(comment)
    assert str(Comment.from_line(str(comment)).value) == str(tree)


def test_sentence_without_document():
    """A Sentence built on its own has no annotations rather than no attributes"""
    doc = Document([[{"id": (1,), "text": "Hi"}]])
    sentence = doc.sentences[0]
    assert sentence.doc_id is None
    assert sentence.speaker is None
    assert sentence.sentiment is None
    assert sentence.constituency is None
    # a document does number its sentences
    assert sentence.sent_id == "0"


def test_serialized_comments(doc):
    """Comments survive to_serialized / from_serialized"""
    reloaded = Document.from_serialized(doc.to_serialized())
    assert reloaded.sentence_comments() == doc.sentence_comments()
    assert "{:C}".format(reloaded) == "{:C}".format(doc)
    assert reloaded.sentences[0].sentiment == 2
