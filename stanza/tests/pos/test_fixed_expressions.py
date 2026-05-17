"""
Unit tests for the fixed-expression input feature introduced for issue #1575.

Covers:
  * The standalone :class:`FixedExpressionVocab` lookup and serialisation.
  * The :meth:`Dataset.build_fixed_expressions` extractor.
  * Wiring through :meth:`Dataset.init_vocab` /
    :meth:`Dataset.preprocess` / collation when ``--use_fixed_expressions``
    is enabled, and the no-op default behaviour when it is not.
"""

from stanza.models import tagger
from stanza.models.pos.data import Dataset
from stanza.models.pos.vocab import (
    FixedExpressionVocab,
    MultiVocab,
    FIXED_NO_ID,
    FIXED_YES_ID,
)
from stanza.utils.conll import CoNLL


# A pair of UD sentences with ExtPos-bearing fixed expressions:
#   - Spanish "de hecho" (in fact)         -> ADV
#   - English "in case of" (in case of)    -> ADP
SPANISH_FIXED = """
# sent_id = es-fixed-1
# text = De hecho, llegó tarde.
1	De	de	ADP	IN	ExtPos=ADV	4	advmod	_	_
2	hecho	hecho	NOUN	NN	_	1	fixed	_	_
3	,	,	PUNCT	,	_	1	punct	_	_
4	llegó	llegar	VERB	VBD	Mood=Ind|Tense=Past	0	root	_	_
5	tarde	tarde	ADV	RB	_	4	advmod	_	SpaceAfter=No
6	.	.	PUNCT	.	_	4	punct	_	_

""".lstrip()

ENGLISH_FIXED = """
# sent_id = en-fixed-1
# text = In case of fire, run.
1	In	in	ADP	IN	ExtPos=ADP	6	case	_	_
2	case	case	NOUN	NN	_	1	fixed	_	_
3	of	of	ADP	IN	_	1	fixed	_	_
4	fire	fire	NOUN	NN	Number=Sing	6	obl	_	SpaceAfter=No
5	,	,	PUNCT	,	_	6	punct	_	_
6	run	run	VERB	VB	VerbForm=Inf	0	root	_	SpaceAfter=No
7	.	.	PUNCT	.	_	6	punct	_	_

""".lstrip()

# A control sentence with no fixed relations; should contribute nothing.
PLAIN_SENT = """
# sent_id = plain-1
# text = She left quickly.
1	She	she	PRON	PRP	_	2	nsubj	_	_
2	left	leave	VERB	VBD	Mood=Ind|Tense=Past	0	root	_	_
3	quickly	quickly	ADV	RB	_	2	advmod	_	SpaceAfter=No
4	.	.	PUNCT	.	_	2	punct	_	_

""".lstrip()


# ---------------------------------------------------------------------------
# FixedExpressionVocab unit tests
# ---------------------------------------------------------------------------

class TestFixedExpressionVocab:
    def test_empty_lookup(self):
        v = FixedExpressionVocab()
        assert len(v) == 0
        # Empty vocab tags every token as "no".
        assert v.map(["a", "b", "c"]) == [FIXED_NO_ID] * 3
        assert v.map([]) == []

    def test_basic_lookup_lowercases(self):
        v = FixedExpressionVocab([("de", "hecho"), ("in", "case", "of")])
        # Match is case-insensitive by default.
        flags = v.map(["De", "hecho", "y", "In", "case", "of", "fire", "de"])
        assert flags == [
            FIXED_YES_ID,  # De -> "de hecho"
            FIXED_NO_ID,   # hecho (middle of expression, not a start)
            FIXED_NO_ID,   # y
            FIXED_YES_ID,  # In -> "in case of"
            FIXED_NO_ID,   # case
            FIXED_NO_ID,   # of
            FIXED_NO_ID,   # fire
            FIXED_NO_ID,   # de (no "hecho" follows)
        ]

    def test_longest_match_window_respected(self):
        # max_len should clamp the lookup window even with a long sentence.
        v = FixedExpressionVocab([("a", "b")])
        assert v.max_len == 2
        # Trailing single-token windows must not flag anything.
        assert v.map(["a"]) == [FIXED_NO_ID]
        assert v.map(["a", "b"]) == [FIXED_YES_ID, FIXED_NO_ID]
        assert v.map(["a", "c", "a", "b"]) == [
            FIXED_NO_ID,
            FIXED_NO_ID,
            FIXED_YES_ID,
            FIXED_NO_ID,
        ]

    def test_case_sensitive_mode(self):
        v = FixedExpressionVocab([("De", "Hecho")], lowercase=False)
        assert v.map(["De", "Hecho", "de", "hecho"]) == [
            FIXED_YES_ID,
            FIXED_NO_ID,
            FIXED_NO_ID,
            FIXED_NO_ID,
        ]

    def test_add_skips_singletons(self):
        v = FixedExpressionVocab()
        v.add(("just_one",))  # singletons are not multi-word expressions
        v.add(("two", "words"))
        assert len(v) == 1
        assert ("two", "words") in v

    def test_state_dict_roundtrip(self):
        v = FixedExpressionVocab([("de", "hecho"), ("in", "case", "of")])
        sd = v.state_dict()
        restored = FixedExpressionVocab.load_state_dict(sd)
        assert restored.expressions == v.expressions
        assert restored.max_len == v.max_len
        # The serialised form is deterministic (sorted) for reproducible IO.
        assert sd["expressions"] == sorted(v.expressions)


# ---------------------------------------------------------------------------
# Dataset.build_fixed_expressions tests
# ---------------------------------------------------------------------------

class TestBuildFixedExpressions:
    def _docs(self, *blobs):
        return [CoNLL.conll2doc(input_str=b) for b in blobs]

    def test_extracts_de_hecho_and_in_case_of(self):
        docs = self._docs(SPANISH_FIXED, ENGLISH_FIXED, PLAIN_SENT)
        v = Dataset.build_fixed_expressions(docs)
        assert ("de", "hecho") in v
        assert ("in", "case", "of") in v
        # The plain sentence shouldn't contribute any spurious entries.
        assert len(v) == 2
        assert v.max_len == 3

    def test_min_count_filters_singletons(self):
        # "de hecho" appears once; with min_count=2 it should be dropped.
        docs = self._docs(SPANISH_FIXED, ENGLISH_FIXED)
        v = Dataset.build_fixed_expressions(docs, min_count=2)
        assert len(v) == 0

    def test_external_file_merges_in(self, tmp_path):
        extra = tmp_path / "extra_fixed.txt"
        # Whitespace-separated forms, blank/comment lines ignored.
        extra.write_text("\n".join([
            "# extras",
            "à propos",
            "ad hoc",
            "",
            "by the way",
        ]))
        docs = self._docs(SPANISH_FIXED)
        v = Dataset.build_fixed_expressions(docs, extra_file=str(extra))
        assert ("de", "hecho") in v
        assert ("à", "propos") in v
        assert ("ad", "hoc") in v
        assert ("by", "the", "way") in v
        assert v.max_len == 3


# ---------------------------------------------------------------------------
# Integration with Dataset / DataLoader
# ---------------------------------------------------------------------------

class TestDatasetIntegration:
    def _train_args(self, extra=None):
        args = [
            "--shorthand", "en_test",
            "--augment_nopunct", "0.0",
            "--batch_size", "4",
        ]
        if extra:
            args = args + extra
        return tagger.parse_args(args=args)

    def test_default_off_keeps_legacy_behaviour(self):
        args = self._train_args()
        doc = CoNLL.conll2doc(input_str=ENGLISH_FIXED)
        data = Dataset(doc, args, None)
        # Without the flag, no 'fixed' vocab is built.
        assert "fixed" not in data.vocab
        loader = data.to_loader(batch_size=2)
        batch = next(iter(loader))
        assert batch.fixed_flags is None

    def test_flag_enabled_threads_through_to_batch(self):
        args = self._train_args(extra=["--use_fixed_expressions"])
        doc = CoNLL.conll2doc(input_str=ENGLISH_FIXED)
        # Both build the vocab from the same doc and run it through Dataset.
        data = Dataset(doc, args, None)
        assert "fixed" in data.vocab
        fixedvocab = data.vocab["fixed"]
        assert ("in", "case", "of") in fixedvocab

        loader = data.to_loader(batch_size=2)
        batch = next(iter(loader))
        assert batch.fixed_flags is not None
        flags = batch.fixed_flags
        # Single sentence, so batch is [1, seq_len].
        assert flags.shape[0] == 1
        # The first token "In" should be flagged as the start of a known expression.
        assert flags[0, 0].item() == FIXED_YES_ID
        # Internal positions of the expression and the rest are "no".
        for j in (1, 2):
            assert flags[0, j].item() == FIXED_NO_ID


# ---------------------------------------------------------------------------
# MultiVocab serialisation roundtrip including FixedExpressionVocab
# ---------------------------------------------------------------------------

def test_multivocab_serialises_fixed_vocab():
    fv = FixedExpressionVocab([("de", "hecho")])
    mv = MultiVocab()
    mv["fixed"] = fv
    state = mv.state_dict()
    restored = MultiVocab.load_state_dict(state)
    assert "fixed" in restored
    assert ("de", "hecho") in restored["fixed"]
