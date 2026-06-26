"""
Tests for the lemmatizer trainer, focusing on the dictionary component
and the new pos_dict storage format.
"""

import pytest

import glob
import os
import tempfile

import torch

from stanza.models import lemmatizer
from stanza.models.lemma import trainer
from stanza.models.lemma.trainer import (
    _POS_INDEPENDENT,
    _DICTS_VERSION_LEGACY,
    _DICTS_VERSION_POS,
    _pack_pos_dict,
    _unpack_pos_dict,
    _legacy_dicts_to_pos_dict,
)
from stanza.tests import *
from stanza.utils.training.common import choose_lemma_charlm, build_charlm_args

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope="module")
def english_model():
    models_path = os.path.join(TEST_MODELS_DIR, "en", "lemma", "*")
    models = glob.glob(models_path)
    assert len(models) >= 1, "No English lemma models downloaded during setup!  Please make sure to run the setup script."
    for model_file in models:
        if "nocharlm" in model_file:
            return trainer.Trainer(model_file=model_file)
    raise FileNotFoundError("Should have downloaded the nocharlm English lemmatizer during setup.  Please rerun the setup script.")

def test_load_model(english_model):
    """
    Does nothing, just tests that loading works
    """

def test_save_load_model(english_model):
    """
    Load, save, and load again
    """
    with tempfile.TemporaryDirectory() as tempdir:
        save_file = os.path.join(tempdir, "resaved", "lemma.pt")
        english_model.save(save_file)
        reloaded = trainer.Trainer(model_file=save_file)

TRAIN_DATA = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003
# text = DPA: Iraqi authorities announced that they had busted up 3 terrorist cells operating in Baghdad.
1	DPA	DPA	PROPN	NNP	Number=Sing	0	root	0:root	SpaceAfter=No
2	:	:	PUNCT	:	_	1	punct	1:punct	_
3	Iraqi	Iraqi	ADJ	JJ	Degree=Pos	4	amod	4:amod	_
4	authorities	authority	NOUN	NNS	Number=Plur	5	nsubj	5:nsubj	_
5	announced	announce	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	1	parataxis	1:parataxis	_
6	that	that	SCONJ	IN	_	9	mark	9:mark	_
7	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	9	nsubj	9:nsubj	_
8	had	have	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	9	aux	9:aux	_
9	busted	bust	VERB	VBN	Tense=Past|VerbForm=Part	5	ccomp	5:ccomp	_
10	up	up	ADP	RP	_	9	compound:prt	9:compound:prt	_
11	3	3	NUM	CD	NumForm=Digit|NumType=Card	13	nummod	13:nummod	_
12	terrorist	terrorist	ADJ	JJ	Degree=Pos	13	amod	13:amod	_
13	cells	cell	NOUN	NNS	Number=Plur	9	obj	9:obj	_
14	operating	operate	VERB	VBG	VerbForm=Ger	13	acl	13:acl	_
15	in	in	ADP	IN	_	16	case	16:case	_
16	Baghdad	Baghdad	PROPN	NNP	Number=Sing	14	obl	14:obl:in	SpaceAfter=No
17	.	.	PUNCT	.	_	1	punct	1:punct	_

# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0004
# text = Two of them were being run by 2 officials of the Ministry of the Interior!
1	Two	two	NUM	CD	NumForm=Word|NumType=Card	6	nsubj:pass	6:nsubj:pass	_
2	of	of	ADP	IN	_	3	case	3:case	_
3	them	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	1	nmod	1:nmod:of	_
4	were	be	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	6	aux	6:aux	_
5	being	be	AUX	VBG	VerbForm=Ger	6	aux:pass	6:aux:pass	_
6	run	run	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_
7	by	by	ADP	IN	_	9	case	9:case	_
8	2	2	NUM	CD	NumForm=Digit|NumType=Card	9	nummod	9:nummod	_
9	officials	official	NOUN	NNS	Number=Plur	6	obl	6:obl:by	_
10	of	of	ADP	IN	_	12	case	12:case	_
11	the	the	DET	DT	Definite=Def|PronType=Art	12	det	12:det	_
12	Ministry	Ministry	PROPN	NNP	Number=Sing	9	nmod	9:nmod:of	_
13	of	of	ADP	IN	_	15	case	15:case	_
14	the	the	DET	DT	Definite=Def|PronType=Art	15	det	15:det	_
15	Interior	Interior	PROPN	NNP	Number=Sing	12	nmod	12:nmod:of	SpaceAfter=No
16	!	!	PUNCT	.	_	6	punct	6:punct	_

""".lstrip()

DEV_DATA = """
1	From	from	ADP	IN	_	3	case	3:case	_
2	the	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
3	AP	AP	PROPN	NNP	Number=Sing	4	obl	4:obl:from	_
4	comes	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
5	this	this	DET	DT	Number=Sing|PronType=Dem	6	det	6:det	_
6	story	story	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
7	:	:	PUNCT	:	_	4	punct	4:punct	_

""".lstrip()

class TestLemmatizer:
    @pytest.fixture(scope="class")
    def charlm_args(self):
        charlm = choose_lemma_charlm("en", "test", "default")
        charlm_args = build_charlm_args("en", charlm, model_dir=TEST_MODELS_DIR)
        return charlm_args

    def run_training(self, tmp_path, train_text, dev_text, extra_args=None):
        """
        Run the training for a few iterations, load & return the model
        """
        pred_file = str(tmp_path / "pred.conllu")

        save_name = "test_tagger.pt"
        save_file = str(tmp_path / save_name)

        train_file = str(tmp_path / "train.conllu")
        with open(train_file, "w", encoding="utf-8") as fout:
            fout.write(train_text)

        dev_file = str(tmp_path / "dev.conllu")
        with open(dev_file, "w", encoding="utf-8") as fout:
            fout.write(dev_text)

        args = ["--train_file", train_file,
                "--eval_file", dev_file,
                "--output_file", pred_file,
                "--num_epoch", "2",
                "--log_step", "10",
                "--save_dir", str(tmp_path),
                "--save_name", save_name,
                "--shorthand", "en_test"]
        if extra_args is not None:
            args = args + extra_args
        lemmatizer.main(args)

        assert os.path.exists(save_file)
        saved_model = trainer.Trainer(model_file=save_file)
        return saved_model

    def test_basic_train(self, tmp_path):
        """
        Simple test of a few 'epochs' of lemmatizer training
        """
        self.run_training(tmp_path, TRAIN_DATA, DEV_DATA)

    def test_dict_only_train(self, tmp_path):
        """
        Train a dictionary-only lemmatizer via --dict_only and verify:
          - no seq2seq model was built
          - the dictionary was populated from the training data
          - known words are predicted correctly
          - unknown words fall back to the word itself
          - all known words are flagged as skippable by skip_seq2seq
        """
        saved_model = self.run_training(tmp_path, TRAIN_DATA, DEV_DATA,
                                        extra_args=["--dict_only"])

        # no seq2seq model should have been constructed
        assert saved_model.model is None

        # spot-check words that appear in TRAIN_DATA with known lemmas
        known_pairs = [("authorities", "NOUN"),
                       ("announced",   "VERB"),
                       ("operating",   "VERB")]
        expected    = ["authority", "announce", "operate"]
        preds = saved_model.predict_dict(known_pairs)
        assert preds == expected, \
            f"Dict lemmatizer produced wrong lemmas: {list(zip(known_pairs, preds))}"

        # unknown words should be returned as-is
        assert saved_model.predict_dict([("xyzzy", "NOUN")]) == ["xyzzy"]

        # all known pairs should be flagged as skippable
        skip = saved_model.skip_seq2seq(known_pairs)
        assert all(skip), \
            f"Expected all known words to be skippable, but got: {list(zip(known_pairs, skip))}"

    def test_charlm_train(self, tmp_path, charlm_args):
        """
        Simple test of a few 'epochs' of lemmatizer training
        """
        saved_model = self.run_training(tmp_path, TRAIN_DATA, DEV_DATA, extra_args=charlm_args)

        # check that the charlm wasn't saved in here
        args = saved_model.args
        save_name = os.path.join(args['save_dir'], args['save_name'])
        checkpoint = torch.load(save_name, lambda storage, loc: storage, weights_only=True)
        assert not any(x.startswith("contextual_embedding") for x in checkpoint['model'].keys())


def _make_trainer():
    """Return a minimal dict-only Trainer with empty pos_dict, no seq2seq."""
    t = trainer.Trainer.__new__(trainer.Trainer)
    t.args = {'dict_only': True, 'caseless': False}
    t.caseless = False
    t.pos_dict = {}
    t.contextual_lemmatizers = []
    t.model = None
    return t


class TestPosDictFormat:
    """
    Tests for the new {pos: {word: lemma}} storage format:
    pack/unpack, legacy conversion, and the _POS_INDEPENDENT fallback.
    """

    def test_pack_unpack_roundtrip(self):
        """_pack_pos_dict / _unpack_pos_dict must be exact inverses."""
        pos_dict = {
            _POS_INDEPENDENT: {"run": "run", "left": "leave"},
            "ADJ":            {"left": "left"},
        }
        assert _unpack_pos_dict(_pack_pos_dict(pos_dict)) == pos_dict

    def test_pack_produces_bytes(self):
        """Packed format should be bytes (gzip-compressed pickle)."""
        packed = _pack_pos_dict({_POS_INDEPENDENT: {"run": "run"}})
        assert isinstance(packed, bytes)
        # gzip magic number
        assert packed[:2] == b'\x1f\x8b'

    def test_legacy_conversion_pos_independent(self):
        """word_dict entries become _POS_INDEPENDENT entries."""
        word_dict = {"run": "run", "left": "leave"}
        composite_dict = {}
        pos_dict = _legacy_dicts_to_pos_dict(word_dict, composite_dict)
        assert pos_dict[_POS_INDEPENDENT] == {"run": "run", "left": "leave"}

    def test_legacy_conversion_drops_redundant_composite(self):
        """
        Composite entries that agree with word_dict are dropped —
        they're recoverable via the _POS_INDEPENDENT fallback.
        """
        word_dict      = {"run": "run"}
        composite_dict = {("run", "VERB"): "run"}   # redundant
        pos_dict = _legacy_dicts_to_pos_dict(word_dict, composite_dict)
        assert "VERB" not in pos_dict or "run" not in pos_dict.get("VERB", {})

    def test_legacy_conversion_keeps_differing_composite(self):
        """
        Composite entries that differ from word_dict must be preserved,
        since they carry real information (e.g. "left" ADJ != "left" VERB).
        """
        word_dict      = {"left": "leave"}           # most-frequent mapping
        composite_dict = {("left", "ADJ"): "left"}   # genuinely different
        pos_dict = _legacy_dicts_to_pos_dict(word_dict, composite_dict)
        assert pos_dict["ADJ"]["left"] == "left"
        assert pos_dict[_POS_INDEPENDENT]["left"] == "leave"

    def test_lookup_pos_specific_beats_fallback(self):
        """Composite (pos-specific) entry takes priority over _POS_INDEPENDENT."""
        t = _make_trainer()
        t.pos_dict = {
            _POS_INDEPENDENT: {"left": "leave"},
            "ADJ":            {"left": "left"},
        }
        assert t.predict_dict([("left", "ADJ")]) == ["left"]

    def test_lookup_pos_independent_fallback(self):
        """When no pos-specific entry exists, _POS_INDEPENDENT is used."""
        t = _make_trainer()
        t.pos_dict = {_POS_INDEPENDENT: {"running": "run"}}
        assert t.predict_dict([("running", "VERB")]) == ["run"]

    def test_lookup_unknown_returns_word(self):
        """Words absent from all dicts should be returned as-is."""
        t = _make_trainer()
        t.pos_dict = {}
        assert t.predict_dict([("xyzzy", "NOUN")]) == ["xyzzy"]

    def test_lookup_caseless(self):
        """With caseless=True, lookup should lowercase the word before lookup."""
        t = _make_trainer()
        t.caseless = True
        t.pos_dict = {_POS_INDEPENDENT: {"baghdad": "baghdad"}}
        # "Baghdad" should match "baghdad" after lowercasing
        assert t.predict_dict([("Baghdad", "PROPN")]) == ["baghdad"]

    def test_lookup_caseless_does_not_affect_lemma(self):
        """Caseless only lowercases the lookup key, not the returned lemma."""
        t = _make_trainer()
        t.caseless = True
        t.pos_dict = {_POS_INDEPENDENT: {"baghdad": "Baghdad"}}
        assert t.predict_dict([("BAGHDAD", "PROPN")]) == ["Baghdad"]


class TestDictLemmatizer:
    """
    Tests for train_dict, predict_dict, skip_seq2seq, and ensemble
    using the new pos_dict interface.
    """

    def test_train_dict_basic(self):
        """Known triples should be retrievable after training."""
        t = _make_trainer()
        triples = [("authorities", "NOUN", "authority"),
                   ("announced",   "VERB", "announce"),
                   ("running",     "VERB", "run")]
        t.train_dict(triples)

        assert t.predict_dict([("authorities", "NOUN")]) == ["authority"]
        assert t.predict_dict([("announced",   "VERB")]) == ["announce"]
        assert t.predict_dict([("running",     "VERB")]) == ["run"]

    def test_train_dict_pos_independent_set(self):
        """train_dict should populate the _POS_INDEPENDENT fallback."""
        t = _make_trainer()
        t.train_dict([("running", "VERB", "run")])
        assert t.pos_dict[_POS_INDEPENDENT]["running"] == "run"

    def test_train_dict_redundant_not_stored(self):
        """
        When the composite lemma matches the pos-independent entry,
        no separate pos-specific entry should be stored.
        """
        t = _make_trainer()
        # "run" VERB -> "run": same as the pos-independent entry
        t.train_dict([("run", "VERB", "run")])
        assert t.pos_dict[_POS_INDEPENDENT]["run"] == "run"
        # VERB bucket should not have a redundant entry
        assert "run" not in t.pos_dict.get("VERB", {})

    def test_train_dict_differing_pos_stored(self):
        """
        When a word has different lemmas for different POS, the
        non-default mapping must be stored in a pos-specific bucket.
        """
        t = _make_trainer()
        # two occurrences of VERB (most frequent -> pos_independent)
        # one occurrence of ADJ with a different lemma
        triples = [("left", "VERB", "leave"),
                   ("left", "VERB", "leave"),
                   ("left", "ADJ",  "left")]
        t.train_dict(triples)
        # most frequent overall mapping wins for pos_independent
        assert t.pos_dict[_POS_INDEPENDENT]["left"] == "leave"
        # ADJ entry stored because it differs
        assert t.pos_dict["ADJ"]["left"] == "left"

    def test_train_dict_frequency_priority(self):
        """Most frequent mapping wins when a word appears with multiple lemmas."""
        t = _make_trainer()
        triples = [("ran", "VERB", "run"),
                   ("ran", "VERB", "run"),
                   ("ran", "VERB", "run"),
                   ("ran", "VERB", "ran")]   # less frequent
        t.train_dict(triples)
        assert t.predict_dict([("ran", "VERB")]) == ["run"]

    def test_train_dict_no_word_dict_update(self):
        """With update_word_dict=False, _POS_INDEPENDENT is not populated."""
        t = _make_trainer()
        t.train_dict([("authorities", "NOUN", "authority")],
                     update_word_dict=False)
        assert "authorities" not in t.pos_dict.get(_POS_INDEPENDENT, {})
        # but the pos-specific bucket should still be set
        assert t.pos_dict.get("NOUN", {}).get("authorities") == "authority"

    def test_predict_dict_composite_priority(self):
        """pos-specific entry takes priority over _POS_INDEPENDENT."""
        t = _make_trainer()
        t.pos_dict = {
            _POS_INDEPENDENT: {"left": "leave"},
            "ADJ":            {"left": "left"},
        }
        assert t.predict_dict([("left", "VERB")]) == ["leave"]  # fallback
        assert t.predict_dict([("left", "ADJ")])  == ["left"]   # pos-specific

    def test_predict_dict_word_dict_fallback(self):
        """_POS_INDEPENDENT is used when no pos-specific entry exists."""
        t = _make_trainer()
        t.pos_dict = {_POS_INDEPENDENT: {"running": "run"}}
        assert t.predict_dict([("running", "VERB")]) == ["run"]

    def test_predict_dict_unknown_fallback(self):
        """Unknown words return as-is."""
        t = _make_trainer()
        assert t.predict_dict([("xyzzy", "NOUN")]) == ["xyzzy"]

    def test_skip_seq2seq(self):
        """skip_seq2seq returns True for known words, False for unknowns."""
        t = _make_trainer()
        t.pos_dict = {
            _POS_INDEPENDENT: {"running": "run"},
            "VERB":           {"announced": "announce"},
        }
        pairs = [("announced", "VERB"),   # pos-specific
                 ("running",   "NOUN"),   # pos-independent fallback
                 ("xyzzy",     "NOUN")]   # unknown
        assert t.skip_seq2seq(pairs) == [True, True, False]

    def test_ensemble_prefers_dict(self):
        """ensemble uses dict for known words, seq2seq pred for unknowns."""
        t = _make_trainer()
        t.pos_dict = {
            _POS_INDEPENDENT: {"running": "run"},
            "VERB":           {"announced": "announce"},
        }
        pairs      = [("announced", "VERB"), ("running", "NOUN"), ("xyzzy", "NOUN")]
        seq2seq    = ["wrong",               "wrong",             "the_seq2seq_answer"]
        result     = t.ensemble(pairs, seq2seq)
        assert result == ["announce", "run", "the_seq2seq_answer"]

    def test_ensemble_none_fallback(self):
        """
        ensemble never returns None.

        A corrupt None entry is treated as a miss by _lookup, so ensemble
        falls through to the seq2seq prediction.  If that is also None
        (which shouldn't happen in practice), it falls back to the word.
        """
        t = _make_trainer()
        # corrupt None entry -> _lookup returns None -> seq2seq pred is used
        t.pos_dict = {"NOUN": {"broken": None}}
        result = t.ensemble([("broken", "NOUN")], ["seq2seq_pred"])
        assert result == ["seq2seq_pred"]

        # if seq2seq pred is also None, fall back to the word itself
        result = t.ensemble([("broken", "NOUN")], [None])
        assert result == ["broken"]

    def test_dict_save_load_roundtrip(self, tmp_path):
        """
        Save a dict-only Trainer with the new format and reload it;
        predictions must be identical before and after.
        """
        from stanza.models.lemma.vocab import MultiVocab, Vocab

        t = _make_trainer()
        triples = [("authorities", "NOUN", "authority"),
                   ("left",        "VERB", "leave"),
                   ("left",        "VERB", "leave"),
                   ("left",        "ADJ",  "left"),
                   ("running",     "VERB", "run")]
        t.train_dict(triples)

        char_vocab = Vocab("abcdefghijklmnopqrstuvwxyz", "en")
        pos_vocab  = Vocab(["NOUN", "VERB", "ADJ"], "en")
        t.vocab    = MultiVocab({'char': char_vocab, 'pos': pos_vocab})

        save_path = str(tmp_path / "dict_lemmatizer.pt")
        t.save(save_path)

        # verify the checkpoint carries the new version tag
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
        assert checkpoint['dicts_version'] == _DICTS_VERSION_POS
        assert isinstance(checkpoint['dicts'], bytes)

        loaded = trainer.Trainer(model_file=save_path, device='cpu')

        pairs = [("authorities", "NOUN"), ("left", "VERB"), ("left", "ADJ"),
                 ("running", "VERB"), ("running", "NOUN"), ("xyzzy", "NOUN")]
        assert loaded.predict_dict(pairs) == t.predict_dict(pairs)

    def test_legacy_load(self, tmp_path):
        """
        A checkpoint in the old (word_dict, composite_dict) format should
        load transparently and produce correct predictions.
        """
        from stanza.models.lemma.vocab import MultiVocab, Vocab

        word_dict      = {"running": "run", "left": "leave"}
        composite_dict = {("left", "ADJ"): "left",   # differs -> must be kept
                          ("running", "VERB"): "run"} # same as word_dict -> redundant

        char_vocab = Vocab("abcdefghijklmnopqrstuvwxyz", "en")
        pos_vocab  = Vocab(["VERB", "ADJ"], "en")
        vocab      = MultiVocab({'char': char_vocab, 'pos': pos_vocab})

        legacy_checkpoint = {
            'model':    None,
            'dicts':    (word_dict, composite_dict),
            # no 'dicts_version' key — simulates an old checkpoint
            'vocab':    vocab.state_dict(),
            'config':   {'dict_only': True, 'caseless': False,
                         'charlm_forward_file': None, 'charlm_backward_file': None},
            'contextual': [],
        }
        save_path = str(tmp_path / "legacy_lemmatizer.pt")
        torch.save(legacy_checkpoint, save_path, _use_new_zipfile_serialization=False)

        loaded = trainer.Trainer(model_file=save_path, device='cpu')

        # pos-independent fallback
        assert loaded.predict_dict([("running", "NOUN")]) == ["run"]
        # pos-specific entry preserved because it differs
        assert loaded.predict_dict([("left", "ADJ")])  == ["left"]
        # pos-independent fallback for VERB (redundant composite was dropped)
        assert loaded.predict_dict([("left", "VERB")]) == ["leave"]
        # unknown word
        assert loaded.predict_dict([("xyzzy", "NOUN")]) == ["xyzzy"]
