"""
End-to-end training tests for the MWT expander, and unit tests for its
dictionary component.

The MWT expander has three operating modes:
  1. seq2seq          -- learns to generate expanded text character by character
  2. force_exact_pieces / CharacterClassifier -- learns where to split the
     token; the text of each piece is copied exactly from the input.
     This is tested separately in test_character_classifier.py.
  3. dict_only        -- a frequency-based dictionary, no neural model

The tests here cover modes 1 and 3, plus the dict methods in isolation,
plus the auto-detection logic that selects the model type from the training data.

Note: train() returns (trainer, _) and main() propagates that, so callers
must unpack accordingly.
"""

import os
import pytest
import torch

from stanza.models import mwt_expander
from stanza.models.mwt.character_classifier import CharacterClassifier
from stanza.models.mwt.data import BinaryDataLoader, DataLoader
from stanza.models.mwt.trainer import Trainer
from stanza.models.common.seq2seq_model import Seq2SeqModel
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# ---------------------------------------------------------------------------
# Training data
#
# FR_TRAIN / FR_DEV: French-style MWT (du = de le, au = à le, des = de les).
#   The expansions are NOT exact character subsets of the token, so
#   force_exact_pieces auto-detects as False and the seq2seq path is used.
#
# EN_TRAIN / EN_DEV: English-style possessives (Elena's -> Elena 's).
#   The expansion IS composed of exact character subsets, so
#   force_exact_pieces auto-detects as True and CharacterClassifier is used.
# ---------------------------------------------------------------------------

FR_TRAIN = """
# text = du pain
1-2	du	_	_	_	_	_	_	_	_
1	de	de	ADP	_	_	2	case	_	_
2	le	le	DET	_	_	3	det	_	_
3	pain	pain	NOUN	_	_	0	root	_	_

# text = au marché
1-2	au	_	_	_	_	_	_	_	_
1	à	à	ADP	_	_	2	case	_	_
2	le	le	DET	_	_	3	det	_	_
3	marché	marché	NOUN	_	_	0	root	_	_

# text = des amis
1-2	des	_	_	_	_	_	_	_	_
1	de	de	ADP	_	_	2	case	_	_
2	les	les	DET	_	_	3	det	_	_
3	amis	ami	NOUN	_	_	0	root	_	_

# text = au café du matin
1-2	au	_	_	_	_	_	_	_	_
1	à	à	ADP	_	_	2	case	_	_
2	le	le	DET	_	_	5	det	_	_
3	café	café	NOUN	_	_	0	root	_	_
4-5	du	_	_	_	_	_	_	_	_
4	de	de	ADP	_	_	5	case	_	_
5	le	le	DET	_	_	3	nmod	_	_
6	matin	matin	NOUN	_	_	3	nmod	_	_

""".lstrip()

FR_DEV = """
# text = des livres
1-2	des	_	_	_	_	_	_	_	_
1	de	de	ADP	_	_	2	case	_	_
2	les	les	DET	_	_	3	det	_	_
3	livres	livre	NOUN	_	_	0	root	_	_

""".lstrip()

# English possessives: the expansion is always the token split at the
# apostrophe, so both pieces are exact substrings of the original token.
# This triggers the CharacterClassifier auto-detection path.
EN_TRAIN = """
# text = Elena's motorcycle tour
1-2	Elena's	_	_	_	_	_	_	_	_
1	Elena	Elena	PROPN	NNP	Number=Sing	4	nmod:poss	4:nmod:poss	_
2	's	's	PART	POS	_	1	case	1:case	_
3	motorcycle	motorcycle	NOUN	NN	Number=Sing	4	compound	4:compound	_
4	tour	tour	NOUN	NN	Number=Sing	0	root	0:root	_

# text = women's reproductive health
1-2	women's	_	_	_	_	_	_	_	_
1	women	woman	NOUN	NNS	Number=Plur	4	nmod:poss	4:nmod:poss	_
2	's	's	PART	POS	_	1	case	1:case	_
3	reproductive	reproductive	ADJ	JJ	Degree=Pos	4	amod	4:amod	_
4	health	health	NOUN	NN	Number=Sing	0	root	0:root	SpaceAfter=No

""".lstrip()

EN_DEV = """
# text = The Children's Project
1	The	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
2-3	Children's	_	_	_	_	_	_	_	_
2	Children	Children	PROPN	NNP	Number=Sing	4	nmod:poss	4:nmod:poss	_
3	's	's	PART	POS	_	2	case	2:case	_
4	Project	Project	PROPN	NNP	Number=Sing	0	root	0:root	_

""".lstrip()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_training(tmp_path, train_text=FR_TRAIN, dev_text=FR_DEV, extra_args=None):
    """
    Write CoNLL-U files, run mwt_expander.main(), and return the trainer.

    main() returns (trainer, _) from train(), so we unpack and return
    only the trainer for convenience.
    """
    train_file = str(tmp_path / "fr_test.train.conllu")
    dev_file   = str(tmp_path / "fr_test.dev.conllu")
    output_file = str(tmp_path / "fr_test.dev.pred.conllu")
    model_name = "fr_test_mwt.pt"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(dev_file, "w", encoding="utf-8") as f:
        f.write(dev_text)

    args = [
        "--train_file",  train_file,
        "--eval_file",   dev_file,
        "--gold_file",   dev_file,
        "--output_file", output_file,
        "--lang",        "fr",
        "--shorthand",   "fr_test",
        "--save_dir",    str(tmp_path),
        "--save_name",   model_name,
        "--num_epoch",   "2",
        "--log_step",    "10",
        # keep the model tiny for speed
        "--hidden_dim",  "16",
        "--emb_dim",     "8",
    ]
    if extra_args:
        args += extra_args

    result = mwt_expander.main(args=args)
    trainer, _ = result
    return trainer, str(tmp_path / model_name)


# ---------------------------------------------------------------------------
# End-to-end training tests
# ---------------------------------------------------------------------------

class TestMWTTraining:

    def test_basic_train(self, tmp_path):
        """
        Smoke test: training should not crash and should produce a saved model
        using the seq2seq path (French MWT are not exact character splits).
        --no_force_exact_pieces is passed explicitly so auto-detection is
        bypassed; the auto-detection behavior is tested separately.
        """
        trainer, model_file = run_training(tmp_path, extra_args=["--no_force_exact_pieces"])
        assert os.path.exists(model_file)
        assert trainer.model is not None
        assert isinstance(trainer.model, Seq2SeqModel), \
            "Expected seq2seq model for non-exact-pieces MWT data"

    def test_autodetect_seq2seq(self, tmp_path):
        """
        Without any --force_exact_pieces flag, the training loop should
        detect that French MWT are not composed of their subwords and
        automatically choose the seq2seq model.
        """
        trainer, model_file = run_training(tmp_path)
        assert os.path.exists(model_file)
        assert not trainer.args['force_exact_pieces'], \
            "Expected auto-detection to leave force_exact_pieces unset/False for French MWT"
        assert isinstance(trainer.model, Seq2SeqModel), \
            "Expected seq2seq model when auto-detection picks non-exact-pieces path"

    def test_autodetect_character_classifier(self, tmp_path):
        """
        Without any --force_exact_pieces flag, the training loop should
        detect that English possessives ARE composed of their subwords
        (Elena's -> Elena + 's) and automatically choose the
        CharacterClassifier model.
        """
        trainer, model_file = run_training(tmp_path, train_text=EN_TRAIN, dev_text=EN_DEV)
        assert os.path.exists(model_file)
        assert trainer.args['force_exact_pieces'] is True, \
            "Expected auto-detection to set force_exact_pieces=True for English possessives"
        assert isinstance(trainer.model, CharacterClassifier), \
            "Expected CharacterClassifier when auto-detection picks exact-pieces path"

    def test_dict_populated_after_training(self, tmp_path):
        """
        After training, the expansion_dict should contain the MWT from
        the training data with their correct expansions.
        """
        trainer, _ = run_training(tmp_path)

        # all three MWT patterns appear in FR_TRAIN
        assert "du"  in trainer.expansion_dict, "Expected 'du' in expansion_dict"
        assert "au"  in trainer.expansion_dict, "Expected 'au' in expansion_dict"
        assert "des" in trainer.expansion_dict, "Expected 'des' in expansion_dict"

        assert trainer.expansion_dict["du"]  == "de le"
        assert trainer.expansion_dict["au"]  == "à le"
        assert trainer.expansion_dict["des"] == "de les"

    def test_seq2seq_inference(self, tmp_path):
        """
        After training the seq2seq model, predict() should run without error
        and return one expansion string per MWT in the eval doc.

        FR_DEV has one sentence with one MWT ("des"), so we expect exactly
        one prediction.  We do not assert the exact text — two epochs of
        training on tiny data is not reliable enough for that — but we do
        assert it is a non-empty string, which rules out the blank-output
        fallback path in predict() that substitutes the original token.
        """
        trainer, model_file = run_training(tmp_path, extra_args=["--no_force_exact_pieces"])
        loaded = Trainer(model_file=model_file)

        doc = CoNLL.conll2doc(input_str=FR_DEV)
        dataloader = DataLoader(doc, 10, loaded.args, vocab=loaded.vocab,
                                evaluation=True, expand_unk_vocab=True)
        preds = []
        for batch in dataloader.to_loader():
            preds += loaded.predict(batch)

        # FR_DEV has one MWT ("des")
        assert len(preds) == 1, f"Expected 1 prediction, got {len(preds)}"
        assert isinstance(preds[0], str) and preds[0], \
            f"Expected a non-empty string prediction, got {preds[0]!r}"

    def test_character_classifier_inference(self, tmp_path):
        """
        After training the CharacterClassifier model on English possessives,
        predict() should return one expansion per MWT.

        EN_DEV has one sentence with one MWT ("Children's"), so we expect
        exactly one prediction.  As with the seq2seq test we don't assert
        exact text given the short training run, but we do assert it is
        non-empty and contains a space (i.e. the token was actually split).
        """
        trainer, model_file = run_training(tmp_path, train_text=EN_TRAIN, dev_text=EN_DEV)
        loaded = Trainer(model_file=model_file)

        doc = CoNLL.conll2doc(input_str=EN_DEV)
        dataloader = BinaryDataLoader(doc, 10, loaded.args, vocab=loaded.vocab,
                                      evaluation=True, expand_unk_vocab=True)
        preds = []
        for batch in dataloader.to_loader():
            preds += loaded.predict(batch, never_decode_unk=True, vocab=dataloader.vocab)

        # EN_DEV has one MWT ("Children's")
        assert len(preds) == 1, f"Expected 1 prediction, got {len(preds)}"
        assert isinstance(preds[0], str) and preds[0], \
            f"Expected a non-empty string prediction, got {preds[0]!r}"
        assert ' ' in preds[0], \
            f"Expected the prediction to contain a space (token was split), got {preds[0]!r}"

    def test_dict_only_inference(self, tmp_path):
        """
        The dict-only model's predict_dict() should return exact expansions
        for known MWT and the original token for unknowns.

        Unlike the neural inference tests, dict predictions are fully
        deterministic from the training data, so we can assert exact values.
        """
        trainer, _ = run_training(tmp_path, extra_args=["--dict_only", "--no_force_exact_pieces"])

        doc = CoNLL.conll2doc(input_str=FR_DEV)
        # get_mwt_expansions(evaluation=True) returns the surface forms of MWT tokens
        mwt_tokens = doc.get_mwt_expansions(evaluation=True)

        preds = trainer.predict_dict(mwt_tokens)

        # FR_DEV has one MWT: "des" -> "de les"
        assert len(preds) == 1, f"Expected 1 prediction, got {len(preds)}"
        assert preds[0] == "de les", \
            f"Expected dict to expand 'des' to 'de les', got {preds[0]!r}"

    def test_save_load_roundtrip(self, tmp_path):
        """
        Save after training, reload, and verify the model type and
        expansion_dict are preserved.
        """
        trainer, model_file = run_training(tmp_path)
        assert os.path.exists(model_file)

        loaded = Trainer(model_file=model_file)
        assert isinstance(loaded.model, Seq2SeqModel)
        assert loaded.expansion_dict == trainer.expansion_dict

    def test_save_load_resave(self, tmp_path):
        """
        Load a saved model and save it again — the resaved checkpoint
        should contain all expected keys.
        """
        _, model_file = run_training(tmp_path)
        loaded = Trainer(model_file=model_file)

        resave_path = str(tmp_path / "resaved_mwt.pt")
        loaded.save(resave_path)
        assert os.path.exists(resave_path)

        checkpoint = torch.load(resave_path, lambda storage, loc: storage, weights_only=True)
        assert 'model'  in checkpoint
        assert 'dict'   in checkpoint
        assert 'vocab'  in checkpoint
        assert 'config' in checkpoint

    def test_dict_only_train(self, tmp_path):
        """
        With --dict_only, no neural model should be built and the
        expansion_dict should still be populated from the training data.
        """
        trainer, model_file = run_training(tmp_path, extra_args=["--dict_only", "--no_force_exact_pieces"])
        assert os.path.exists(model_file)

        # no neural model
        assert trainer.model is None

        # dict should still be populated
        assert trainer.expansion_dict.get("du")  == "de le"
        assert trainer.expansion_dict.get("au")  == "à le"
        assert trainer.expansion_dict.get("des") == "de les"

        # unknown token falls back to identity
        assert trainer.predict_dict(["inconnu"]) == ["inconnu"]


# ---------------------------------------------------------------------------
# Dictionary unit tests
#
# These test the dict methods in isolation using a minimal Trainer
# constructed via __new__, analogous to TestDictLemmatizer in
# test_lemma_trainer.py.  The MWT dict is simpler than the lemma dict:
# it has one dict (expansion_dict) keyed by word only, no POS.
# ---------------------------------------------------------------------------

class TestMWTDict:

    @pytest.fixture
    def empty_trainer(self):
        """
        A minimal Trainer with an empty expansion_dict and no neural model.
        """
        t = Trainer.__new__(Trainer)
        t.args = {'dict_only': True}
        t.expansion_dict = {}
        t.model = None
        return t

    def test_train_dict_basic(self, empty_trainer):
        """
        train_dict should populate expansion_dict from (word, expansion) pairs.
        """
        pairs = [("du", "de le"), ("au", "à le"), ("des", "de les")]
        empty_trainer.train_dict(pairs)

        assert empty_trainer.expansion_dict["du"]  == "de le"
        assert empty_trainer.expansion_dict["au"]  == "à le"
        assert empty_trainer.expansion_dict["des"] == "de les"

    def test_train_dict_frequency_priority(self, empty_trainer):
        """
        When a token appears with multiple expansions, the most frequent wins.
        """
        pairs = [("du", "de le"),
                 ("du", "de le"),
                 ("du", "de le"),
                 ("du", "de la")]  # less frequent
        empty_trainer.train_dict(pairs)

        assert empty_trainer.expansion_dict["du"] == "de le"

    def test_train_dict_identity_excluded(self, empty_trainer):
        """
        Pairs where word == expansion (i.e. no actual expansion) should
        not be stored — these are non-MWT tokens passed in for training
        signal via --non_mwt_replacement.
        """
        pairs = [("pain", "pain"),   # identity — should be skipped
                 ("du", "de le")]
        empty_trainer.train_dict(pairs)

        assert "pain" not in empty_trainer.expansion_dict
        assert empty_trainer.expansion_dict["du"] == "de le"

    def test_predict_dict_known(self, empty_trainer):
        """
        Known tokens should be expanded correctly.
        """
        empty_trainer.expansion_dict["du"] = "de le"
        assert empty_trainer.predict_dict(["du"]) == ["de le"]

    def test_predict_dict_unknown_fallback(self, empty_trainer):
        """
        Unknown tokens should be returned as-is.
        """
        assert empty_trainer.predict_dict(["inconnu"]) == ["inconnu"]

    def test_dict_expansion_uppercase(self, empty_trainer):
        """
        dict_expansion should handle ALL-CAPS by looking up the lowercase
        form and uppercasing the result.
        """
        empty_trainer.expansion_dict["du"] = "de le"
        result = empty_trainer.dict_expansion("DU")
        assert result == "DE LE"

    def test_dict_expansion_leading_cap(self, empty_trainer):
        """
        dict_expansion should handle Leading-cap by looking up lowercase
        and capitalising only the first character of the result.
        """
        empty_trainer.expansion_dict["du"] = "de le"
        result = empty_trainer.dict_expansion("Du")
        assert result == "De le"

    def test_dict_expansion_unknown(self, empty_trainer):
        """
        dict_expansion should return None for tokens not in the dict
        under any casing variant.
        """
        result = empty_trainer.dict_expansion("inconnu")
        assert result is None

    def test_ensemble_prefers_dict(self, empty_trainer):
        """
        ensemble() should use the dict expansion for known tokens and
        fall back to the seq2seq prediction for unknown ones.
        """
        empty_trainer.expansion_dict["du"] = "de le"

        cands      = ["du",      "inconnu"]
        seq2_preds = ["de la",   "some seq2seq guess"]
        result = empty_trainer.ensemble(cands, seq2_preds)

        assert result == ["de le", "some seq2seq guess"]

    def test_ensemble_casing(self, empty_trainer):
        """
        ensemble() should apply casing logic when looking up dict entries,
        so DU and Du both expand correctly.
        """
        empty_trainer.expansion_dict["du"] = "de le"

        result = empty_trainer.ensemble(["DU", "Du"], ["wrong", "wrong"])
        assert result == ["DE LE", "De le"]
