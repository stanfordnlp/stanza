"""
Tests for Trainer.update_contextual_preds and predict_contextual.

These tests verify that an attached LemmaClassifier actually overwrites the
correct slot(s) in the flat lemma prediction list that the seq2seq lemmatizer
produces, and that tokens which are *not* target words are left untouched.

Strategy
--------
Rather than training a full seq2seq model (slow) we build a minimal Trainer
directly — dict_only mode, no seq2seq — and attach a real trained LSTM
classifier to it.  update_contextual_preds only needs the Trainer to carry
contextual_lemmatizers; it does not call the seq2seq model at all.

The classifier is trained with convert_english_dataset (the existing EWT
fixture) so it handles "'s AUX" and produces either "be" or "have".

A small CoNLL-U document with known 's tokens is parsed with CoNLL.conll2doc
and used as the doc argument.  The initial flat preds list is filled with a
placeholder lemma ("PLACEHOLDER") for every token.  After update_contextual_preds
the 's slots should contain "be" or "have" and all other slots should still
hold "PLACEHOLDER".
"""

import os

import pytest

from stanza.models.lemma.trainer import Trainer
from stanza.models.lemma_classifier import train_lstm_model
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.tests import TEST_WORKING_DIR
from stanza.tests.lemma_classifier.test_data_preparation import convert_english_dataset
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# ---------------------------------------------------------------------------
# A small CoNLL-U document containing two sentences with "'s AUX" tokens.
# One is clearly "be" (copula), one is clearly "have" (perfect auxiliary).
# We use real UD annotations so CoNLL.conll2doc produces a proper doc.
# ---------------------------------------------------------------------------

CONLLU_TWO_S_SENTENCES = """\
# sent_id = ctx-001
# text = She 's happy .
1\tShe\tshe\tPRON\tPRP\tCase=Nom\t3\tnsubj\t3:nsubj\t_
2\t's\tbe\tAUX\tVBZ\tMood=Ind\t3\tcop\t3:cop\t_
3\thappy\thappy\tADJ\tJJ\tDegree=Pos\t0\troot\t0:root\t_
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t3:punct\tSpaceAfter=No

# sent_id = ctx-002
# text = It 's finished .
1\tIt\tit\tPRON\tPRP\tCase=Nom\t3\tnsubj\t3:nsubj\t_
2\t's\thave\tAUX\tVBZ\tMood=Ind\t3\taux\t3:aux\t_
3\tfinished\tfinish\tVERB\tVBN\tTense=Past\t0\troot\t0:root\t_
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t3:punct\tSpaceAfter=No

""".lstrip()

# A sentence with no 's token — used to confirm non-targets are untouched.
CONLLU_NO_S_SENTENCE = """\
# sent_id = ctx-003
# text = The cat sat here .
1\tThe\tthe\tDET\tDT\tDefinite=Def\t2\tdet\t2:det\t_
2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t3:nsubj\t_
3\tsat\tsit\tVERB\tVBD\tMood=Ind\t0\troot\t0:root\t_
4\there\there\tADV\tRB\t_\t3\tadvmod\t3:advmod\t_
5\t.\t.\tPUNCT\t.\t_\t3\tpunct\t3:punct\tSpaceAfter=No

""".lstrip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLACEHOLDER = "PLACEHOLDER"

def _make_dict_only_trainer(classifier):
    """
    Build the lightest possible Trainer: dict_only (no seq2seq weights),
    empty dicts, and a single attached contextual lemmatizer.

    update_contextual_preds only touches contextual_lemmatizers; it never
    calls the seq2seq model.
    """
    args = {
        'dict_only': True,
        'caseless': False,
        # remaining keys are not read by update_contextual_preds
        'shorthand': 'en_test',
        'lang': 'en',
    }
    # Trainer.__init__ with model_file=None and dict_only=True skips
    # building the seq2seq model; we just fill in the rest by hand.
    t = object.__new__(Trainer)
    t.args = args
    t.vocab = None
    t.model = None
    t.word_dict = {}
    t.composite_dict = {}
    t.contextual_lemmatizers = [classifier]
    t.caseless = False
    return t


def _flat_preds(doc):
    """
    Build a flat list of PLACEHOLDER strings, one per word across all
    sentences — this is what the seq2seq model would normally produce.
    """
    return [PLACEHOLDER for sent in doc.sentences for _ in sent.words]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pretrain_file():
    return os.path.join(TEST_WORKING_DIR, "in", "tiny_emb.pt")


@pytest.fixture(scope="module")
def trained_s_classifier(tmp_path_factory, pretrain_file):
    """
    Train a tiny LSTM classifier on the EWT "'s AUX" data and return the
    loaded LemmaClassifier object.
    """
    tmp = tmp_path_factory.mktemp("ctx_preds_classifier")
    converted_files = convert_english_dataset(tmp)
    save_name = str(tmp / "s_classifier.pt")
    train_lstm_model.main([
        "--wordvec_pretrain_file", pretrain_file,
        "--save_name",             save_name,
        "--train_file",            converted_files[0],
        "--eval_file",             converted_files[1],
    ])
    return LemmaClassifier.load(save_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpdateContextualPreds:

    def test_target_slots_are_overwritten(self, tmp_path, trained_s_classifier):
        """
        update_contextual_preds must replace the PLACEHOLDER at every "'s AUX"
        position with a real lemma ("be" or "have").
        """
        conllu_file = str(tmp_path / "two_s.conllu")
        with open(conllu_file, "w", encoding="utf-8") as f:
            f.write(CONLLU_TWO_S_SENTENCES)
        doc = CoNLL.conll2doc(conllu_file)

        trainer = _make_dict_only_trainer(trained_s_classifier)
        preds = _flat_preds(doc)

        result = trainer.update_contextual_preds(doc, preds)

        # Sentence 1: tokens [She, 's, happy, .]  -> index 1 is 's
        # Sentence 2: tokens [It,  's, finished, .] -> index 5 overall (4 + 1)
        s_indices = [1, 5]
        valid_lemmas = {"be", "have"}
        for idx in s_indices:
            assert result[idx] in valid_lemmas, (
                f"Expected result[{idx}] to be 'be' or 'have', got {result[idx]!r}"
            )

    def test_non_target_slots_are_unchanged(self, tmp_path, trained_s_classifier):
        """
        Tokens that are not "'s AUX" must keep their original prediction
        (PLACEHOLDER) after update_contextual_preds.
        """
        conllu_file = str(tmp_path / "two_s.conllu")
        with open(conllu_file, "w", encoding="utf-8") as f:
            f.write(CONLLU_TWO_S_SENTENCES)
        doc = CoNLL.conll2doc(conllu_file)

        trainer = _make_dict_only_trainer(trained_s_classifier)
        preds = _flat_preds(doc)

        result = trainer.update_contextual_preds(doc, preds)

        s_indices = {1, 5}
        for idx, lemma in enumerate(result):
            if idx not in s_indices:
                assert lemma == PLACEHOLDER, (
                    f"Expected non-target result[{idx}] to remain {PLACEHOLDER!r}, "
                    f"got {lemma!r}"
                )

    def test_sentence_with_no_target_token_unchanged(self, tmp_path, trained_s_classifier):
        """
        A document with no "'s AUX" tokens must be returned completely
        unchanged.
        """
        conllu_file = str(tmp_path / "no_s.conllu")
        with open(conllu_file, "w", encoding="utf-8") as f:
            f.write(CONLLU_NO_S_SENTENCE)
        doc = CoNLL.conll2doc(conllu_file)

        trainer = _make_dict_only_trainer(trained_s_classifier)
        preds = _flat_preds(doc)
        original_preds = list(preds)

        result = trainer.update_contextual_preds(doc, preds)

        assert result == original_preds, (
            f"Expected preds to be unchanged when no target tokens are present, "
            f"got {result}"
        )

    def test_flat_length_preserved(self, tmp_path, trained_s_classifier):
        """
        update_contextual_preds must return a flat list of the same length
        as the input, regardless of sentence boundaries.
        """
        combined = CONLLU_TWO_S_SENTENCES + CONLLU_NO_S_SENTENCE
        conllu_file = str(tmp_path / "combined.conllu")
        with open(conllu_file, "w", encoding="utf-8") as f:
            f.write(combined)
        doc = CoNLL.conll2doc(conllu_file)

        trainer = _make_dict_only_trainer(trained_s_classifier)
        preds = _flat_preds(doc)

        result = trainer.update_contextual_preds(doc, preds)
        assert len(result) == len(preds), (
            f"Expected output length {len(preds)}, got {len(result)}"
        )
