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
from types import SimpleNamespace

import pytest
import torch

from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.common.vocab import PAD_ID, UNK_ID, VOCAB_PREFIX, VOCAB_PREFIX_SIZE
from stanza.models.lemma.trainer import Trainer
from stanza.models.lemma_classifier import train_lstm_model
from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.evaluate_models import evaluate_model
from stanza.models.lemma_classifier.lstm_model import LemmaClassifierLSTM
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


# A sentence with an 's token and a UPOS tag the classifier never saw
# in training.  This happened with the Greek classifier, which had no
# SYM or INTJ in its training data.
UNSEEN_TAG = "NOTATAG"
CONLLU_UNSEEN_TAG_SENTENCE = """\
# sent_id = ctx-004
# text = She 's happy :)
1\tShe\tshe\tPRON\tPRP\tCase=Nom\t3\tnsubj\t3:nsubj\t_
2\t's\tbe\tAUX\tVBZ\tMood=Ind\t3\tcop\t3:cop\t_
3\thappy\thappy\tADJ\tJJ\tDegree=Pos\t0\troot\t0:root\t_
4\t:)\t:)\t%s\tNFP\t_\t3\tdiscourse\t3:discourse\tSpaceAfter=No

""".lstrip() % UNSEEN_TAG


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

    def test_unseen_upos_tag(self, tmp_path, trained_s_classifier):
        """
        A UPOS tag the classifier never saw in training must not crash prediction
        """
        assert UNSEEN_TAG not in trained_s_classifier.upos_to_id

        conllu_file = str(tmp_path / "unseen_tag.conllu")
        with open(conllu_file, "w", encoding="utf-8") as f:
            f.write(CONLLU_UNSEEN_TAG_SENTENCE)
        doc = CoNLL.conll2doc(conllu_file)

        trainer = _make_dict_only_trainer(trained_s_classifier)
        preds = _flat_preds(doc)

        result = trainer.update_contextual_preds(doc, preds)
        assert result[1] in {"be", "have"}
        assert result[0] == PLACEHOLDER
        assert result[2] == PLACEHOLDER
        assert result[3] == PLACEHOLDER


def test_convert_tags_unseen(trained_s_classifier):
    """
    Unseen tags map to <UNK>, and the padding row of the tag embedding stays all zeros
    """
    upos_to_id = trained_s_classifier.upos_to_id
    assert [upos_to_id[tag] for tag in VOCAB_PREFIX] == list(range(len(VOCAB_PREFIX)))
    converted = trained_s_classifier.convert_tags([["AUX", UNSEEN_TAG]])
    assert converted == [[upos_to_id["AUX"], UNK_ID]]
    assert trained_s_classifier.upos_emb.padding_idx == PAD_ID
    assert not trained_s_classifier.upos_emb.weight[PAD_ID].any()

def test_loaded_classifier_in_eval_mode(trained_s_classifier):
    """
    A loaded classifier is used for prediction, so it should not apply dropout
    """
    assert not trained_s_classifier.training

def test_prediction_independent_of_batch(trained_s_classifier):
    """
    A sentence's logits should not depend on the length of the other sentences in its batch
    """
    model = trained_s_classifier
    short = ["She", "'s", "happy", "."]
    short_tags = ["PRON", "AUX", "ADJ", "PUNCT"]
    long = ["It", "'s", "been", "a", "very", "long", "day", "at", "the", "office", "today", "."]
    long_tags = ["PRON", "AUX", "AUX", "DET", "ADV", "ADJ", "NOUN", "ADP", "DET", "NOUN", "NOUN", "PUNCT"]

    with torch.no_grad():
        alone = model(torch.tensor([1]), [short], model.convert_tags([short_tags]))
        batched = model(torch.tensor([1, 1]), [short, long], model.convert_tags([short_tags, long_tags]))
        reversed_batch = model(torch.tensor([1, 1]), [long, short], model.convert_tags([long_tags, short_tags]))
    assert torch.allclose(alone[0], batched[0], atol=1e-5)
    assert torch.allclose(alone[0], reversed_batch[1], atol=1e-5)

def test_drop_tags():
    """
    drop_tags replaces context tags with <UNK>, but never padding or the target word
    """
    fake_model = SimpleNamespace(upos_dropout=1.0, unknown_upos_id=UNK_ID)
    upos_tags = torch.tensor([[5, 6, 7, 8],
                              [9, 10, PAD_ID, PAD_ID]])
    dropped = LemmaClassifierLSTM.drop_tags(fake_model, upos_tags, [1, 0])
    expected = torch.tensor([[UNK_ID, 6, UNK_ID, UNK_ID],
                             [9, UNK_ID, PAD_ID, PAD_ID]])
    assert torch.equal(dropped, expected)

    fake_model.upos_dropout = 0.0
    assert torch.equal(LemmaClassifierLSTM.drop_tags(fake_model, upos_tags, [1, 0]), upos_tags)

def test_evaluate_model_restores_mode(tmp_path, trained_s_classifier):
    """
    evaluate_model scores in eval mode, then returns the model to the mode it was in
    """
    converted_files = convert_english_dataset(tmp_path)
    model = trained_s_classifier
    try:
        model.train()
        evaluate_model(model, converted_files[1], verbose=False)
        assert model.training
    finally:
        model.eval()
    evaluate_model(model, converted_files[1], verbose=False)
    assert not model.training

def test_delta_embedding_reserved_rows(trained_s_classifier):
    """
    Known words start after the reserved rows, and unknown words get the zero <UNK> row
    """
    model = trained_s_classifier
    assert model.known_words[:VOCAB_PREFIX_SIZE] == VOCAB_PREFIX
    assert min(model.known_word_map.values()) == VOCAB_PREFIX_SIZE
    # one extra row so releases from before VOCAB_PREFIX can load the model
    assert model.delta_embedding.num_embeddings == len(model.known_words) + 1
    assert model.unknown_word_id == UNK_ID
    # <UNK> never appears in training, so both reserved rows are still zero after training
    assert not model.delta_embedding.weight[PAD_ID].any()
    assert not model.delta_embedding.weight[UNK_ID].any()

def _build_lstm(pretrain_file, known_words):
    model_args = {"hidden_dim": 8, "num_heads": 0, "upos_emb_dim": 4}
    upos_to_id = {tag: idx for idx, tag in enumerate(VOCAB_PREFIX)}
    upos_to_id["AUX"] = len(upos_to_id)
    return LemmaClassifierLSTM(model_args, 2, load_pretrain(pretrain_file), {"be": 0, "have": 1}, upos_to_id,
                               known_words, {"'s"}, {"AUX"})

def test_delta_embedding_old_known_words(pretrain_file):
    """
    Models from before VOCAB_PREFIX number known words from 0, and unknown words use the padding row
    """
    model = _build_lstm(pretrain_file, ["'s", "she"])
    assert model.known_word_map == {"'s": 0, "she": 1}
    assert model.unknown_word_id == PAD_ID
    assert model.delta_embedding.num_embeddings == 3

def test_delta_embedding_new_known_words(pretrain_file):
    model = _build_lstm(pretrain_file, VOCAB_PREFIX + ["'s", "she"])
    assert model.known_word_map == {"'s": VOCAB_PREFIX_SIZE, "she": VOCAB_PREFIX_SIZE + 1}
    assert model.unknown_word_id == UNK_ID
    assert model.delta_embedding.num_embeddings == VOCAB_PREFIX_SIZE + 3
    assert not model.delta_embedding.weight[PAD_ID].any()
    assert not model.delta_embedding.weight[UNK_ID].any()
    assert not model.delta_embedding.weight[-1].any()
    assert model.delta_embedding.weight[VOCAB_PREFIX_SIZE].any()

def test_delta_embedding_loadable_by_old_releases(trained_s_classifier):
    """
    A VOCAB_PREFIX model must work in releases from before VOCAB_PREFIX

    Those releases build the delta embedding with len(known_words) + 1 rows,
    number words with enumerate(known_words), and send unknown words to row 0.
    """
    model = trained_s_classifier
    known_words = model.known_words

    # same shape as the old code builds, so load_state_dict works there
    assert model.get_save_dict()["params"]["delta_embedding.weight"].shape[0] == len(known_words) + 1

    # every real word gets the same row in the old numbering
    old_known_word_map = {word: idx for idx, word in enumerate(known_words)}
    for word, idx in model.known_word_map.items():
        assert old_known_word_map[word] == idx

    # unknown words get row 0 in the old code, which is zero like <UNK>
    assert torch.equal(model.delta_embedding.weight[PAD_ID], model.delta_embedding.weight[UNK_ID])
