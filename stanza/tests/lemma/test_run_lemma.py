"""
Integration test for run_lemma.run_treebank.

Verifies the end-to-end flow in which:
  1. A seq2seq lemmatizer is trained.
  2. Two lemma classifiers are trained via run_lemma_classifier (one for
     "'s AUX" and one for "her PRON"), matching the en_combined DATASET_TARGETS.
  3. Both classifiers are attached to the seq2seq model by attach_lemma_classifier.
  4. The reloaded Trainer reports has_contextual_lemmatizers() == True,
     contains exactly two classifiers, and each targets the expected word
     with the expected UPOS tag.

All file I/O is rooted at tmp_path; no monkey-patching is required because:
  - paths["LEMMA_DATA_DIR"] is pointed at a temp directory.
  - --save_dir in extra_args steers the seq2seq model into tmp_path.
  - command_args.lemma_classifier_save_dir steers the classifier models into
    tmp_path (via the --lemma_classifier_save_dir argument added to run_lemma).
  - --wordvec_pretrain_file in command_args is forwarded into the
    run_lemma_classifier call

An important note here is that it uses run_lemma.py directly.
This means that the results will be wrong and have assertion failures
if the underlying assumptions that make the model change.
However, that is a good thing - our goal is to test exactly how
run_lemma.py works in practice, so in the unlikely event that
the downstream target words change, we want the test to change as well.
(Theoretically we could disambiguate 'wound' or 'found', for example,
but that would require a lot of artificial training data.)
"""

import argparse
import json
import os

import pytest

from stanza.models.lemma.trainer import Trainer
from stanza.tests import TEST_WORKING_DIR
import stanza.utils.datasets.prepare_lemma_classifier as prepare_lemma_classifier
from stanza.utils.training import run_lemma
from stanza.utils.training.common import Mode

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# ---------------------------------------------------------------------------
# CoNLL-U data
# ---------------------------------------------------------------------------
# Sentences cover both "'s AUX" (lemmas "be" / "have") and "her PRON"
# (lemmas "her" / "she") so a single pair of files drives the seq2seq step.

TRAIN_CONLLU = """\
# sent_id = test-001
# text = She 's happy .
1\tShe\tshe\tPRON\tPRP\tCase=Nom\t3\tnsubj\t3:nsubj\t_
2\t's\tbe\tAUX\tVBZ\tMood=Ind\t3\tcop\t3:cop\t_
3\thappy\thappy\tADJ\tJJ\tDegree=Pos\t0\troot\t0:root\t_
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t3:punct\tSpaceAfter=No

# sent_id = test-002
# text = She 's finished her work .
1\tShe\tshe\tPRON\tPRP\tCase=Nom\t3\tnsubj:pass\t3:nsubj:pass\t_
2\t's\thave\tAUX\tVBZ\tMood=Ind\t3\taux\t3:aux\t_
3\tfinished\tfinish\tVERB\tVBN\tTense=Past\t0\troot\t0:root\t_
4\ther\ther\tPRON\tPRP$\tCase=Gen\t5\tnmod:poss\t5:nmod:poss\t_
5\twork\twork\tNOUN\tNN\tNumber=Sing\t3\tobj\t3:obj\tSpaceAfter=No
6\t.\t.\tPUNCT\t.\t_\t3\tpunct\t3:punct\tSpaceAfter=No

# sent_id = test-003
# text = I gave her the book .
1\tI\tI\tPRON\tPRP\tCase=Nom\t2\tnsubj\t2:nsubj\t_
2\tgave\tgive\tVERB\tVBD\tMood=Ind\t0\troot\t0:root\t_
3\ther\tshe\tPRON\tPRP\tCase=Acc\t2\tiobj\t2:iobj\t_
4\tthe\tthe\tDET\tDT\tDefinite=Def\t5\tdet\t5:det\t_
5\tbook\tbook\tNOUN\tNN\tNumber=Sing\t2\tobj\t2:obj\tSpaceAfter=No
6\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\tSpaceAfter=No

# sent_id = test-004
# text = She 's been here before .
1\tShe\tshe\tPRON\tPRP\tCase=Nom\t4\tnsubj\t4:nsubj\t_
2\t's\tbe\tAUX\tVBZ\tMood=Ind\t4\taux\t4:aux\t_
3\tbeen\tbe\tAUX\tVBN\tTense=Past\t4\taux\t4:aux\t_
4\there\there\tADV\tRB\t_\t0\troot\t0:root\t_
5\tbefore\tbefore\tADV\tRB\t_\t4\tadvmod\t4:advmod\tSpaceAfter=No
6\t.\t.\tPUNCT\t.\t_\t4\tpunct\t4:punct\tSpaceAfter=No

# sent_id = test-005
# text = I saw her yesterday .
1\tI\tI\tPRON\tPRP\tCase=Nom\t2\tnsubj\t2:nsubj\t_
2\tsaw\tsee\tVERB\tVBD\tMood=Ind\t0\troot\t0:root\t_
3\ther\tshe\tPRON\tPRP\tCase=Acc\t2\tobj\t2:obj\t_
4\tyesterday\tyesterday\tADV\tRB\t_\t2\tadvmod\t2:advmod\tSpaceAfter=No
5\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\tSpaceAfter=No

""".lstrip()

DEV_CONLLU = """\
# sent_id = dev-001
# text = That 's interesting .
1\tThat\tthat\tPRON\tDT\tNumber=Sing\t3\tnsubj\t3:nsubj\t_
2\t's\tbe\tAUX\tVBZ\tMood=Ind\t3\tcop\t3:cop\t_
3\tinteresting\tinteresting\tADJ\tJJ\tDegree=Pos\t0\troot\t0:root\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t3:punct\tSpaceAfter=No

# sent_id = dev-002
# text = I called her .
1\tI\tI\tPRON\tPRP\tCase=Nom\t2\tnsubj\t2:nsubj\t_
2\tcalled\tcall\tVERB\tVBD\tMood=Ind\t0\troot\t0:root\t_
3\ther\tshe\tPRON\tPRP\tCase=Acc\t2\tobj\t2:obj\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\tSpaceAfter=No

""".lstrip()


# ---------------------------------------------------------------------------
# .lemma file helpers
# ---------------------------------------------------------------------------
# The format is exactly what prepare_dataset.DataProcessor.write_output_file
# produces: a JSON object with a "upos" list and a "sentences" array of
# {words, upos_tags, index, lemma} dicts.  We need at least two examples of
# each lemma class so train_lstm_model sees both labels during training.

def _write_lemma_file(path, target_upos, sentences):
    content = {
        "upos": [target_upos],
        "sentences": sentences,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


# "'s" AUX — two classes: "be" and "have"
_S_TRAIN = [
    {"words": ["She", "'s", "happy", "."],
     "upos_tags": ["PRON", "AUX", "ADJ", "PUNCT"], "index": 1, "lemma": "be"},
    {"words": ["She", "'s", "happy", "."],
     "upos_tags": ["PRON", "AUX", "ADJ", "PUNCT"], "index": 1, "lemma": "be"},
    {"words": ["She", "'s", "finished", "her", "work", "."],
     "upos_tags": ["PRON", "AUX", "VERB", "PRON", "NOUN", "PUNCT"], "index": 1, "lemma": "have"},
    {"words": ["She", "'s", "finished", "her", "work", "."],
     "upos_tags": ["PRON", "AUX", "VERB", "PRON", "NOUN", "PUNCT"], "index": 1, "lemma": "have"},
    {"words": ["She", "'s", "been", "here", "before", "."],
     "upos_tags": ["PRON", "AUX", "AUX", "ADV", "ADV", "PUNCT"], "index": 1, "lemma": "be"},
    {"words": ["That", "'s", "interesting", "."],
     "upos_tags": ["PRON", "AUX", "ADJ", "PUNCT"], "index": 1, "lemma": "be"},
]
_S_DEV = [
    {"words": ["That", "'s", "interesting", "."],
     "upos_tags": ["PRON", "AUX", "ADJ", "PUNCT"], "index": 1, "lemma": "be"},
    {"words": ["She", "'s", "finished", "her", "work", "."],
     "upos_tags": ["PRON", "AUX", "VERB", "PRON", "NOUN", "PUNCT"], "index": 1, "lemma": "have"},
]

# "her" PRON — two classes: "her" (possessive) and "she" (accusative)
_HER_TRAIN = [
    {"words": ["She", "'s", "finished", "her", "work", "."],
     "upos_tags": ["PRON", "AUX", "VERB", "PRON", "NOUN", "PUNCT"], "index": 3, "lemma": "her"},
    {"words": ["She", "'s", "finished", "her", "work", "."],
     "upos_tags": ["PRON", "AUX", "VERB", "PRON", "NOUN", "PUNCT"], "index": 3, "lemma": "her"},
    {"words": ["I", "gave", "her", "the", "book", "."],
     "upos_tags": ["PRON", "VERB", "PRON", "DET", "NOUN", "PUNCT"], "index": 2, "lemma": "she"},
    {"words": ["I", "gave", "her", "the", "book", "."],
     "upos_tags": ["PRON", "VERB", "PRON", "DET", "NOUN", "PUNCT"], "index": 2, "lemma": "she"},
    {"words": ["I", "saw", "her", "yesterday", "."],
     "upos_tags": ["PRON", "VERB", "PRON", "ADV", "PUNCT"], "index": 2, "lemma": "she"},
    {"words": ["I", "called", "her", "."],
     "upos_tags": ["PRON", "VERB", "PRON", "PUNCT"], "index": 2, "lemma": "she"},
]
_HER_DEV = [
    {"words": ["I", "called", "her", "."],
     "upos_tags": ["PRON", "VERB", "PRON", "PUNCT"], "index": 2, "lemma": "she"},
    {"words": ["She", "'s", "finished", "her", "work", "."],
     "upos_tags": ["PRON", "AUX", "VERB", "PRON", "NOUN", "PUNCT"], "index": 3, "lemma": "her"},
]

# Map from DATASET_TARGETS filename stem -> (upos, train sentences, dev sentences).
# The stems "s" and "her" come from prepare_lemma_classifier.DATASET_TARGETS["en_combined"].
_LC_DATA = {
    "s":   ("AUX",  _S_TRAIN,   _S_DEV),
    "her": ("PRON", _HER_TRAIN, _HER_DEV),
}


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the full pipeline once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pretrain_file():
    """Tiny pretrained embedding already present in the test suite."""
    return os.path.join(TEST_WORKING_DIR, "in", "tiny_emb.pt")


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory, pretrain_file):
    """
    Calls run_lemma.run_treebank(mode=TRAIN) for en_combined and returns the
    path to the final .pt file (containing two attached contextual lemmatizers).

    Directory layout
    ----------------
    tmp/
      data/lemma/
        en_combined.train.in.conllu
        en_combined.dev.in.conllu
      data/lemma_classifier/
        en_combined.s.{train,dev,test}.lemma
        en_combined.her.{train,dev,test}.lemma
      saved_models/lemma/             <- seq2seq model (via --save_dir extra_arg)
      saved_models/lemma_classifier/  <- classifier models (via
                                           --lemma_classifier_save_dir)

    Argument wiring
    ---------------
    paths["LEMMA_DATA_DIR"]              -> tmp/data/lemma/
    extra_args --save_dir                -> tmp/saved_models/lemma/
    command_args.lemma_classifier_save_dir
                                         -> tmp/saved_models/lemma_classifier/
    extra_args --wordvec_pretrain_file   -> forwarded into run_lemma_classifier
                                           (prevents any network download)
    """
    # Fail fast if DATASET_TARGETS has changed, rather than letting the
    # mismatch surface as a confusing target_upos or classifier-count error.
    expected_stems = {"'s", "her"}
    actual_stems = {lc.word for lc in prepare_lemma_classifier.DATASET_TARGETS["en_combined"]}
    assert actual_stems == expected_stems, (
        f"DATASET_TARGETS['en_combined'] has changed from {expected_stems} to "
        f"{actual_stems}. Update _LC_DATA, TRAIN_CONLLU, and the target_words/"
        f"target_upos assertions in this test to match the new targets."
    )

    tmp = tmp_path_factory.mktemp("run_lemma_integration")

    lemma_data_dir = tmp / "data" / "lemma"
    lc_data_dir    = tmp / "data" / "lemma_classifier"
    lemma_save_dir = tmp / "saved_models" / "lemma"
    lc_save_dir    = tmp / "saved_models" / "lemma_classifier"
    for d in (lemma_data_dir, lc_data_dir, lemma_save_dir, lc_save_dir):
        d.mkdir(parents=True, exist_ok=True)

    # en_combined is in both DATASET_MAPPING and DATASET_TARGETS, so
    # run_treebank will trigger the two-classifier pipeline automatically.
    short_name = "en_combined"

    # ------------------------------------------------------------------
    # Write CoNLL-U files consumed by lemmatizer.main
    # ------------------------------------------------------------------
    (lemma_data_dir / f"{short_name}.train.in.conllu").write_text(TRAIN_CONLLU, encoding="utf-8")
    (lemma_data_dir / f"{short_name}.dev.in.conllu").write_text(DEV_CONLLU,   encoding="utf-8")

    # ------------------------------------------------------------------
    # Write .lemma files consumed by run_lemma_classifier / train_lstm_model.
    # Filenames mirror what prepare_lemma_classifier.process_en_combined
    # would produce:  en_combined.{stem}.{split}.lemma
    # ------------------------------------------------------------------
    for stem, (upos, train_sents, dev_sents) in _LC_DATA.items():
        lc_prefix = f"{short_name}.{stem}"
        _write_lemma_file(lc_data_dir / f"{lc_prefix}.train.lemma", upos, train_sents)
        _write_lemma_file(lc_data_dir / f"{lc_prefix}.dev.lemma",   upos, dev_sents)
        _write_lemma_file(lc_data_dir / f"{lc_prefix}.test.lemma",  upos, dev_sents)

    # ------------------------------------------------------------------
    # Assemble the arguments for run_treebank
    # ------------------------------------------------------------------
    paths = {
        # run_treebank reads train/dev .conllu files from here, and
        # build_model_filename uses it to verify the train file exists.
        "LEMMA_DATA_DIR": str(lemma_data_dir),
        # this is used in run_lemma_classifier to find the data files
        # for the contextual classifiers
        "LEMMA_CLASSIFIER_DATA_DIR": str(lc_data_dir),
    }

    command_args = argparse.Namespace(
        # No charlm: keeps the test fast and avoids downloads.
        # lemma_classifier == None means "use classifiers iff charlm is set",
        # so we set it explicitly to True to always exercise the classifier path.
        charlm=None,
        save_output=False,
        force=False,
        lemma_classifier=True,
        lemma_classifier_save_dir=str(lc_save_dir),
        # save_dir on command_args is read by common.main when it appends
        # --save_dir to extra_args; since we call run_treebank directly we
        # pass --save_dir through extra_args below instead.
        save_dir=None,
        wordvec_pretrain_file=str(pretrain_file),
    )

    # extra_args are forwarded verbatim to lemmatizer.main.  The pretrain arg
    # is also forwarded into run_lemma_classifier by the code addition
    # described in the module docstring.
    extra_args = [
        "--num_epoch",             "2",
        "--log_step",              "10",
        "--save_dir",              str(lemma_save_dir),
    ]

    # ------------------------------------------------------------------
    # Run the full pipeline via run_lemma.run_treebank
    # ------------------------------------------------------------------
    run_lemma.run_treebank(
        mode=Mode.TRAIN,
        paths=paths,
        treebank=short_name,   # only used in log messages by run_treebank
        short_name=short_name,
        command_args=command_args,
        extra_args=extra_args,
    )

    # Locate the saved seq2seq model using the same helper as run_treebank so
    # the test stays correct if the filename convention ever changes.
    model_path = run_lemma.build_model_filename(paths, short_name, command_args, extra_args)
    assert model_path is not None, "build_model_filename returned None"
    assert os.path.exists(model_path), \
        f"Expected seq2seq lemmatizer at {model_path} after run_treebank"
    return model_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunLemmaAttachesTwoClassifiers:
    """
    Checks that run_lemma.run_treebank correctly trains and attaches two lemma
    classifiers for en_combined (one for "'s" AUX, one for "her" PRON).
    """

    def test_has_contextual_lemmatizers(self, trained_model_path):
        """has_contextual_lemmatizers() must be True after the full pipeline."""
        t = Trainer(model_file=trained_model_path)
        assert t.has_contextual_lemmatizers(), (
            "Expected has_contextual_lemmatizers() == True after run_lemma "
            "trained and attached both classifiers"
        )

    def test_exactly_two_classifiers(self, trained_model_path):
        """Exactly two classifiers should be embedded in the model."""
        t = Trainer(model_file=trained_model_path)
        n = len(t.contextual_lemmatizers)
        assert n == 2, (
            f"Expected 2 contextual lemmatizers "
            f"(one for \"'s\", one for \"her\"), got {n}"
        )

    def test_classifier_target_words(self, trained_model_path):
        """
        The union of target_words across both classifiers must cover "'s"
        and "her".
        """
        t = Trainer(model_file=trained_model_path)
        all_words = {w for clf in t.contextual_lemmatizers for w in clf.target_words}
        assert "'s"  in all_words, \
            f"\"'s\" missing from classifier target_words; got {all_words}"
        assert "her" in all_words, \
            f"\"her\" missing from classifier target_words; got {all_words}"

    def test_classifier_target_upos(self, trained_model_path):
        """
        Each classifier must be bound to the correct UPOS tag:
          "'s"  -> AUX
          "her" -> PRON
        """
        t = Trainer(model_file=trained_model_path)
        upos_by_word = {}
        for clf in t.contextual_lemmatizers:
            for word in clf.target_words:
                upos_by_word[word] = clf.target_upos

        s_upos = upos_by_word.get("'s", set())
        her_upos = upos_by_word.get("her", set())
        assert "AUX"  in s_upos,   f"Expected AUX in target_upos for \"'s\", got {s_upos}"
        assert "PRON" in her_upos, f"Expected PRON in target_upos for \"her\", got {her_upos}"

    def test_save_reload_preserves_classifiers(self, trained_model_path, tmp_path):
        """
        A save/reload round-trip must preserve both classifiers and their
        target_words exactly.
        """
        original = Trainer(model_file=trained_model_path)
        resave_path = str(tmp_path / "resaved_lemmatizer.pt")
        original.save(resave_path)

        reloaded = Trainer(model_file=resave_path)
        assert reloaded.has_contextual_lemmatizers()
        assert len(reloaded.contextual_lemmatizers) == len(original.contextual_lemmatizers)

        original_words = {w for c in original.contextual_lemmatizers for w in c.target_words}
        reloaded_words = {w for c in reloaded.contextual_lemmatizers for w in c.target_words}
        assert original_words == reloaded_words, (
            f"target_words changed after save/reload: "
            f"{original_words} -> {reloaded_words}"
        )
