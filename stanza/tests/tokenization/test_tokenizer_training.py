"""
Run the tokenizer for a few steps on some fake data

Uses a couple sentences of UD_English-EWT as training/dev data,
expressed in the character-label format the tokenizer expects.

The tokenizer input format is:
  - a plain text file of raw characters (no newline between sentences,
    paragraph breaks represented by a blank line)
  - a label file where each character gets one of:
      0  = not a token/sentence boundary
      1  = token boundary (end of a non-MWT token, space follows)
      2  = sentence boundary (end of last token in sentence, no MWT)
      3  = MWT boundary (this surface form expands to multiple words,
           not at end of sentence)
      4  = MWT boundary at end of sentence (rare)

Whether the model includes MWT output layers is detected automatically
from whether label 3 or 4 appears in the training data.

The MWT JSON file format is a list of [[surface, [word1, word2, ...]], count]
entries, where count is the corpus frequency of the MWT.

The charlm test requires a pretrained forward charlm for English to be
present under TEST_MODELS_DIR.  It fails (not skips) if none is found,
since the CI setup script is expected to download those models.
"""

import glob
import json
import os
import pytest
import torch

from stanza.models import tokenizer as tokenizer_module
from stanza.models.tokenization.trainer import Trainer
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]


# ---------------------------------------------------------------------------
# Helpers: build .txt / .label / _mwt.json file triples from word lists
# ---------------------------------------------------------------------------

def _make_labels(sentence_words):
    """
    Given a list of words for one sentence, produce (chars, labels).

    Each entry in sentence_words is either:
      - a plain string: a normal token
      - a tuple (surface, [word1, word2, ...]): an MWT whose surface form
        expands to the given words

    Labels assigned:
      0  mid-token character
      1  end of a normal token (not last in sentence)
      2  end of sentence, last token is a normal token
      3  end of an MWT surface form (not last in sentence)
      4  end of an MWT surface form that is also the last token in the sentence

    Returns (chars, labels) where both are lists of the same length.
    """
    chars = []
    labels = []
    for word_idx, entry in enumerate(sentence_words):
        is_mwt = isinstance(entry, tuple)
        surface = entry[0] if is_mwt else entry
        is_last = (word_idx == len(sentence_words) - 1)

        for char_idx, ch in enumerate(surface):
            chars.append(ch)
            is_last_char = (char_idx == len(surface) - 1)
            if is_last_char:
                if is_mwt:
                    labels.append(4 if is_last else 3)
                else:
                    labels.append(2 if is_last else 1)
            else:
                labels.append(0)

        if not is_last:
            # inter-token space
            chars.append(' ')
            labels.append(0)

    return chars, labels


def _build_data_files(tmp_path, sentences, prefix):
    """
    Write <prefix>.txt, <prefix>.label, and <prefix>_mwt.json files under
    tmp_path.  MWT entries are collected from any tuple entries in sentences
    and written to the JSON file with a count of 1.

    The label file format mirrors the text file: each paragraph is a flat
    string of single digit labels with no separator between them, and
    paragraphs are separated by blank lines — e.g. "01121\n\n011021\n\n".
    This is what data.py's TokenizationDataset expects: it splits on
    NEWLINE_WHITESPACE_RE then calls map(int, chunk) character by character.

    Returns (txt_path, label_path, mwt_path).
    """
    txt_paragraphs = []
    label_paragraphs = []
    mwt_entries = []

    for sentence in sentences:
        chars, labels = _make_labels(sentence)
        txt_paragraphs.append(''.join(chars))
        label_paragraphs.append(''.join(str(l) for l in labels))
        for entry in sentence:
            if isinstance(entry, tuple):
                surface, expansion_words = entry
                mwt_entries.append([[surface, expansion_words], 1])

    txt_path = str(tmp_path / f"{prefix}.txt")
    label_path = str(tmp_path / f"{prefix}.label")
    mwt_path = str(tmp_path / f"{prefix}_mwt.json")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(txt_paragraphs) + '\n\n')
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(label_paragraphs) + '\n\n')
    with open(mwt_path, 'w', encoding='utf-8') as f:
        json.dump(mwt_entries, f)

    return txt_path, label_path, mwt_path


# ---------------------------------------------------------------------------
# Shared training data — two variants: with and without MWT
# ---------------------------------------------------------------------------

# Plain sentences with no MWT expansions
TRAIN_SENTENCES_NO_MWT = [
    ["From", "the", "AP", "comes", "this", "story", ":"],
    ["Two", "of", "them", "were", "run", "by", "officials", "."],
]

# Sentences with MWT contractions.  Each MWT is a tuple of
# (surface_form, [expansion_word1, expansion_word2, ...]).
TRAIN_SENTENCES_WITH_MWT = [
    ["From", "the", "AP", "comes", "this", "story", ":"],
    ["It", ("isn't", ["is", "n't"]), "clear", "whether", "they", ("can't", ["ca", "n't"]), "confirm", "."],
]

DEV_SENTENCES = [
    ["Iraqi", "authorities", "announced", "that", "they", "had", "busted", "up", "cells", "."],
]


# ---------------------------------------------------------------------------
# Core training helper
# ---------------------------------------------------------------------------

def run_training(tmp_path, train_sentences, extra_args=None):
    """
    Run a small number of tokenizer training steps and return (trainer, args).

    train_sentences controls whether MWT output layers are built: if any
    entry uses a tuple (surface, expansion_words), labels 3/4 will appear
    and the DataLoader will set use_mwt=True automatically.

    Steps and eval_steps are kept tiny so tests finish quickly.
    """
    train_txt, train_label, train_mwt = _build_data_files(tmp_path, train_sentences, "train")
    dev_txt, dev_label, dev_mwt = _build_data_files(tmp_path, DEV_SENTENCES, "dev")

    save_dir = str(tmp_path / "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    args_list = [
        '--txt_file',       train_txt,
        '--label_file',     train_label,
        '--mwt_json_file',  train_mwt,
        '--dev_txt_file',   dev_txt,
        '--dev_label_file', dev_label,
        '--lang',           'en',
        '--shorthand',      'en_ewt',
        '--save_dir',       save_dir,
        '--steps',          '10',  # just enough to hit one eval and one save
        '--eval_steps',     '5',
        '--report_steps',   '5',
        '--hidden_dim',     '16',  # tiny model for speed
        '--emb_dim',        '8',
        '--rnn_layers',     '1',
        '--device',         'cpu',
        # use_mwt is intentionally omitted: the DataLoader detects it from data
    ]
    if extra_args:
        args_list += extra_args

    args = tokenizer_module.parse_args(args_list)

    # Replicate the three setup steps that main() does before calling train()
    args['feat_funcs'] = ['space_before', 'capitalized', 'numeric', 'end_of_para', 'start_of_para']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = tokenizer_module.model_file_name(args)
    os.makedirs(os.path.split(args['save_name'])[0], exist_ok=True)

    trainer, _ = tokenizer_module.train(args)
    return trainer, args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokenizer:

    @pytest.fixture(scope="class")
    def english_charlm_forward_file(self):
        """
        Return the path to the first available English forward charlm model.
        Fails if none is found — the CI setup script is expected to have
        downloaded these models, so their absence indicates a setup problem.
        """
        models_path = os.path.join(TEST_MODELS_DIR, "en", "forward_charlm", "*")
        models = glob.glob(models_path)
        assert len(models) >= 1, \
            f"No English forward charlm found under {models_path} — check that the CI setup script downloaded the character models"
        return models[0]

    def test_train_no_mwt(self, tmp_path):
        """
        Basic smoke test with no MWT in the training data.  The DataLoader
        should detect use_mwt=False and build a model without MWT layers.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT)
        assert os.path.exists(args['save_name']), \
            f"Expected model file at {args['save_name']}"
        assert trainer.args['use_mwt'] == False

    def test_train_with_mwt(self, tmp_path):
        """
        Smoke test with MWT tokens in the training data.  The DataLoader
        should detect use_mwt=True from the presence of labels 3/4 and
        build a model with MWT output layers.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_WITH_MWT)
        assert os.path.exists(args['save_name'])
        assert trainer.args['use_mwt'] == True

    def test_no_residual_no_hierarchical(self, tmp_path):
        """
        Verify training still works with residual and hierarchical RNN
        connections disabled.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT,
                                     extra_args=['--no-residual', '--no-hierarchical'])
        assert os.path.exists(args['save_name'])

    def test_input_dropout(self, tmp_path):
        """
        Verify --input_dropout doesn't break the training loop.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT,
                                     extra_args=['--input_dropout'])
        assert os.path.exists(args['save_name'])

    def test_conv_res(self, tmp_path):
        """
        Verify --conv_res (convolutional residual layers) trains without error.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT,
                                     extra_args=['--conv_res', '1'])
        assert os.path.exists(args['save_name'])

    def test_save_load_roundtrip(self, tmp_path):
        """
        Train a model, load it back via Trainer, and verify that the config
        and vocab from the checkpoint are intact.

        We pass args=None to Trainer so the loaded config comes entirely
        from the checkpoint — that is the meaningful thing to verify.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT)
        save_name = args['save_name']
        assert os.path.exists(save_name)

        loaded_trainer = Trainer(
            model_file=save_name,
            args=None,
            device='cpu',
            foundation_cache=None,
        )

        assert loaded_trainer.args['hidden_dim'] == 16
        assert loaded_trainer.args['emb_dim'] == 8
        assert loaded_trainer.vocab is not None

    def test_save_load_resave(self, tmp_path):
        """
        Train, load, then save again — verify the resaved file is a valid
        checkpoint.  Catches cases where loading drops state needed for
        re-serialization (e.g. lexicon, vocab).
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT)
        save_name = args['save_name']

        loaded_trainer = Trainer(
            model_file=save_name,
            args=None,
            device='cpu',
            foundation_cache=None,
        )

        resave_path = str(tmp_path / "resaved_tokenizer.pt")
        loaded_trainer.save(resave_path)
        assert os.path.exists(resave_path)

        checkpoint = torch.load(resave_path, lambda storage, loc: storage, weights_only=True)
        assert 'model' in checkpoint
        assert 'vocab' in checkpoint
        assert 'config' in checkpoint

    def test_charlm(self, tmp_path, english_charlm_forward_file):
        """
        Train with a real pretrained forward charlm.  Verifies the charlm
        path through DataLoader and Trainer doesn't crash, that charlm
        weights are stripped from the saved checkpoint (it is an
        unsaved_module), and that the model reloads correctly with the
        charlm path restored from args.
        """
        trainer, args = run_training(tmp_path, TRAIN_SENTENCES_NO_MWT, extra_args=[
            '--charlm',
            '--charlm_forward_file', english_charlm_forward_file,
        ])
        save_name = args['save_name']
        assert os.path.exists(save_name)

        checkpoint = torch.load(save_name, lambda storage, loc: storage, weights_only=True)
        assert not any(k.startswith('charlm') for k in checkpoint['model'].keys()), \
            "charlm weights should not be serialized into the model checkpoint"

        loaded_trainer = Trainer(
            model_file=save_name,
            args={'charlm_forward_file': english_charlm_forward_file},
            device='cpu',
            foundation_cache=None,
        )
        assert loaded_trainer.args.get('charlm_forward_file') == english_charlm_forward_file
