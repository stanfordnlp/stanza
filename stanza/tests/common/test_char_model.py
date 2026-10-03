"""
Currently tests a few configurations of files for creating a charlm vocab

Also has a skeleton test of loading & saving a charlm,
and tests of the text preprocessing schemes such as BHO
"""

from collections import Counter
import glob
import logging
import lzma
import os
import tempfile

import pytest
import torch

from stanza.models import charlm
from stanza.models.common import char_model
from stanza.models.common.char_model import Preprocessing
from stanza.models.common.vocab import CharVocab
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

fake_text_1 = """
Unban mox opal!
I hate watching Peppa Pig
"""

fake_text_2 = """
This is plastic cheese
"""

class TestCharModel:
    def test_single_file_vocab(self):
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "text.txt")
            with open(sample_file, "w", encoding="utf-8") as fout:
                fout.write(fake_text_1)
            vocab = char_model.build_charlm_vocab(sample_file)

        for i in fake_text_1:
            assert i in vocab
        assert "Q" not in vocab

    def test_single_file_xz_vocab(self):
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "text.txt.xz")
            with lzma.open(sample_file, "wt", encoding="utf-8") as fout:
                fout.write(fake_text_1)
            vocab = char_model.build_charlm_vocab(sample_file)

        for i in fake_text_1:
            assert i in vocab
        assert "Q" not in vocab

    def test_single_file_dir_vocab(self):
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "text.txt")
            with open(sample_file, "w", encoding="utf-8") as fout:
                fout.write(fake_text_1)
            vocab = char_model.build_charlm_vocab(tempdir)

        for i in fake_text_1:
            assert i in vocab
        assert "Q" not in vocab

    def test_multiple_files_vocab(self):
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "t1.txt")
            with open(sample_file, "w", encoding="utf-8") as fout:
                fout.write(fake_text_1)
            sample_file = os.path.join(tempdir, "t2.txt.xz")
            with lzma.open(sample_file, "wt", encoding="utf-8") as fout:
                fout.write(fake_text_2)
            vocab = char_model.build_charlm_vocab(tempdir)

        for i in fake_text_1:
            assert i in vocab
        for i in fake_text_2:
            assert i in vocab
        assert "Q" not in vocab

    def test_cutoff_vocab(self):
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "t1.txt")
            with open(sample_file, "w", encoding="utf-8") as fout:
                fout.write(fake_text_1)
            sample_file = os.path.join(tempdir, "t2.txt.xz")
            with lzma.open(sample_file, "wt", encoding="utf-8") as fout:
                fout.write(fake_text_2)

            vocab = char_model.build_charlm_vocab(tempdir, cutoff=2)

        counts = Counter(fake_text_1) + Counter(fake_text_2)
        for letter, count in counts.most_common():
            if count < 2:
                assert letter not in vocab
            else:
                assert letter in vocab

    def test_build_model(self):
        """
        Test the whole thing on a small dataset for an iteration or two
        """
        with tempfile.TemporaryDirectory() as tempdir:
            eval_file = os.path.join(tempdir, "en_test.dev.txt")
            with open(eval_file, "w", encoding="utf-8") as fout:
                fout.write(fake_text_1)
            train_file = os.path.join(tempdir, "en_test.train.txt")
            with open(train_file, "w", encoding="utf-8") as fout:
                for i in range(1000):
                    fout.write(fake_text_1)
                    fout.write("\n")
                    fout.write(fake_text_2)
                    fout.write("\n")
            save_name = 'en_test.forward.pt'
            vocab_save_name = 'en_text.vocab.pt'
            checkpoint_save_name = 'en_text.checkpoint.pt'
            args = ['--train_file', train_file,
                    '--eval_file', eval_file,
                    '--eval_steps', '0', # eval once per opoch
                    '--epochs', '2',
                    '--cutoff', '1',
                    '--batch_size', '%d' % len(fake_text_1),
                    '--shorthand', 'en_test',
                    '--save_dir', tempdir,
                    '--save_name', save_name,
                    '--vocab_save_name', vocab_save_name,
                    '--checkpoint_save_name', checkpoint_save_name]
            args = charlm.parse_args(args)
            charlm.train(args)

            assert os.path.exists(os.path.join(tempdir, vocab_save_name))

            # test that saving & loading of the model worked
            assert os.path.exists(os.path.join(tempdir, save_name))
            model = char_model.CharacterLanguageModel.load(os.path.join(tempdir, save_name))

            # test that saving & loading of the checkpoint worked
            assert os.path.exists(os.path.join(tempdir, checkpoint_save_name))
            model = char_model.CharacterLanguageModel.load(os.path.join(tempdir, checkpoint_save_name))
            trainer = char_model.CharacterLanguageModelTrainer.load(args, os.path.join(tempdir, checkpoint_save_name))

            assert trainer.global_step > 0
            assert trainer.epoch == 2

            # quick test to verify this method works with a trained model
            charlm.get_current_lr(trainer, args)

            # test loading a vocab built by the training method...
            vocab = charlm.load_char_vocab(os.path.join(tempdir, vocab_save_name))
            trainer = char_model.CharacterLanguageModelTrainer.from_new_model(args, vocab)
            # ... and test the get_current_lr for an untrained model as well
            # this test is super "eager"
            assert charlm.get_current_lr(trainer, args) == args['lr0']

    @pytest.fixture(scope="class")
    def english_forward(self):
        # eg, stanza_test/models/en/forward_charlm/1billion.pt
        models_path = os.path.join(TEST_MODELS_DIR, "en", "forward_charlm", "*")
        models = glob.glob(models_path)
        # we expect at least one English model downloaded for the tests
        assert len(models) >= 1
        model_file = models[0]
        return char_model.CharacterLanguageModel.load(model_file)

    @pytest.fixture(scope="class")
    def english_backward(self):
        # eg, stanza_test/models/en/forward_charlm/1billion.pt
        models_path = os.path.join(TEST_MODELS_DIR, "en", "backward_charlm", "*")
        models = glob.glob(models_path)
        # we expect at least one English model downloaded for the tests
        assert len(models) >= 1
        model_file = models[0]
        return char_model.CharacterLanguageModel.load(model_file)

    def test_load_model(self, english_forward, english_backward):
        """
        Check that basic loading functions work
        """
        assert english_forward.is_forward_lm
        assert not english_backward.is_forward_lm

    def test_save_load_model(self, english_forward, english_backward):
        """
        Load, save, and load again
        """
        with tempfile.TemporaryDirectory() as tempdir:
            for model in (english_forward, english_backward):
                save_file = os.path.join(tempdir, "resaved", "charlm.pt")
                model.save(save_file)
                reloaded = char_model.CharacterLanguageModel.load(save_file)
                assert model.is_forward_lm == reloaded.is_forward_lm

CANDRABINDU = "\N{DEVANAGARI SIGN CANDRABINDU}"
ANUSVARA = "\N{DEVANAGARI SIGN ANUSVARA}"
# U+F06C is a Private Use Area glyph found in some crawled Bhojpuri text
PUA_GLYPH = "\uF06C"

# Bhojpuri-ish text which uses both nasal marks and has the stray PUA glyph
bho_text = ("कँ कं खँ खं क%sं ख\n" % PUA_GLYPH) * 20

def build_tiny_model(preprocessing, chars="कखं", forward=True):
    """
    An untrained charlm small enough to run on CPU instantly

    The vocab is built by hand so the tests decide exactly which characters
    have embeddings.  The default vocab has anusvara but not candrabindu or
    the PUA glyph, which is what a BHO vocab should look like.
    """
    args = {
        'char_emb_dim': 8,
        'char_hidden_dim': 16,
        'char_num_layers': 1,
        'char_dropout': 0.0,
        'char_unit_dropout': 0.0,
        'char_rec_dropout': 0.0,
        'preprocessing': preprocessing,
    }
    vocab = {'char': CharVocab([sorted(set(chars + char_model.CHARLM_START + char_model.CHARLM_END))])}
    model = char_model.CharacterLanguageModel(args, vocab, is_forward_lm=forward)
    model.eval()
    return model

def assert_same_reps(rep1, rep2):
    assert len(rep1) == len(rep2)
    for x, y in zip(rep1, rep2):
        assert x.shape == y.shape
        assert torch.allclose(x, y)

class TestPreprocessing:
    def test_enum_lookup(self):
        """
        Lookup by value accepts any case and is idempotent on members
        """
        assert Preprocessing("bho") is Preprocessing.BHO
        assert Preprocessing("BHO") is Preprocessing.BHO
        assert Preprocessing(Preprocessing.BHO) is Preprocessing.BHO
        with pytest.raises(ValueError):
            Preprocessing("xyz")

    def test_preprocess_bho(self):
        assert Preprocessing.NONE.preprocess("क" + CANDRABINDU) == "क" + CANDRABINDU
        assert Preprocessing.BHO.preprocess("क" + CANDRABINDU) == "क" + ANUSVARA
        assert Preprocessing.BHO.preprocess("क" + PUA_GLYPH + ANUSVARA) == "क" + ANUSVARA
        # a word made up entirely of removed characters is left alone
        assert Preprocessing.BHO.preprocess(PUA_GLYPH) == PUA_GLYPH
        assert Preprocessing.BHO.preprocess("") == ""

    def test_preprocess_preserve_length(self):
        """
        Per-character consumers need a transform which does not change lengths

        Normalization still applies, but nothing is removed
        """
        text = "क" + PUA_GLYPH + CANDRABINDU
        result = Preprocessing.BHO.preprocess(text, preserve_length=True)
        assert result == "क" + PUA_GLYPH + ANUSVARA
        assert Preprocessing.NONE.preprocess(text, preserve_length=True) == text

    def test_preprocess_allow_empty(self):
        """
        With allow_empty, text made up entirely of removed characters becomes empty

        Bulk text such as vocab counting needs this, while words keep the guard
        """
        assert Preprocessing.BHO.preprocess(PUA_GLYPH * 5, allow_empty=True) == ""
        assert Preprocessing.BHO.preprocess(PUA_GLYPH * 5) == PUA_GLYPH * 5
        assert Preprocessing.BHO.preprocess("क" + PUA_GLYPH, allow_empty=True) == "क"
        assert Preprocessing.NONE.preprocess(PUA_GLYPH, allow_empty=True) == PUA_GLYPH

    def test_str(self):
        """
        str() gives the value, so help text and log lines read cleanly
        """
        assert str(Preprocessing.BHO) == "bho"
        assert str(Preprocessing.NONE) == "none"

    def test_argparse(self):
        args = charlm.parse_args([])
        assert args['preprocessing'] is Preprocessing.NONE
        args = charlm.parse_args(['--preprocessing', 'bho'])
        assert args['preprocessing'] is Preprocessing.BHO
        args = charlm.parse_args(['--preprocessing', 'BHO'])
        assert args['preprocessing'] is Preprocessing.BHO
        with pytest.raises(SystemExit):
            charlm.parse_args(['--preprocessing', 'xyz'])

    def test_argparse_help(self):
        """
        The choices and the default show up as the strings the user types
        """
        help_text = charlm.build_argparse().format_help()
        assert "{none,bho}" in help_text
        assert "Preprocessing." not in help_text

    def test_full_state_saves_value(self):
        """
        The saved config holds the same string the command line accepts
        """
        model = build_tiny_model(Preprocessing.BHO)
        state = model.full_state()
        assert state['args']['preprocessing'] == "bho"
        # saving does not alter the live model's config
        assert model.args['preprocessing'] is Preprocessing.BHO

    @pytest.mark.parametrize("preprocessing", list(Preprocessing))
    def test_save_load_round_trip(self, preprocessing):
        model = build_tiny_model(preprocessing)
        with tempfile.TemporaryDirectory() as tempdir:
            save_file = os.path.join(tempdir, "charlm.pt")
            model.save(save_file)
            reloaded = char_model.CharacterLanguageModel.load(save_file)
        assert reloaded.preprocessing is preprocessing

    def test_load_without_preprocessing(self):
        """
        Models saved before preprocessing existed load as NONE
        """
        model = build_tiny_model(Preprocessing.NONE)
        state = model.full_state()
        del state['args']['preprocessing']
        reloaded = char_model.CharacterLanguageModel.from_full_state(state)
        assert reloaded.preprocessing is Preprocessing.NONE

    def test_load_saved_name(self):
        """
        Model files which stored the member name rather than the value still load
        """
        model = build_tiny_model(Preprocessing.BHO)
        state = model.full_state()
        state['args']['preprocessing'] = "BHO"
        reloaded = char_model.CharacterLanguageModel.from_full_state(state)
        assert reloaded.preprocessing is Preprocessing.BHO

    @pytest.mark.parametrize("forward", [True, False])
    def test_build_char_representation_control(self, forward):
        """
        Without preprocessing, the two nasal marks give different representations

        This is what makes the BHO tests below meaningful
        """
        model = build_tiny_model(Preprocessing.NONE, forward=forward)
        rep1 = model.build_char_representation([["क" + CANDRABINDU, "ख"]])
        rep2 = model.build_char_representation([["क" + ANUSVARA, "ख"]])
        assert not torch.allclose(rep1[0], rep2[0])

    @pytest.mark.parametrize("forward", [True, False])
    def test_build_char_representation_normalizes(self, forward):
        """
        Inference applies the same preprocessing as training
        """
        model = build_tiny_model(Preprocessing.BHO, forward=forward)
        rep1 = model.build_char_representation([["क" + CANDRABINDU, "ख"]])
        rep2 = model.build_char_representation([["क" + ANUSVARA, "ख"]])
        assert_same_reps(rep1, rep2)

    @pytest.mark.parametrize("forward", [True, False])
    def test_build_char_representation_removes(self, forward):
        """
        Removed characters vanish, and the word offsets still line up
        """
        model = build_tiny_model(Preprocessing.BHO, forward=forward)
        rep1 = model.build_char_representation([["क" + PUA_GLYPH + ANUSVARA, "ख"], ["ख", PUA_GLYPH + "क"]])
        rep2 = model.build_char_representation([["क" + ANUSVARA, "ख"], ["ख", "क"]])
        assert_same_reps(rep1, rep2)

    @pytest.mark.parametrize("forward", [True, False])
    def test_build_char_representation_empty_word(self, forward):
        """
        A word made up entirely of the removed glyph still gets a row
        """
        model = build_tiny_model(Preprocessing.BHO, forward=forward)
        rep = model.build_char_representation([[PUA_GLYPH, "क"], ["ख", PUA_GLYPH, PUA_GLYPH]])
        assert [x.shape[0] for x in rep] == [2, 3]

    def test_per_char_representation_normalizes(self):
        """
        Per-character output applies the length-preserving normalization
        """
        model = build_tiny_model(Preprocessing.BHO)
        rep1 = model.per_char_representation(["क" + CANDRABINDU, "ख" + CANDRABINDU + "क"])
        rep2 = model.per_char_representation(["क" + ANUSVARA, "ख" + ANUSVARA + "क"])
        assert_same_reps(rep1, rep2)

    def test_per_char_representation_preserves_length(self):
        """
        One vector per input character, even for characters BHO removes
        """
        model = build_tiny_model(Preprocessing.BHO)
        words = ["क" + PUA_GLYPH + ANUSVARA, PUA_GLYPH, "ख"]
        rep = model.per_char_representation(words)
        assert [x.shape[0] for x in rep] == [len(x) for x in words]

    def test_word_adapter_shape(self):
        """
        The adapter output lines up with the original words, START and END included
        """
        model = build_tiny_model(Preprocessing.BHO)
        adapter = char_model.CharacterLanguageModelWordAdapter([model])
        words = ["क" + PUA_GLYPH + CANDRABINDU, PUA_GLYPH]
        rep = adapter(words)
        assert rep.shape == (2, 3 + 2, model.hidden_dim())

    @pytest.mark.parametrize("preprocessing", list(Preprocessing))
    def test_vocab_preprocessing(self, preprocessing):
        """
        The vocab is counted on preprocessed text

        With a cutoff of 2, anusvara only clears the cutoff once the
        candrabindu are merged into it
        """
        text = "क" * 5 + ANUSVARA + CANDRABINDU + PUA_GLYPH * 5
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "text.txt")
            with open(sample_file, "w", encoding="utf-8") as fout:
                fout.write(text)
            vocab = char_model.build_charlm_vocab(sample_file, cutoff=2, preprocessing=preprocessing)

        assert "क" in vocab
        if preprocessing is Preprocessing.BHO:
            assert ANUSVARA in vocab
            assert CANDRABINDU not in vocab
            assert PUA_GLYPH not in vocab
        else:
            assert ANUSVARA not in vocab
            assert CANDRABINDU not in vocab
            assert PUA_GLYPH in vocab

    def test_vocab_removed_char_only_line(self):
        """
        A line which is nothing but the removed glyph does not sneak it into the vocab

        Lines usually end in a newline, which survives preprocessing, so the
        case which matters is a final line with no trailing newline
        """
        with tempfile.TemporaryDirectory() as tempdir:
            sample_file = os.path.join(tempdir, "text.txt")
            with open(sample_file, "w", encoding="utf-8") as fout:
                fout.write("कं\n")
                fout.write(PUA_GLYPH * 5)
            vocab = char_model.build_charlm_vocab(sample_file, cutoff=1, preprocessing=Preprocessing.BHO)

        assert "क" in vocab
        assert PUA_GLYPH not in vocab

    def test_vocab_removed_char_only_file(self):
        """
        A file in a training directory which is nothing but the removed glyph does not add it to the vocab
        """
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "t1.txt"), "w", encoding="utf-8") as fout:
                fout.write("कं\n")
            with open(os.path.join(tempdir, "t2.txt"), "w", encoding="utf-8") as fout:
                fout.write(PUA_GLYPH * 5)
            vocab = char_model.build_charlm_vocab(tempdir, cutoff=1, preprocessing=Preprocessing.BHO)

        assert "क" in vocab
        assert PUA_GLYPH not in vocab

    def train_tiny_charlm(self, tempdir, extra_args=()):
        eval_file = os.path.join(tempdir, "bho_test.dev.txt")
        with open(eval_file, "w", encoding="utf-8") as fout:
            fout.write(bho_text)
        train_file = os.path.join(tempdir, "bho_test.train.txt")
        with open(train_file, "w", encoding="utf-8") as fout:
            fout.write(bho_text * 5)
        args = ['--train_file', train_file,
                '--eval_file', eval_file,
                '--eval_steps', '0',
                '--epochs', '1',
                '--cutoff', '1',
                '--batch_size', '4',
                '--bptt_size', '20',
                '--char_emb_dim', '8',
                '--char_hidden_dim', '16',
                '--shorthand', 'bho_test',
                '--save_dir', tempdir,
                '--save_name', 'bho_test.forward.pt',
                '--vocab_save_name', 'bho_test.vocab.pt',
                '--no_checkpoint',
                '--preprocessing', 'bho'] + list(extra_args)
        args = charlm.parse_args(args)
        charlm.train(args)
        return args

    def test_train_bho(self):
        """
        Training with BHO builds a BHO vocab and saves the setting in the model file
        """
        with tempfile.TemporaryDirectory() as tempdir:
            self.train_tiny_charlm(tempdir)
            vocab = charlm.load_char_vocab(os.path.join(tempdir, 'bho_test.vocab.pt'))
            model = char_model.CharacterLanguageModel.load(os.path.join(tempdir, 'bho_test.forward.pt'))
            state = torch.load(os.path.join(tempdir, 'bho_test.forward.pt'), lambda storage, loc: storage, weights_only=True)

        assert ANUSVARA in vocab['char']
        assert CANDRABINDU not in vocab['char']
        assert PUA_GLYPH not in vocab['char']
        assert model.preprocessing is Preprocessing.BHO
        assert state['model']['args']['preprocessing'] == "bho"

    def test_train_stale_vocab_warns(self, caplog):
        """
        Reusing a vocab with characters the preprocessing removes is flagged
        """
        with tempfile.TemporaryDirectory() as tempdir:
            stale_vocab = CharVocab([sorted(set(bho_text))])
            assert CANDRABINDU in stale_vocab
            torch.save(stale_vocab.state_dict(), os.path.join(tempdir, 'bho_test.vocab.pt'))
            with caplog.at_level(logging.WARNING, logger='stanza'):
                self.train_tiny_charlm(tempdir)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("vocab" in r.getMessage().lower() for r in warnings)

    def test_evaluate_bho(self):
        """
        Evaluating a trained BHO model applies its preprocessing to the eval file
        """
        with tempfile.TemporaryDirectory() as tempdir:
            args = self.train_tiny_charlm(tempdir)
            args['mode'] = 'predict'
            charlm.evaluate(args)
