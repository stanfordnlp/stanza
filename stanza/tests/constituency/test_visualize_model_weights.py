"""
Unit tests for stanza/utils/constituency/visualize_model_weights.py

Tests are grouped into four levels:

  1. State-dict level: extraction and statistics functions work correctly
     on hand-built synthetic dicts.  No model construction needed.
     These tests have no external dependencies and run in milliseconds.

  2. Model level: extraction and statistics work on a state dict produced
     by a real (untrained) LSTMModel, built via the same build_trainer
     path used by the rest of the constituency test suite.
     Requires tiny_emb.pt from TEST_WORKING_DIR.

  3. Rendering level: render_weight_image writes a valid PNG to a temp
     path without raising.  Image content is not checked.

  4. Forget gate summary: print_forget_gate_summary produces a correctly
     formatted table, handles collapsed LSTMs gracefully, and covers the
     multi-LSTM and missing-checkpoint cases.

  5. --no_plots flag: run_stats_mode with no_plots=True produces no PNG
     files; without the flag, PNG files are written as expected.

  6. Degeneracy detection: _is_degenerate correctly identifies collapsed
     LSTMs (all weights driven to zero by optimizer starvation), and
     _draw_degenerate_panel renders without error.
"""

import logging
import os
import tempfile

import numpy as np
import pytest
import torch

from stanza.models import constituency_parser
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.constituency import lstm_model, parser_training, tree_reader
from stanza.tests import *
from stanza.utils.constituency.visualize_model_weights import (
    GATE_NAMES,
    compute_gate_stats,
    compute_linear_stats,
    get_linear,
    get_lstm_gate_weights,
    render_weight_image,
    print_forget_gate_summary,
    _is_degenerate,
    _draw_degenerate_panel,
)

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

logger = logging.getLogger('stanza.tests.constituency.test_visualize_model_weights')

# ---------------------------------------------------------------------------
# Minimal treebank — same as test_trainer.py
# ---------------------------------------------------------------------------

TREEBANK = """
( (S
    (VP (VBG Enjoying)
      (NP (PRP$  my) (JJ favorite) (NN Friday) (NN tradition)))
    (. .)))

( (NP
    (VP (VBG Sitting)
      (PP (IN in)
        (NP (DT a) (RB stifling) (JJ hot) (NNP South) (NNP Station)))
      (VP (VBG waiting)
        (PP (IN for)
          (NP (PRP$  my) (JJ delayed) (NNP @MBTA) (NN train)))))
    (. .)))

( (S
    (NP (PRP I))
    (VP
      (ADVP (RB really))
      (VBP hate)
      (NP (DT the) (NNP @MBTA)))))

( (S
    (S (VP (VB Seek)))
    (CC and)
    (S (NP (PRP ye))
      (VP (MD shall)
        (VP (VB find))))
    (. .)))
"""


def build_trainer(wordvec_pretrain_file, *extra_args, treebank=TREEBANK):
    """Build an untrained model via the standard training path."""
    train_trees = tree_reader.read_trees(treebank)
    dev_trees   = train_trees[-1:]
    silver_trees = []
    args = constituency_parser.parse_args(
        ['--device', 'cpu', '--wordvec_pretrain_file', wordvec_pretrain_file] + list(extra_args)
    )
    foundation_cache = FoundationCache()
    model_load_name  = args['load_name']
    tr, _, _, _ = parser_training.build_trainer(
        args, train_trees, dev_trees, silver_trees, foundation_cache, model_load_name
    )
    assert isinstance(tr.model, lstm_model.LSTMModel)
    return tr


# ---------------------------------------------------------------------------
# Helpers: synthetic state dicts (no model needed)
# ---------------------------------------------------------------------------

def _make_linear_state_dict(out_features, in_features, name='reduce_linear', with_bias=True):
    sd = {f'{name}.weight': torch.randn(out_features, in_features)}
    if with_bias:
        sd[f'{name}.bias'] = torch.randn(out_features)
    return sd


def _make_lstm_state_dict(prefix, input_size, hidden_size, num_layers=1, bidirectional=False):
    """
    Synthetic LSTM state dict.
    PyTorch gate order: input(i), forget(f), cell(g), output(o) → [4*H, ...].
    """
    sd = {}
    directions = [''] + (['_reverse'] if bidirectional else [])
    for layer in range(num_layers):
        in_sz = input_size if layer == 0 else hidden_size * len(directions)
        for suffix in directions:
            sd[f'{prefix}.weight_ih_l{layer}{suffix}'] = torch.randn(4 * hidden_size, in_sz)
            sd[f'{prefix}.weight_hh_l{layer}{suffix}'] = torch.randn(4 * hidden_size, hidden_size)
            sd[f'{prefix}.bias_ih_l{layer}{suffix}']   = torch.randn(4 * hidden_size)
            sd[f'{prefix}.bias_hh_l{layer}{suffix}']   = torch.randn(4 * hidden_size)
    return sd


# ---------------------------------------------------------------------------
# 1. State-dict level: get_linear
# ---------------------------------------------------------------------------

class TestGetLinear:
    def test_returns_weight_and_bias(self):
        sd = _make_linear_state_dict(8, 16)
        w, b = get_linear(sd, 'reduce_linear')
        assert w is not None and b is not None
        assert w.shape == (8, 16)
        assert b.shape == (8,)

    def test_returns_none_for_missing_name(self):
        sd = _make_linear_state_dict(8, 16)
        w, b = get_linear(sd, 'nonexistent')
        assert w is None and b is None

    def test_no_bias(self):
        sd = _make_linear_state_dict(8, 16, with_bias=False)
        w, b = get_linear(sd, 'reduce_linear')
        assert w is not None
        assert b is None

    def test_non_square_matrix(self):
        """BILSTM reduce_linear is [H, 2H] — not square."""
        sd = _make_linear_state_dict(64, 128)
        w, _ = get_linear(sd, 'reduce_linear')
        assert w.shape == (64, 128)

    def test_nested_name(self):
        """output_layers.0 style dotted prefix."""
        sd = _make_linear_state_dict(32, 64, name='output_layers.0')
        w, b = get_linear(sd, 'output_layers.0')
        assert w.shape == (32, 64)

    def test_returns_float32(self):
        sd = {'reduce_linear.weight': torch.randn(4, 4, dtype=torch.float16),
              'reduce_linear.bias':   torch.randn(4,     dtype=torch.float16)}
        w, b = get_linear(sd, 'reduce_linear')
        assert w.dtype == np.float32
        assert b.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. State-dict level: get_lstm_gate_weights
# ---------------------------------------------------------------------------

class TestGetLSTMGateWeights:
    def test_single_layer_unidirectional(self):
        hidden, inp = 16, 8
        sd = _make_lstm_state_dict('word_lstm', inp, hidden)
        gd = get_lstm_gate_weights(sd, 'word_lstm')
        assert len(gd) == 1
        entry = gd[0]
        assert entry['layer']     == 0
        assert entry['direction'] == 'forward'
        for gate in GATE_NAMES:
            assert entry['weight_ih'][gate].shape == (hidden, inp)
            assert entry['weight_hh'][gate].shape == (hidden, hidden)
            assert entry['bias'][gate].shape      == (hidden,)

    def test_bidirectional(self):
        sd = _make_lstm_state_dict('word_lstm', 8, 16, bidirectional=True)
        gd = get_lstm_gate_weights(sd, 'word_lstm')
        assert len(gd) == 2
        assert {e['direction'] for e in gd.values()} == {'forward', 'reverse'}

    def test_multi_layer(self):
        sd = _make_lstm_state_dict('word_lstm', 8, 16, num_layers=3)
        gd = get_lstm_gate_weights(sd, 'word_lstm')
        assert len(gd) == 3
        assert sorted(e['layer'] for e in gd.values()) == [0, 1, 2]

    def test_multi_layer_bidirectional(self):
        sd = _make_lstm_state_dict('word_lstm', 8, 16, num_layers=2, bidirectional=True)
        gd = get_lstm_gate_weights(sd, 'word_lstm')
        assert len(gd) == 4   # 2 layers × 2 directions

    def test_missing_prefix_returns_empty(self):
        sd = _make_lstm_state_dict('word_lstm', 8, 16)
        assert get_lstm_gate_weights(sd, 'nonexistent') == {}

    def test_gate_order(self):
        """Verify the i/f/g/o slice ordering against known arange values."""
        hidden = 4
        bih = torch.arange(4 * hidden, dtype=torch.float32)
        sd = {
            'lstm.weight_ih_l0': torch.zeros(4 * hidden, 4),
            'lstm.weight_hh_l0': torch.zeros(4 * hidden, hidden),
            'lstm.bias_ih_l0':   bih,
            'lstm.bias_hh_l0':   torch.zeros(4 * hidden),
        }
        gd = get_lstm_gate_weights(sd, 'lstm')
        np.testing.assert_array_equal(gd[0]['bias']['input'],  bih[0:4].numpy())
        np.testing.assert_array_equal(gd[0]['bias']['forget'], bih[4:8].numpy())
        np.testing.assert_array_equal(gd[0]['bias']['cell'],   bih[8:12].numpy())
        np.testing.assert_array_equal(gd[0]['bias']['output'], bih[12:16].numpy())

    def test_bias_is_sum_of_ih_and_hh(self):
        hidden = 8
        bih = torch.randn(4 * hidden)
        bhh = torch.randn(4 * hidden)
        sd = {
            'lstm.weight_ih_l0': torch.randn(4 * hidden, 4),
            'lstm.weight_hh_l0': torch.randn(4 * hidden, hidden),
            'lstm.bias_ih_l0':   bih,
            'lstm.bias_hh_l0':   bhh,
        }
        gd = get_lstm_gate_weights(sd, 'lstm')
        expected = (bih + bhh)[hidden:2*hidden].numpy()
        np.testing.assert_allclose(gd[0]['bias']['forget'], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. State-dict level: compute_linear_stats
# ---------------------------------------------------------------------------

class TestComputeLinearStats:
    def _stats(self, shape=(16, 16)):
        sd = _make_linear_state_dict(*shape)
        w, b = get_linear(sd, 'reduce_linear')
        return compute_linear_stats(w, b)

    def test_required_keys_present(self):
        s = self._stats()
        for key in ('weight_norm', 'weight_mean', 'weight_std',
                    'weight_abs_max', 'spectral_radius',
                    'bias_norm', 'bias_mean', 'bias_std'):
            assert key in s, f"Missing key: {key}"

    def test_no_bias_keys_absent(self):
        sd = _make_linear_state_dict(8, 8, with_bias=False)
        w, b = get_linear(sd, 'reduce_linear')
        s = compute_linear_stats(w, b)
        assert 'bias_mean' not in s
        assert 'bias_norm' not in s

    def test_zero_matrix(self):
        w = np.zeros((8, 8), dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        s = compute_linear_stats(w, b)
        assert s['weight_norm']     == pytest.approx(0.0)
        assert s['spectral_radius'] == pytest.approx(0.0)
        assert s['bias_mean']       == pytest.approx(0.0)

    def test_spectral_radius_identity(self):
        """Spectral radius of the identity matrix is 1.0."""
        s = compute_linear_stats(np.eye(32, dtype=np.float32), None)
        assert s['spectral_radius'] == pytest.approx(1.0, abs=1e-4)

    def test_all_values_finite(self):
        for v in self._stats((64, 64)).values():
            assert np.isfinite(v)


# ---------------------------------------------------------------------------
# 4. State-dict level: compute_gate_stats
# ---------------------------------------------------------------------------

class TestComputeGateStats:
    def _gate_stats(self, hidden=16, inp=8, num_layers=1, bidirectional=False):
        sd = _make_lstm_state_dict('word_lstm', inp, hidden,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional)
        return compute_gate_stats(get_lstm_gate_weights(sd, 'word_lstm'))

    def test_required_gate_stat_keys(self):
        gs = self._gate_stats()
        for gate in GATE_NAMES:
            assert gate in gs[0]
            for stat in ('bias_mean', 'bias_std', 'bias_median',
                         'saturation_pct', 'weight_ih_norm',
                         'weight_hh_norm', 'spectral_radius'):
                assert stat in gs[0][gate], f"Missing {stat} for gate {gate}"

    def test_metadata_keys(self):
        gs = self._gate_stats()
        assert 'layer'     in gs[0]
        assert 'direction' in gs[0]

    def test_saturation_pct_range(self):
        gs = self._gate_stats()
        for gate in GATE_NAMES:
            assert 0.0 <= gs[0][gate]['saturation_pct'] <= 100.0

    def test_zero_bias_no_saturation(self):
        hidden = 8
        sd = {
            'lstm.weight_ih_l0': torch.randn(4 * hidden, 4),
            'lstm.weight_hh_l0': torch.randn(4 * hidden, hidden),
            'lstm.bias_ih_l0':   torch.zeros(4 * hidden),
            'lstm.bias_hh_l0':   torch.zeros(4 * hidden),
        }
        gs = compute_gate_stats(get_lstm_gate_weights(sd, 'lstm'))
        for gate in GATE_NAMES:
            assert gs[0][gate]['saturation_pct'] == pytest.approx(0.0)

    def test_large_bias_full_saturation(self):
        hidden = 8
        sd = {
            'lstm.weight_ih_l0': torch.zeros(4 * hidden, 4),
            'lstm.weight_hh_l0': torch.zeros(4 * hidden, hidden),
            'lstm.bias_ih_l0':   torch.full((4 * hidden,), 10.0),
            'lstm.bias_hh_l0':   torch.zeros(4 * hidden),
        }
        gs = compute_gate_stats(get_lstm_gate_weights(sd, 'lstm'))
        for gate in GATE_NAMES:
            assert gs[0][gate]['saturation_pct'] == pytest.approx(100.0)

    def test_all_stats_finite(self):
        gs = self._gate_stats(hidden=32, inp=16, num_layers=2, bidirectional=True)
        for entry in gs.values():
            for gate in GATE_NAMES:
                for v in entry[gate].values():
                    assert np.isfinite(v)

    def test_spectral_radius_identity_gate(self):
        hidden = 16
        eye4 = torch.eye(hidden).repeat(4, 1)   # [4H, H] — each gate slice is I
        sd = {
            'lstm.weight_ih_l0': torch.randn(4 * hidden, 4),
            'lstm.weight_hh_l0': eye4,
            'lstm.bias_ih_l0':   torch.zeros(4 * hidden),
            'lstm.bias_hh_l0':   torch.zeros(4 * hidden),
        }
        gs = compute_gate_stats(get_lstm_gate_weights(sd, 'lstm'))
        for gate in GATE_NAMES:
            assert gs[0][gate]['spectral_radius'] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Model-level tests: state dict from a real untrained LSTMModel
# ---------------------------------------------------------------------------

class TestModelStateDict:
    @pytest.fixture(scope='class')
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    @pytest.fixture(scope='class')
    def state_dict(self, wordvec_pretrain_file):
        return build_trainer(wordvec_pretrain_file).model.state_dict()

    def test_reduce_linear_extractable(self, state_dict):
        w, b = get_linear(state_dict, 'reduce_linear')
        assert w is not None
        # MAX composition → square weight
        assert w.shape[0] == w.shape[1]

    def test_word_lstm_extractable(self, state_dict):
        gd = get_lstm_gate_weights(state_dict, 'word_lstm')
        assert len(gd) > 0
        # word_lstm is bidirectional → at least 2 entries
        assert len(gd) >= 2

    def test_all_four_gates_present_in_model(self, state_dict):
        gd = get_lstm_gate_weights(state_dict, 'word_lstm')
        for entry in gd.values():
            for gate in GATE_NAMES:
                assert gate in entry['weight_ih']
                assert gate in entry['weight_hh']
                assert gate in entry['bias']

    def test_compute_linear_stats_from_model(self, state_dict):
        w, b = get_linear(state_dict, 'reduce_linear')
        s = compute_linear_stats(w, b)
        assert s['weight_norm'] > 0
        assert np.isfinite(s['spectral_radius'])

    def test_compute_gate_stats_from_model(self, state_dict):
        gd = get_lstm_gate_weights(state_dict, 'word_lstm')
        gs = compute_gate_stats(gd)
        for entry in gs.values():
            for gate in GATE_NAMES:
                assert np.isfinite(entry[gate]['spectral_radius'])


# ---------------------------------------------------------------------------
# 6. Rendering: images are written without error
# ---------------------------------------------------------------------------

class TestRenderWeightImage:
    def _w(self, rows=32, cols=32):
        return np.random.randn(rows, cols).astype(np.float32)

    def _b(self, size=32):
        return np.random.randn(size).astype(np.float32)

    def test_writes_file_with_bias(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.png')
            render_weight_image(self._w(), self._b(), 'test', path)
            assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_writes_file_without_bias(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.png')
            render_weight_image(self._w(), None, 'test', path)
            assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_non_square_weight(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.png')
            render_weight_image(self._w(32, 64), self._b(32), 'test', path)
            assert os.path.exists(path)

    def test_large_weight(self):
        """512×512 as produced by a real model should not OOM or crash."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.png')
            render_weight_image(self._w(512, 512), self._b(512), 'test', path)
            assert os.path.exists(path)

    def test_all_zeros_weight(self):
        """Dead matrix should still render without error."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'out.png')
            render_weight_image(np.zeros((32, 32), dtype=np.float32), None, 'test', path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 7. Forget gate summary table
# ---------------------------------------------------------------------------

class TestForgetGateSummary:
    def _make_gate_stats(self, forget_bias_mean, hidden=8, inp=4, collapsed=False):
        """Build a minimal gate_stats dict with a controlled forget gate bias mean."""
        if collapsed:
            # simulate a fully collapsed LSTM — all norms zero
            sd = {
                'lstm.weight_ih_l0': torch.zeros(4 * hidden, inp),
                'lstm.weight_hh_l0': torch.zeros(4 * hidden, hidden),
                'lstm.bias_ih_l0':   torch.zeros(4 * hidden),
                'lstm.bias_hh_l0':   torch.zeros(4 * hidden),
            }
        else:
            bih = torch.zeros(4 * hidden)
            bih[hidden:2*hidden] = forget_bias_mean  # forget gate slice
            sd = {
                'lstm.weight_ih_l0': torch.randn(4 * hidden, inp),
                'lstm.weight_hh_l0': torch.randn(4 * hidden, hidden),
                'lstm.bias_ih_l0':   bih,
                'lstm.bias_hh_l0':   torch.zeros(4 * hidden),
            }
        return compute_gate_stats(get_lstm_gate_weights(sd, 'lstm'))

    def test_prints_without_error(self, capsys):
        """Basic smoke test: table prints without raising."""
        gs1 = self._make_gate_stats(1.0)
        gs2 = self._make_gate_stats(0.5)
        all_lstm_stats = {'word_lstm': [gs1, gs2]}
        print_forget_gate_summary(['ckpt_0', 'ckpt_1'], ['word_lstm'], all_lstm_stats)
        out = capsys.readouterr().out
        assert 'word_lstm' in out
        assert 'ckpt_0' in out

    def test_forget_bias_values_appear(self, capsys):
        """The forget gate mean for each checkpoint should appear in the output."""
        gs1 = self._make_gate_stats(1.0)
        gs2 = self._make_gate_stats(0.5)
        all_lstm_stats = {'lstm': [gs1, gs2]}
        print_forget_gate_summary(['epoch_1', 'epoch_2'], ['lstm'], all_lstm_stats)
        out = capsys.readouterr().out
        assert '+1.000' in out
        assert '+0.500' in out

    def test_collapsed_lstm_shown_as_collapsed(self, capsys):
        """A collapsed LSTM entry should show COLLAPSED rather than a number."""
        gs_healthy   = self._make_gate_stats(1.0)
        gs_collapsed = self._make_gate_stats(0.0, collapsed=True)
        all_lstm_stats = {'word_lstm': [gs_healthy, gs_collapsed]}
        print_forget_gate_summary(['ckpt_0', 'ckpt_1'], ['word_lstm'], all_lstm_stats)
        out = capsys.readouterr().out
        assert 'COLLAPSED' in out
        assert '+1.000' in out

    def test_multiple_lstms(self, capsys):
        """All LSTM prefixes should appear as columns."""
        gs = self._make_gate_stats(0.75)
        all_lstm_stats = {
            'word_lstm':             [gs],
            'constituent_stack.lstm': [gs],
        }
        print_forget_gate_summary(['ckpt_0'], ['word_lstm', 'constituent_stack.lstm'],
                                  all_lstm_stats)
        out = capsys.readouterr().out
        assert 'word_lstm' in out
        assert 'constituent_stack.lstm' in out

    def test_empty_stats_shown_as_na(self, capsys):
        """A checkpoint where an LSTM wasn't found should show N/A."""
        gs = self._make_gate_stats(1.0)
        all_lstm_stats = {'word_lstm': [gs, {}]}  # second checkpoint missing
        print_forget_gate_summary(['ckpt_0', 'ckpt_1'], ['word_lstm'], all_lstm_stats)
        out = capsys.readouterr().out
        assert 'N/A' in out


# ---------------------------------------------------------------------------
# 8. --no_plots flag
# ---------------------------------------------------------------------------

class TestNoPlotsFlag:
    """
    Verify that run_stats_mode with no_plots=True produces terminal output
    but writes no PNG files to the output directory.
    """

    @pytest.fixture(scope='class')
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    @pytest.fixture(scope='class')
    def trainer(self, wordvec_pretrain_file):
        return build_trainer(wordvec_pretrain_file)

    def test_no_plots_writes_no_files(self, trainer):
        from stanza.utils.constituency.visualize_model_weights import run_stats_mode
        state = trainer.model.state_dict()

        # Write a temporary checkpoint that load_checkpoint can read
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, 'model.pt')
            out_dir   = os.path.join(d, 'out')
            os.makedirs(out_dir)

            # Wrap in the structure load_checkpoint expects
            torch.save({'params': {'model': state}}, ckpt_path,
                       _use_new_zipfile_serialization=False)

            run_stats_mode(
                checkpoints=[ckpt_path],
                labels=['test'],
                out_dir=out_dir,
                lstm_prefixes=['word_lstm'],
                linear_names=['reduce_linear'],
                no_plots=True,
            )

            png_files = [f for f in os.listdir(out_dir) if f.endswith('.png')]
            assert png_files == [], f"Expected no PNGs with no_plots=True, got: {png_files}"

    def test_plots_written_without_flag(self, trainer):
        from stanza.utils.constituency.visualize_model_weights import run_stats_mode
        state = trainer.model.state_dict()

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, 'model.pt')
            out_dir   = os.path.join(d, 'out')
            os.makedirs(out_dir)

            torch.save({'params': {'model': state}}, ckpt_path,
                       _use_new_zipfile_serialization=False)

            run_stats_mode(
                checkpoints=[ckpt_path],
                labels=['test'],
                out_dir=out_dir,
                lstm_prefixes=['word_lstm'],
                linear_names=['reduce_linear'],
                no_plots=False,
            )

            png_files = [f for f in os.listdir(out_dir) if f.endswith('.png')]
            assert len(png_files) > 0, "Expected PNG files with no_plots=False"


# ---------------------------------------------------------------------------
# 9. Degeneracy detection
# ---------------------------------------------------------------------------

class TestDegeneracyDetection:
    def test_zero_array_is_degenerate(self):
        assert _is_degenerate(np.zeros(64, dtype=np.float32))

    def test_underflow_values_are_degenerate(self):
        # float32 underflow artifacts as seen with collapsed AdaDelta models
        arr = np.full(64, 1e-44, dtype=np.float32)
        assert _is_degenerate(arr)

    def test_normal_weights_not_degenerate(self):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(64).astype(np.float32)
        assert not _is_degenerate(arr)

    def test_small_but_real_weights_not_degenerate(self):
        # 0.01 is small but well above the 1e-30 threshold
        arr = np.full(64, 0.01, dtype=np.float32)
        assert not _is_degenerate(arr)

    def test_mixed_array_not_degenerate(self):
        # A single non-zero value should be enough to avoid the degenerate path
        arr = np.zeros(64, dtype=np.float32)
        arr[32] = 0.05
        assert not _is_degenerate(arr)

    def test_degenerate_panel_renders(self):
        """_draw_degenerate_panel should produce a file without raising."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _draw_degenerate_panel(ax, 'transition_stack.lstm', 'forget')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'degenerate.png')
            fig.savefig(path)
            assert os.path.exists(path) and os.path.getsize(path) > 0
        plt.close(fig)
