"""
test_condense_output_layers.py

Tests for condense_output_layers.py:
  - find_live_rows detection (with bias)
  - condense_model() file I/O
  - Correct weight/bias surgery on layer i and column surgery on layer i+1
  - Idempotency
  - End-to-end: condensed forward == original forward
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

from stanza.utils.constituency.condense_output_layers import (
    find_live_rows,
    condense_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 16      # hidden_size
N_TRANS = 7 # num_transitions (final layer output)
N_IN = 24   # first layer input size (word + transition + constituent)


def _linear(in_size, out_size, dead_rows=()):
    layer = nn.Linear(in_size, out_size)
    with torch.no_grad():
        nn.init.normal_(layer.weight, std=0.2)
        nn.init.normal_(layer.bias,   std=0.1)
        for r in dead_rows:
            layer.weight[r, :] = 0.0
            layer.bias[r]      = 0.0
    return layer


def _synthetic_checkpoint(dead_rows_per_layer=((),), n=N, n_in=N_IN, n_trans=N_TRANS):
    """
    Build a minimal checkpoint with num_output_layers = len(dead_rows_per_layer) + 1.
    dead_rows_per_layer is a sequence of dead-row specs for each middle layer.
    """
    num_output_layers = len(dead_rows_per_layer) + 1
    layers = []
    in_size = n_in
    for i, dead in enumerate(dead_rows_per_layer):
        layers.append(_linear(in_size, n, dead))
        in_size = n
    # final layer
    layers.append(_linear(in_size, n_trans))

    state = {}
    for i, layer in enumerate(layers):
        state[f'output_layers.{i}.weight'] = layer.weight.data.clone()
        state[f'output_layers.{i}.bias']   = layer.bias.data.clone()

    args = {
        'hidden_size': n,
        'num_output_layers': num_output_layers,
        'constituency_composition': 'MAX',
    }
    return {
        'params': {'config': args, 'model': state},
        'model_type': 'LSTM',
        'epochs_trained': 0, 'batches_trained': 0,
        'best_f1': 0.0, 'best_epoch': 0,
    }, layers


def _run_condense(dead_rows_per_layer=((),), threshold=0.005, dry_run=False):
    cp, layers = _synthetic_checkpoint(dead_rows_per_layer)
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = os.path.join(tmpdir, 'model.pt')
        out = os.path.join(tmpdir, 'condensed.pt')
        torch.save(cp, inp)
        summary = condense_model(inp, out, threshold=threshold, dry_run=dry_run)
        loaded = (torch.load(out, map_location='cpu', weights_only=True)
                  if (not dry_run and os.path.exists(out)) else None)
    return summary, loaded, layers


# ---------------------------------------------------------------------------
# find_live_rows
# ---------------------------------------------------------------------------

class TestFindLiveRows:

    def test_all_live(self):
        w = torch.eye(N)
        b = torch.randn(N) * 0.1
        assert find_live_rows(w, b, 0.001).all()

    def test_dead_row_zero_weight_zero_bias(self):
        w = torch.randn(N, N) * 0.2
        b = torch.randn(N) * 0.1
        w[3, :] = 0.0
        b[3]    = 0.0
        mask = find_live_rows(w, b, 0.001)
        assert not mask[3]
        for i in range(N):
            if i != 3:
                assert mask[i]

    def test_nonzero_bias_keeps_row_alive(self):
        """A row with near-zero weight but significant bias is NOT dead."""
        w = torch.randn(N, N) * 0.2
        b = torch.randn(N) * 0.1
        w[5, :] = 0.0
        b[5]    = 0.5   # significant bias: row stays alive
        mask = find_live_rows(w, b, 0.005)
        assert mask[5]

    def test_near_zero_bias_also_needed(self):
        """Near-zero weight AND near-zero bias -> dead."""
        w = torch.randn(N, N) * 0.2
        b = torch.randn(N) * 0.1
        w[7, :] = 1e-6
        b[7]    = 1e-6
        mask = find_live_rows(w, b, 0.005)
        assert not mask[7]

    def test_all_zero_guard(self):
        """All-zero matrix triggers the guard and returns all-live."""
        w = torch.zeros(N, N)
        b = torch.zeros(N)
        assert find_live_rows(w, b, 0.005).all()


# ---------------------------------------------------------------------------
# condense_model: file I/O
# ---------------------------------------------------------------------------

class TestCondenseModel:

    def test_no_dead_neurons(self):
        summary, _, _ = _run_condense(dead_rows_per_layer=((),))
        assert summary['results'][0]['dead'] == 0

    def test_dead_neurons_detected(self):
        summary, loaded, _ = _run_condense(dead_rows_per_layer=((2, 7, 11),))
        r = summary['results'][0]
        assert r['dead'] == 3
        assert r['live'] == N - 3

    def test_layer0_weight_condensed(self):
        dead = (1, 5)
        summary, loaded, _ = _run_condense(dead_rows_per_layer=(dead,))
        w = loaded['params']['model']['output_layers.0.weight']
        assert w.shape == (N - len(dead), N_IN)

    def test_layer0_bias_condensed(self):
        dead = (1, 5)
        _, loaded, _ = _run_condense(dead_rows_per_layer=(dead,))
        b = loaded['params']['model']['output_layers.0.bias']
        assert b.shape == (N - len(dead),)

    def test_layer1_input_cols_condensed(self):
        """Layer 1's input columns must shrink to match layer 0's live outputs."""
        dead = (0, 3)
        _, loaded, _ = _run_condense(dead_rows_per_layer=(dead,))
        w1 = loaded['params']['model']['output_layers.1.weight']
        assert w1.shape == (N_TRANS, N - len(dead))

    def test_layer1_bias_unchanged(self):
        """Layer 1's bias must not be touched."""
        dead = (4,)
        _, loaded, original_layers = _run_condense(dead_rows_per_layer=(dead,))
        b1_orig = original_layers[1].bias.data
        b1_cond = loaded['params']['model']['output_layers.1.bias']
        torch.testing.assert_close(b1_cond, b1_orig)

    def test_output_layer_sizes_in_args(self):
        dead = (2, 9)
        _, loaded, _ = _run_condense(dead_rows_per_layer=(dead,))
        sizes = loaded['params']['config']['output_layer_sizes']
        assert sizes == [N - len(dead)]

    def test_correct_rows_kept(self):
        """The surviving rows should be exactly the live ones."""
        dead = (0, 6)
        _, loaded, original_layers = _run_condense(dead_rows_per_layer=(dead,))
        live_idx = [i for i in range(N) if i not in dead]
        w_orig = original_layers[0].weight.data
        w_cond = loaded['params']['model']['output_layers.0.weight']
        torch.testing.assert_close(w_cond, w_orig[live_idx])

    def test_correct_cols_kept_in_next_layer(self):
        """The surviving input columns of layer 1 should be the live rows of layer 0."""
        dead = (1, 8)
        _, loaded, original_layers = _run_condense(dead_rows_per_layer=(dead,))
        live_idx = [i for i in range(N) if i not in dead]
        w1_orig = original_layers[1].weight.data
        w1_cond = loaded['params']['model']['output_layers.1.weight']
        torch.testing.assert_close(w1_cond, w1_orig[:, live_idx])

    def test_dry_run_no_output(self):
        summary, loaded, _ = _run_condense(dead_rows_per_layer=((0,),), dry_run=True)
        assert loaded is None

    def test_wrong_model_type_raises(self):
        cp, _ = _synthetic_checkpoint()
        cp['model_type'] = 'WRONG'
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'm.pt'); out = os.path.join(tmpdir, 'o.pt')
            torch.save(cp, inp)
            with pytest.raises(ValueError):
                condense_model(inp, out)

    def test_single_layer_no_op(self):
        """num_output_layers=1 has no middle layers; should return cleanly with empty results."""
        cp, _ = _synthetic_checkpoint()
        cp['params']['config']['num_output_layers'] = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'm.pt'); out = os.path.join(tmpdir, 'o.pt')
            torch.save(cp, inp)
            summary = condense_model(inp, out)
        assert summary['middle_layers'] == 0
        assert summary['results'] == []

    def test_idempotent(self):
        """Running twice on an already-condensed checkpoint finds no further dead neurons."""
        _, condensed, _ = _run_condense(dead_rows_per_layer=((3, 7),))
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'condensed.pt')
            out = os.path.join(tmpdir, 'condensed2.pt')
            torch.save(condensed, inp)
            summary2 = condense_model(inp, out)
        # After first condensation, no further dead rows should be found
        assert all(r['dead'] == 0 for r in summary2['results'])

    def test_further_condensation_after_training_specified_sizes(self):
        """
        If output_layer_sizes is already in args (e.g. user specified at training time),
        condensation still runs on the current weights and can condense further.
        """
        # Build a checkpoint that already has output_layer_sizes set but still has dead rows
        cp, layers = _synthetic_checkpoint(dead_rows_per_layer=((2, 5),))
        # Pretend the user had already specified sizes at training time
        cp['params']['config']['output_layer_sizes'] = [N]  # N = full size, not yet condensed
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'm.pt'); out = os.path.join(tmpdir, 'o.pt')
            torch.save(cp, inp)
            summary = condense_model(inp, out)
            loaded = torch.load(out, map_location='cpu', weights_only=True)
        # Should have detected the 2 dead rows and condensed
        assert summary['results'][0]['dead'] == 2
        assert loaded['params']['config']['output_layer_sizes'] == [N - 2]

    def test_all_dead_layer_skipped(self):
        """A layer where all rows are dead should be left at its current size with a warning."""
        cp, layers = _synthetic_checkpoint(dead_rows_per_layer=(tuple(range(N)),))
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'm.pt'); out = os.path.join(tmpdir, 'o.pt')
            torch.save(cp, inp)
            summary = condense_model(inp, out)
        # All-dead layer should be reported as 0 dead (skipped) to avoid 0-width matrix
        assert summary['results'][0]['dead'] == 0
        assert summary['results'][0]['live'] == N

    def test_three_layer_model(self):
        """A 3-layer model condenses two middle layers independently."""
        dead0, dead1 = (2, 7), (3,)
        cp, layers = _synthetic_checkpoint(dead_rows_per_layer=(dead0, dead1))
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'm.pt'); out = os.path.join(tmpdir, 'o.pt')
            torch.save(cp, inp)
            summary = condense_model(inp, out)
            loaded = torch.load(out, map_location='cpu', weights_only=True)

        assert loaded['params']['config']['output_layer_sizes'] == [N - len(dead0), N - len(dead1)]
        w0 = loaded['params']['model']['output_layers.0.weight']
        w1 = loaded['params']['model']['output_layers.1.weight']
        w2 = loaded['params']['model']['output_layers.2.weight']
        assert w0.shape == (N - len(dead0), N_IN)
        assert w1.shape == (N - len(dead1), N - len(dead0))
        assert w2.shape == (N_TRANS, N - len(dead1))


# ---------------------------------------------------------------------------
# End-to-end: condensed forward == original forward
# ---------------------------------------------------------------------------

class TestEndToEndEquivalence:
    """
    Verify that running forward through the condensed layers gives the same
    result as the original, for random input.

    This mirrors what lstm_model.py would do after loading the condensed
    checkpoint: build_output_layers uses output_layer_sizes from args to
    construct layers of the right shape, then load_state_dict fills weights.
    """

    def _forward(self, layers, x, nonlinearity=torch.relu):
        hx = x
        for i, layer in enumerate(layers):
            hx = nonlinearity(hx)
            hx = layer(hx)
        return hx

    def _pipeline(self, dead_rows_per_layer, n_in=N_IN):
        torch.manual_seed(0)
        cp, original_layers = _synthetic_checkpoint(dead_rows_per_layer, n_in=n_in)
        # Copy exact weights into checkpoint
        for i, layer in enumerate(original_layers):
            cp['params']['model'][f'output_layers.{i}.weight'] = layer.weight.data.clone()
            cp['params']['model'][f'output_layers.{i}.bias']   = layer.bias.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, 'm.pt'); out = os.path.join(tmpdir, 'o.pt')
            torch.save(cp, inp)
            condense_model(inp, out, threshold=0.005)
            loaded = torch.load(out, map_location='cpu', weights_only=True)

        state = loaded['params']['model']
        args  = loaded['params']['config']

        # Build condensed layers as build_output_layers would, using output_layer_sizes
        output_layer_sizes = args.get('output_layer_sizes', [N] * (len(original_layers) - 1))
        sizes_in  = [n_in]  + output_layer_sizes
        sizes_out = output_layer_sizes + [N_TRANS]
        condensed_layers = []
        for i, (in_s, out_s) in enumerate(zip(sizes_in, sizes_out)):
            layer = nn.Linear(in_s, out_s)
            layer.weight.data.copy_(state[f'output_layers.{i}.weight'])
            layer.bias.data.copy_(state[f'output_layers.{i}.bias'])
            condensed_layers.append(layer)

        x = torch.randn(5, n_in)
        orig_out = self._forward(original_layers, x)
        cond_out = self._forward(condensed_layers, x)
        return orig_out, cond_out

    def test_two_layer_dead_middle(self):
        orig, cond = self._pipeline(dead_rows_per_layer=((1, 5, 12),))
        torch.testing.assert_close(orig, cond, atol=1e-5, rtol=0)

    def test_two_layer_no_dead(self):
        orig, cond = self._pipeline(dead_rows_per_layer=((),))
        torch.testing.assert_close(orig, cond, atol=1e-5, rtol=0)

    def test_three_layer(self):
        orig, cond = self._pipeline(dead_rows_per_layer=((2, 9), (4,)))
        torch.testing.assert_close(orig, cond, atol=1e-5, rtol=0)
