"""
Unit tests for MaxoutLinear.

Run with:
    pytest test_maxout_linear.py -v

MaxoutLinear(in_channels, out_channels, maxout_k) implements maxout:
  - A single Linear maps in_channels -> out_channels * maxout_k
  - The output is reshaped to (..., maxout_k, out_channels)
  - The max is taken over the maxout_k dimension
  - Final output shape: (..., out_channels)

With maxout_k=1 the layer degenerates to a plain Linear (the max over one
candidate is a no-op), so MaxoutLinear(in, out, 1) should be equivalent to
nn.Linear(in, out) given the same underlying weights.
"""

import pytest
import torch
import torch.nn as nn

from stanza.models.common.maxout_linear import MaxoutLinear

pytestmark = [pytest.mark.travis]

# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_2d_input(self):
        """Standard batch input (batch, in_channels) -> (batch, out_channels)."""
        layer = MaxoutLinear(in_channels=8, out_channels=4, maxout_k=3)
        x = torch.randn(5, 8)
        y = layer(x)
        assert y.shape == (5, 4)

    def test_3d_input(self):
        """Sequence input (batch, seq, in_channels) -> (batch, seq, out_channels)."""
        layer = MaxoutLinear(in_channels=8, out_channels=4, maxout_k=3)
        x = torch.randn(5, 7, 8)
        y = layer(x)
        assert y.shape == (5, 7, 4)

    def test_1d_input(self):
        """Single vector (in_channels,) -> (out_channels,)."""
        layer = MaxoutLinear(in_channels=8, out_channels=4, maxout_k=3)
        x = torch.randn(8)
        y = layer(x)
        assert y.shape == (4,)

    def test_k1_shape(self):
        """maxout_k=1 still produces the correct output shape."""
        layer = MaxoutLinear(in_channels=8, out_channels=4, maxout_k=1)
        x = torch.randn(5, 8)
        y = layer(x)
        assert y.shape == (5, 4)

    @pytest.mark.parametrize("in_c,out_c,k", [
        (16, 8, 2),
        (32, 16, 5),
        (1, 1, 4),
    ])
    def test_various_sizes(self, in_c, out_c, k):
        layer = MaxoutLinear(in_channels=in_c, out_channels=out_c, maxout_k=k)
        x = torch.randn(3, in_c)
        y = layer(x)
        assert y.shape == (3, out_c)


# ---------------------------------------------------------------------------
# Correctness: the max is actually taken
# ---------------------------------------------------------------------------

class TestMaxIsActuallyTaken:
    def test_output_equals_manual_max(self):
        """
        Manually replicate the maxout computation and confirm it matches.

        The underlying linear has shape (out_channels * maxout_k, in_channels).
        We reshape its output to (batch, maxout_k, out_channels) and take the
        max over dim -2.
        """
        in_c, out_c, k = 6, 4, 3
        batch = 5
        layer = MaxoutLinear(in_channels=in_c, out_channels=out_c, maxout_k=k)

        x = torch.randn(batch, in_c)
        y = layer(x)

        # replicate manually
        raw = layer.linear(x)                           # (batch, out_c * k)
        raw_reshaped = raw.view(batch, k, out_c)        # (batch, k, out_c)
        expected = raw_reshaped.max(dim=-2).values      # (batch, out_c)

        assert torch.allclose(y, expected)

    def test_max_not_identity(self):
        """
        With k>1 the output should differ from a slice of the raw linear output,
        confirming the max is doing something rather than just returning one candidate.

        We construct weights so that different candidates win for different output
        neurons, making it clear the max is not simply selecting one fixed candidate.
        """
        in_c, out_c, k = 4, 2, 3
        layer = MaxoutLinear(in_channels=in_c, out_channels=out_c, maxout_k=k)

        # Zero out all weights then set specific candidates to known values
        # so we know exactly which candidate should win for each output neuron.
        with torch.no_grad():
            layer.linear.weight.zero_()
            layer.linear.bias.zero_()
            # candidate 0 wins for output neuron 0
            layer.linear.bias[0] = 10.0
            # candidate 1 wins for output neuron 1
            # linear weight layout: (out_c * k, in_c), ordered as
            # [cand0_out0, cand0_out1, cand1_out0, cand1_out1, ...]
            layer.linear.bias[out_c * 1 + 1] = 5.0

        x = torch.zeros(1, in_c)
        y = layer(x)

        # output neuron 0 should be 10.0 (from candidate 0)
        assert y[0, 0].item() == pytest.approx(10.0)
        # output neuron 1 should be 5.0 (from candidate 1)
        assert y[0, 1].item() == pytest.approx(5.0)

    def test_k1_equivalent_to_linear(self):
        """
        With maxout_k=1 the layer is mathematically identical to nn.Linear,
        since the max over a single candidate is a no-op.
        """
        in_c, out_c = 6, 4
        maxout = MaxoutLinear(in_channels=in_c, out_channels=out_c, maxout_k=1)
        linear = nn.Linear(in_features=in_c, out_features=out_c)

        # share the same weights
        with torch.no_grad():
            linear.weight.copy_(maxout.linear.weight)
            linear.bias.copy_(maxout.linear.bias)

        x = torch.randn(5, in_c)
        assert torch.allclose(maxout(x), linear(x))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_flow_through(self):
        """Loss.backward() should produce non-None, non-zero gradients."""
        layer = MaxoutLinear(in_channels=8, out_channels=4, maxout_k=3)
        x = torch.randn(5, 8)
        loss = layer(x).sum()
        loss.backward()

        assert layer.linear.weight.grad is not None
        assert layer.linear.bias.grad is not None
        assert layer.linear.weight.grad.abs().sum().item() > 0
