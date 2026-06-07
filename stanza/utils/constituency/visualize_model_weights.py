"""
Visualization tool for constituency parser linear weights and LSTM gate statistics.

Two modes of operation:

1. Weight image mode (--mode image):
   Renders each named linear layer's weight matrix and bias vector as a heatmap
   image, one PNG per (linear, checkpoint) pair.  Useful for watching how weight
   structure evolves across training.

2. Statistics mode (--mode stats):
   Plots distributions and time-series of:
     - Per-linear weight/bias norms, std, spectral radius, and value histograms
     - All four LSTM gate statistics (input, forget, cell, output) for each
       named LSTM covering: bias mean/std/saturation, weight_ih norm,
       weight_hh norm, and weight_hh spectral radius
   This mode requires at least one checkpoint; passing several gives a
   time-series view of how the statistics evolve.
   Use --no_plots to skip PNG output and print terminal summaries only.

Usage examples:

  # Single checkpoint - weight images for reduce_linear (default)
  python visualize_model_weights.py --mode image --checkpoints model_best.pt

  # Multiple linears and checkpoints, stats evolution
  python visualize_model_weights.py --mode stats \\
      --checkpoints epoch_001.pt epoch_010.pt epoch_050.pt epoch_200.pt \\
      --labels "epoch 1" "epoch 10" "epoch 50" "epoch 200" \\
      --linears reduce_linear word_to_constituent output_layers.0

  # All three LSTMs, both modes, custom linears
  python visualize_model_weights.py --mode both \\
      --checkpoints epoch_001.pt epoch_200.pt \\
      --lstms word_lstm transition_stack.lstm constituent_stack.lstm \\
      --linears reduce_linear word_to_constituent

  # Stats only, no PNG output — terminal summaries only
  python visualize_model_weights.py --mode stats --no_plots \\
      --checkpoints epoch_001.pt epoch_010.pt epoch_200.pt \\
      --lstms word_lstm transition_stack.lstm constituent_stack.lstm

  # Discover what linear and LSTM keys are available in a checkpoint
  python visualize_model_weights.py --list_keys --checkpoints model_best.pt

Dependencies: torch, matplotlib, numpy (all already in the Stanza environment)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless - write files rather than display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    """
    Load a raw checkpoint dict without constructing the full model.
    The model state dict lives at checkpoint['params']['model'].
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return checkpoint


def extract_model_state(checkpoint):
    return checkpoint['params']['model']


# ---------------------------------------------------------------------------
# Linear layer extraction
# ---------------------------------------------------------------------------

def get_linear(state_dict, name):
    """
    Extract weight and bias for any named nn.Linear from a state dict.
    Returns (weight, bias) as float32 numpy arrays, or (None, None) if absent.

    `name` is the dotted state-dict prefix, e.g. 'reduce_linear',
    'word_to_constituent', 'output_layers.0'.
    """
    w = state_dict.get(f'{name}.weight', None)
    b = state_dict.get(f'{name}.bias',   None)
    if w is None:
        return None, None
    return w.float().numpy(), (b.float().numpy() if b is not None else None)


def list_keys(state_dict):
    """
    Print a summary of all linear and LSTM parameter prefixes found in the
    state dict, to help the user discover valid --linears / --lstms values.
    """
    linear_prefixes = set()
    lstm_prefixes   = set()
    for key in state_dict:
        if key.endswith('.weight') and state_dict[key].dim() == 2:
            prefix = key[:-len('.weight')]
            # LSTM weights have the form prefix.weight_ih_lN[_reverse]
            # so strip those before recording
            if '.weight_ih_l' not in key and '.weight_hh_l' not in key:
                linear_prefixes.add(prefix)
        if '.weight_ih_l0' in key:
            # recover the module prefix before .weight_ih_l0
            prefix = key[:key.index('.weight_ih_l0')]
            lstm_prefixes.add(prefix)

    print("\nLinear layers (valid --linears values):")
    for p in sorted(linear_prefixes):
        w = state_dict[f'{p}.weight']
        b = state_dict.get(f'{p}.bias')
        bias_str = f'bias [{b.shape[0]}]' if b is not None else 'no bias'
        print(f"  {p:50s}  weight {list(w.shape)}  {bias_str}")

    print("\nLSTM modules (valid --lstms values):")
    for p in sorted(lstm_prefixes):
        h = state_dict[f'{p}.weight_ih_l0'].shape[0] // 4
        bidir = f'{p}.weight_ih_l0_reverse' in state_dict
        print(f"  {p:50s}  hidden={h}  {'bidirectional' if bidir else 'unidirectional'}")
    print()


# ---------------------------------------------------------------------------
# LSTM gate extraction
# ---------------------------------------------------------------------------

# PyTorch packs LSTM weights as:
#   weight_ih_lN: [4*H, input]    gates in order: input(i), forget(f), cell(g), output(o)
#   weight_hh_lN: [4*H, H]
#   bias_ih_lN:   [4*H]
#   bias_hh_lN:   [4*H]
# For bidirectional LSTMs the reverse direction uses suffix '_reverse'.
GATE_NAMES  = ['input', 'forget', 'cell', 'output']

# Gate label convention follows PyTorch's internal naming:
#   i=input, f=forget, g=cell candidate (not 'c', which PyTorch reserves
#   for the cell state c_t = f ⊙ c_{t-1} + i ⊙ g itself), o=output
# Many papers and tutorials use 'c' or 'c̃' for the candidate cell values
# instead (e.g. Olah's "Understanding LSTMs"), so the 'g' here can be
# surprising — but it matches what you'll see in PyTorch's own docs.
GATE_LABELS = {'input': 'i', 'forget': 'f', 'cell': 'g', 'output': 'o'}

# Reference lines drawn on bias plots per gate
# forget: recommended positive init; input/output: neutral at 0; cell: neutral at 0
GATE_BIAS_REFS = {
    'input':  [(0.0, 'red',   'neutral (0)')],
    'forget': [(0.0, 'red',   'neutral (0)'),
               (1.0, 'green', 'recommended init (+1)')],
    'cell':   [(0.0, 'red',   'neutral (0)')],
    'output': [(0.0, 'red',   'neutral (0)')],
}


def _split_gates(tensor, hidden_size):
    """Split a [4*H, ...] tensor into a dict of gate_name -> slice."""
    return {
        name: tensor[i * hidden_size:(i + 1) * hidden_size].float().numpy()
        for i, name in enumerate(GATE_NAMES)
    }


def get_lstm_gate_weights(state_dict, lstm_prefix):
    """
    Extract per-gate weight slices for every layer (and direction) of the
    LSTM at `lstm_prefix`.

    Returns:
      { layer_idx: {
            'weight_ih': { gate_name: ndarray },   # [H, input_size]
            'weight_hh': { gate_name: ndarray },   # [H, H]
            'bias':      { gate_name: ndarray },   # [H], sum of ih + hh biases
            'direction': 'forward' | 'reverse',
        }
      }

    Bidirectional LSTMs produce two entries per layer index
    (keyed as layer*2 and layer*2+1 for forward and reverse).
    """
    result = {}
    entry_idx = 0
    layer = 0
    while True:
        for suffix, direction in [('', 'forward'), ('_reverse', 'reverse')]:
            key = f'{lstm_prefix}.weight_ih_l{layer}{suffix}'
            if key not in state_dict:
                continue
            hidden_size = state_dict[key].shape[0] // 4

            wih = state_dict[f'{lstm_prefix}.weight_ih_l{layer}{suffix}'].float()
            whh = state_dict[f'{lstm_prefix}.weight_hh_l{layer}{suffix}'].float()
            bih = state_dict[f'{lstm_prefix}.bias_ih_l{layer}{suffix}'].float()
            bhh = state_dict[f'{lstm_prefix}.bias_hh_l{layer}{suffix}'].float()

            result[entry_idx] = {
                'weight_ih': _split_gates(wih, hidden_size),
                'weight_hh': _split_gates(whh, hidden_size),
                'bias':      {name: (bih + bhh)[i * hidden_size:(i + 1) * hidden_size].float().numpy()
                              for i, name in enumerate(GATE_NAMES)},
                'direction': direction,
                'layer':     layer,
            }
            entry_idx += 1

        # stop when neither forward nor reverse exists for this layer
        if f'{lstm_prefix}.weight_ih_l{layer}' not in state_dict and \
           f'{lstm_prefix}.weight_ih_l{layer}_reverse' not in state_dict:
            break
        layer += 1

    return result


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def spectral_radius(matrix):
    """
    Approximate the spectral radius (largest singular value) of a 2-D array.

    Uses the top singular value via a randomised SVD (power iteration),
    which is fast enough for H x H matrices up to ~2048.
    For very small H we fall back to full SVD.
    """
    if matrix.shape[0] <= 256:
        s = np.linalg.svd(matrix, compute_uv=False)
        return float(s[0])
    # randomised: single power iteration with a random probe
    rng = np.random.default_rng(0)
    n = matrix.shape[1]
    v = rng.standard_normal(n).astype(np.float32)
    for _ in range(4):
        v = matrix.T @ (matrix @ v)
        norm = np.linalg.norm(v)
        if norm == 0:
            return 0.0
        v /= norm
    return float(np.sqrt(norm))


def compute_linear_stats(weight, bias):
    """Scalar statistics for any nn.Linear weight and bias."""
    stats = {
        'weight_norm':    float(np.linalg.norm(weight)),
        'weight_mean':    float(weight.mean()),
        'weight_std':     float(weight.std()),
        'weight_abs_max': float(np.abs(weight).max()),
        'spectral_radius': spectral_radius(weight),
    }
    if bias is not None:
        stats.update({
            'bias_norm': float(np.linalg.norm(bias)),
            'bias_mean': float(bias.mean()),
            'bias_std':  float(bias.std()),
        })
    return stats


def compute_gate_stats(gate_data):
    """
    Compute per-gate statistics for all layers/directions of one LSTM.

    gate_data: output of get_lstm_gate_weights()

    For each entry and gate:
      bias_mean, bias_std, bias_median  — distribution of combined bias
      saturation_pct  — % of bias units where |b| > 2 (sigmoid in saturation zone)
      weight_ih_norm  — Frobenius norm of the input-hidden slice
      weight_hh_norm  — Frobenius norm of the hidden-hidden slice
      spectral_radius — largest singular value of weight_hh slice

    Returns { entry_idx: { gate_name: { stat: value }, 'layer': int, 'direction': str } }
    """
    result = {}
    for entry_idx, data in gate_data.items():
        entry_stats = {
            'layer':     data['layer'],
            'direction': data['direction'],
        }
        for gate in GATE_NAMES:
            b   = data['bias'][gate]
            wih = data['weight_ih'][gate]
            whh = data['weight_hh'][gate]
            entry_stats[gate] = {
                'bias_mean':       float(b.mean()),
                'bias_std':        float(b.std()),
                'bias_median':     float(np.median(b)),
                'saturation_pct':  float(100.0 * (np.abs(b) > 2.0).mean()),
                'weight_ih_norm':  float(np.linalg.norm(wih)),
                'weight_hh_norm':  float(np.linalg.norm(whh)),
                'spectral_radius': spectral_radius(whh),
            }
        result[entry_idx] = entry_stats
    return result


# ---------------------------------------------------------------------------
# Mode 1: Weight image
# ---------------------------------------------------------------------------

def render_weight_image(weight, bias, title, out_path):
    """
    Render weight [out, in] as a heatmap and bias [out] as a column to its right.
    Diverging colormap (RdBu_r) so sign is immediately visible.

    Layout: the bias column is given a fixed fraction of the total figure width
    (BIAS_WIDTH_FRAC) so it stays legible regardless of the weight matrix shape.
    Each panel gets its own colorbar placed below it to avoid overlap.
    """
    has_bias = bias is not None

    # Bias column gets ~10% of total width; weight gets the rest.
    # Without bias the figure is slightly narrower.
    BIAS_WIDTH_FRAC = 0.10
    FIG_W = 12.0
    FIG_H = 7.0

    if has_bias:
        fig, (ax_w, ax_b) = plt.subplots(
            1, 2,
            figsize=(FIG_W, FIG_H),
            gridspec_kw={'width_ratios': [1 - BIAS_WIDTH_FRAC, BIAS_WIDTH_FRAC],
                         'wspace': 0.35}
        )
    else:
        fig, ax_w = plt.subplots(1, 1, figsize=(FIG_W * 0.9, FIG_H))

    vmax = max(abs(weight.min()), abs(weight.max())) or 1.0
    im = ax_w.imshow(weight, aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax_w.set_title(f'Weight  [{weight.shape[0]} × {weight.shape[1]}]')
    ax_w.set_xlabel('Input dimension')
    ax_w.set_ylabel('Output dimension')
    # colorbar below the weight axes so it doesn't crowd the bias panel
    plt.colorbar(im, ax=ax_w, orientation='horizontal', fraction=0.046, pad=0.08)

    if has_bias:
        bmax = max(abs(bias.min()), abs(bias.max())) or 1.0
        im_b = ax_b.imshow(bias.reshape(-1, 1), aspect='auto', cmap='RdBu_r',
                           vmin=-bmax, vmax=bmax, interpolation='nearest')
        ax_b.set_title('Bias')
        ax_b.set_xticks([])
        ax_b.set_ylabel('Output dimension')
        # colorbar below the bias axes, matching the weight colorbar orientation
        cb_b = plt.colorbar(im_b, ax=ax_b, orientation='horizontal', fraction=0.046, pad=0.12)
        cb_b.set_ticks([-bmax, bmax])
        cb_b.set_ticklabels([f'{-bmax:.2f}', f'{bmax:.2f}'])

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_image_mode(checkpoints, labels, out_dir, linear_names):
    for ckpt_path, label in zip(checkpoints, labels):
        print(f"Loading {ckpt_path} ...")
        state = extract_model_state(load_checkpoint(ckpt_path))
        safe_label = label.replace(' ', '_').replace('/', '-')
        for name in linear_names:
            weight, bias = get_linear(state, name)
            if weight is None:
                print(f"  {name}: not found in checkpoint — skipping")
                continue
            safe_name = name.replace('.', '_')
            out_path = os.path.join(out_dir, f'{safe_name}_{safe_label}.png')
            render_weight_image(weight, bias, f'{name} — {label}', out_path)


# ---------------------------------------------------------------------------
# Mode 2: Statistics — plotting helpers
# ---------------------------------------------------------------------------

def _timeseries_plot(ax, x, vals, labels, title, color='steelblue',
                     ref_lines=None, ylabel=None):
    """Draw a single time-series line with optional horizontal reference lines."""
    ax.plot(x, vals, marker='o', linewidth=2, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=7)
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    if ref_lines:
        for val, col, lbl in ref_lines:
            ax.axhline(val, color=col, linestyle='--', linewidth=1, label=lbl)
        ax.legend(fontsize=7)


# Colours for time-series: one per gate
GATE_COLORS = {
    'input':  'steelblue',
    'forget': 'darkorange',
    'cell':   'forestgreen',
    'output': 'crimson',
}

# Stats that get their own time-series figure per LSTM
GATE_STAT_SPECS = [
    # (stat_key,         ylabel,          ref_line_key)
    ('bias_mean',        'bias mean',      'bias_refs'),
    ('bias_std',         'bias std',       None),
    ('saturation_pct',   '% |b|>2',        'sat_ref'),
    ('weight_ih_norm',   'wih Frob norm',  None),
    ('weight_hh_norm',   'whh Frob norm',  None),
    ('spectral_radius',  'whh spec. rad.', 'spec_ref'),
]


def plot_lstm_timeseries(all_lstm_stats, labels, lstm_name, out_dir):
    """
    One figure per statistic, with one subplot per LSTM entry
    (layer × direction), and one line per gate.
    """
    if not all_lstm_stats or not any(all_lstm_stats):
        return

    # Determine the set of entries from the first non-empty checkpoint
    sample = next(s for s in all_lstm_stats if s)
    entry_keys = sorted(sample.keys())
    x = list(range(len(labels)))

    for stat_key, ylabel, ref_key in GATE_STAT_SPECS:
        ncols = len(entry_keys)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)

        for col, entry_idx in enumerate(entry_keys):
            ax = axes[0][col]
            entry_meta = sample[entry_idx]
            subtitle = (f"layer {entry_meta['layer']} "
                        f"({'fwd' if entry_meta['direction'] == 'forward' else 'rev'})")

            for gate in GATE_NAMES:
                vals = [s.get(entry_idx, {}).get(gate, {}).get(stat_key, float('nan'))
                        for s in all_lstm_stats]
                ax.plot(x, vals, marker='o', linewidth=2,
                        color=GATE_COLORS[gate], label=gate)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=7)
            ax.set_title(subtitle, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)

            # reference lines (drawn once, not per gate)
            if ref_key == 'bias_refs' and stat_key == 'bias_mean':
                # only the forget gate has a non-zero recommended init,
                # so draw a single +1 line labelled accordingly
                ax.axhline(0.0, color='grey', linestyle=':', linewidth=1, label='0 (neutral)')
                ax.axhline(1.0, color='green', linestyle='--', linewidth=1, label='+1 (forget rec.)')
            elif ref_key == 'sat_ref':
                ax.axhline(10.0, color='orange', linestyle='--', linewidth=1, label='10% threshold')
            elif ref_key == 'spec_ref':
                ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='radius = 1')

            ax.legend(fontsize=7)

        safe_stat = stat_key.replace(' ', '_')
        safe_lstm = lstm_name.replace('.', '_')
        fig.suptitle(f'{lstm_name} — {ylabel} — all gates', fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{safe_lstm}_{safe_stat}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_linear_timeseries(all_linear_stats, labels, name, out_dir):
    """Time-series for a single named linear layer's scalar stats."""
    if not any(all_linear_stats):
        return
    keys = ['weight_norm', 'weight_std', 'weight_abs_max', 'spectral_radius',
            'bias_mean', 'bias_std']
    x = list(range(len(labels)))
    ncols = 3
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()
    for ax, key in zip(axes, keys):
        vals = [s.get(key, float('nan')) for s in all_linear_stats]
        ref_lines = None
        if key == 'spectral_radius':
            ref_lines = [(1.0, 'red', 'radius = 1')]
        _timeseries_plot(ax, x, vals, labels, f'{name} {key}',
                         ref_lines=ref_lines)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    safe_name = name.replace('.', '_')
    fig.suptitle(f'{name} Statistics Over Training', fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{safe_name}_timeseries.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _is_degenerate(all_biases, zero_threshold=1e-30):
    """
    Return True when every bias value is effectively zero — indicating the
    LSTM collapsed during training (all weights and biases driven to zero,
    typically by AdaDelta when gradient signal vanishes early).

    The threshold is well below any plausible trained value but safely above
    true floating-point zero, accommodating float32 underflow artifacts.
    """
    return bool(np.all(np.abs(all_biases) < zero_threshold))


def _draw_degenerate_panel(ax, lstm_prefix, gate):
    """Replace a histogram panel with an explanatory text box."""
    ax.set_axis_off()
    msg = (
        f"{gate} gate\n\n"
        "ALL WEIGHTS & BIASES ≈ 0\n\n"
        "This LSTM collapsed during training —\n"
        "all parameters driven to zero.\n\n"
        "Likely cause: gradient signal vanished\n"
        "early (common with AdaDelta), leaving\n"
        "the optimizer accumulator empty and\n"
        "unable to make further updates.\n\n"
        "Consider: forget gate bias init > 0,\n"
        "a different optimizer, or verifying\n"
        "that this stack is used by the model."
    )
    ax.text(0.5, 0.5, msg,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=8,
            color='firebrick',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                      edgecolor='firebrick', linewidth=1.5))


def plot_gate_bias_distributions(all_states, all_lstm_prefixes, labels, out_dir):
    """
    Histogram of bias values for all four gates, one LSTM at a time.
    All checkpoints overlaid; one subplot per gate.

    If all biases for a gate are effectively zero across all checkpoints
    (degenerate / collapsed LSTM), the histogram is replaced with an
    explanatory text panel rather than the misleading float32-underflow spike.
    """
    colors = list(mcolors.TABLEAU_COLORS.values())
    for lstm_prefix in all_lstm_prefixes:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # Collect pooled biases per gate across all checkpoints so we can
        # check for degeneracy before deciding whether to draw a histogram.
        gate_biases = {gate: [] for gate in GATE_NAMES}
        gate_colors = []

        for i, (state, label) in enumerate(zip(all_states, labels)):
            gd = get_lstm_gate_weights(state, lstm_prefix)
            if not gd:
                continue
            gate_colors.append((i, label, colors[i % len(colors)]))
            for gate in GATE_NAMES:
                pooled = np.concatenate([entry['bias'][gate] for entry in gd.values()])
                gate_biases[gate].append((label, pooled, colors[i % len(colors)]))

        degenerate = False
        for g_idx, gate in enumerate(GATE_NAMES):
            ax = axes[g_idx]
            all_pooled = (np.concatenate([b for _, b, _ in gate_biases[gate]])
                          if gate_biases[gate] else np.array([]))

            if len(all_pooled) == 0 or _is_degenerate(all_pooled):
                _draw_degenerate_panel(ax, lstm_prefix, gate)
                degenerate = True
                continue

            for label, pooled, color in gate_biases[gate]:
                ax.hist(pooled, bins=50, alpha=0.55,
                        label=label, color=color, density=True)
            for val, col, lbl in GATE_BIAS_REFS[gate]:
                ax.axvline(val, color=col, linestyle='--', linewidth=1.5, label=lbl)
            ax.set_title(f'{gate} gate bias')
            ax.set_xlabel('bias (pre-sigmoid)')
            ax.set_ylabel('density')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        title = f'{lstm_prefix} — Gate Bias Distributions (all layers pooled)'
        if degenerate:
            title += '  ⚠ COLLAPSED'
        safe_lstm = lstm_prefix.replace('.', '_')
        fig.suptitle(title, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{safe_lstm}_gate_bias_distributions.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_linear_distributions(all_states, linear_names, labels, out_dir):
    """Histogram of weight and bias values for each named linear, across checkpoints."""
    colors = list(mcolors.TABLEAU_COLORS.values())
    for name in linear_names:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, (state, label) in enumerate(zip(all_states, labels)):
            w, b = get_linear(state, name)
            if w is None:
                continue
            color = colors[i % len(colors)]
            axes[0].hist(w.flatten(), bins=80, alpha=0.5, label=label, color=color, density=True)
            if b is not None:
                axes[1].hist(b.flatten(), bins=40, alpha=0.5, label=label, color=color, density=True)

        for ax, subtitle, xlabel in zip(axes,
                                        ['weight distribution', 'bias distribution'],
                                        ['weight value', 'bias value']):
            ax.set_title(f'{name} {subtitle}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        safe_name = name.replace('.', '_')
        fig.suptitle(f'{name} Value Distributions', fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{safe_name}_distributions.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Mode 2: Statistics — top-level runner
# ---------------------------------------------------------------------------

def _print_gate_summary(gate_stats, lstm_prefix):
    # Check for full collapse before printing individual gate lines
    all_norms = [gate_stats[entry_idx][gate]['weight_hh_norm']
                 for entry_idx in gate_stats
                 for gate in GATE_NAMES]
    if all_norms and max(all_norms) < 1e-30:
        print(f"  *** WARNING: {lstm_prefix} appears fully collapsed "
              f"(all weights & biases ≈ 0). "
              f"Likely cause: gradient signal vanished early during AdaDelta training. ***")
        return

    for entry_idx, entry in sorted(gate_stats.items()):
        layer = entry['layer']
        direc = 'fwd' if entry['direction'] == 'forward' else 'rev'
        for gate in GATE_NAMES:
            gs = entry[gate]
            print(f"    {lstm_prefix} L{layer}/{direc} [{GATE_LABELS[gate]}]  "
                  f"bias={gs['bias_mean']:+.3f}±{gs['bias_std']:.3f}  "
                  f"sat={gs['saturation_pct']:5.1f}%  "
                  f"wih={gs['weight_ih_norm']:.3f}  "
                  f"whh={gs['weight_hh_norm']:.3f}  "
                  f"ρ(whh)={gs['spectral_radius']:.3f}")


def print_forget_gate_summary(labels, lstm_prefixes, all_lstm_stats):
    """
    Print a compact summary table of forget gate bias mean across all
    checkpoints for each LSTM, one row per checkpoint.

    Collapsed LSTMs are shown as '  COLLAPSED' rather than a numeric value
    so the table remains readable even when some modules degenerate.

    Example output:

      Forget gate bias summary (mean across all layers/directions)
      checkpoint                word_lstm   transition_stack.lstm   constituent_stack.lstm
      epoch_001                    +1.000                  +1.000                   +1.000
      epoch_010                    +0.910                  +0.972                   +0.936
      epoch_050                    +0.563               COLLAPSED                   +0.872
    """
    print("\n" + "=" * 72)
    print("  Forget gate bias summary (mean across all layers/directions)")

    # column width: max of prefix length and a sample value like '+0.000'
    col_w = max(10, max(len(p) for p in lstm_prefixes))
    label_w = max(12, max(len(l) for l in labels))

    # header
    header = f"  {'checkpoint':<{label_w}}"
    for prefix in lstm_prefixes:
        header += f"  {prefix:>{col_w}}"
    print(header)

    for label, *stats_per_prefix in zip(labels, *[all_lstm_stats[p] for p in lstm_prefixes]):
        row = f"  {label:<{label_w}}"
        for gate_stats in stats_per_prefix:
            if not gate_stats:
                # prefix not found in this checkpoint
                row += f"  {'N/A':>{col_w}}"
                continue

            # check for collapse
            all_norms = [gate_stats[entry_idx][gate]['weight_hh_norm']
                         for entry_idx in gate_stats
                         for gate in GATE_NAMES]
            if all_norms and max(all_norms) < 1e-30:
                row += f"  {'COLLAPSED':>{col_w}}"
                continue

            # average forget gate bias_mean across all layers and directions
            forget_means = [gate_stats[entry_idx]['forget']['bias_mean']
                            for entry_idx in gate_stats]
            avg = sum(forget_means) / len(forget_means)
            row += f"  {avg:>+{col_w}.3f}"
        print(row)

    print("=" * 72 + "\n")


def run_stats_mode(checkpoints, labels, out_dir, lstm_prefixes, linear_names,
                   no_plots=False):
    all_states      = []
    # { linear_name: [stats_per_checkpoint] }
    all_linear_stats = {n: [] for n in linear_names}
    # { lstm_prefix: [stats_per_checkpoint] }
    all_lstm_stats   = {p: [] for p in lstm_prefixes}

    for ckpt_path, label in zip(checkpoints, labels):
        print(f"Loading {ckpt_path} ...")
        ckpt  = load_checkpoint(ckpt_path)
        state = extract_model_state(ckpt)
        all_states.append(state)

        print(f"\n  === {label} ===")

        # each named linear
        for name in linear_names:
            w, b = get_linear(state, name)
            if w is None:
                print(f"  {name}: not found in checkpoint")
                all_linear_stats[name].append({})
                continue
            ls = compute_linear_stats(w, b)
            all_linear_stats[name].append(ls)
            print(f"  {name}: norm={ls['weight_norm']:.4f}  "
                  f"std={ls['weight_std']:.4f}  "
                  f"ρ={ls['spectral_radius']:.4f}  "
                  f"bias_mean={ls.get('bias_mean', float('nan')):.4f}")

        # each LSTM
        for prefix in lstm_prefixes:
            gd = get_lstm_gate_weights(state, prefix)
            if not gd:
                print(f"  {prefix}: not found in checkpoint")
                all_lstm_stats[prefix].append({})
                continue
            gs = compute_gate_stats(gd)
            all_lstm_stats[prefix].append(gs)
            _print_gate_summary(gs, prefix)

    print()

    # --- plots (skipped when --no_plots is set) ---
    if not no_plots:
        for name in linear_names:
            plot_linear_timeseries(all_linear_stats[name], labels, name, out_dir)

        plot_linear_distributions(all_states, linear_names, labels, out_dir)

        for prefix in lstm_prefixes:
            if any(all_lstm_stats[prefix]):
                plot_lstm_timeseries(all_lstm_stats[prefix], labels, prefix, out_dir)

        plot_gate_bias_distributions(all_states, lstm_prefixes, labels, out_dir)

    print_forget_gate_summary(labels, lstm_prefixes, all_lstm_stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_LSTMS   = ['word_lstm']
DEFAULT_LINEARS = ['reduce_linear']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize linear layer weights and LSTM gate statistics '
                    'from constituency parser checkpoints.'
    )
    parser.add_argument('--checkpoints', nargs='+', required=False, default=None,
                        help='One or more .pt checkpoint files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each checkpoint (default: filename stem)')
    parser.add_argument('--mode', choices=['image', 'stats', 'both'], default='both',
                        help='image: weight heatmaps (one PNG per linear × checkpoint); '
                             'stats: distributions & time-series — also writes PNGs, '
                             'one per statistic per LSTM/linear; '
                             'both: do both (default)')
    parser.add_argument('--linears', nargs='+', default=DEFAULT_LINEARS,
                        help='Linear layer prefixes to analyse (default: reduce_linear). '
                             'Use --list_keys to discover valid names. '
                             'Example: --linears reduce_linear word_to_constituent output_layers.0')
    parser.add_argument('--lstms', nargs='+', default=DEFAULT_LSTMS,
                        help='LSTM prefixes to analyse in stats mode '
                             '(default: word_lstm). '
                             'Example: --lstms word_lstm transition_stack.lstm constituent_stack.lstm')
    parser.add_argument('--list_keys', action='store_true',
                        help='Print all linear and LSTM prefixes found in the first '
                             'checkpoint, then exit.  Useful for discovering valid '
                             '--linears / --lstms values.')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip all PNG output in stats mode and print terminal '
                             'summaries only.  Has no effect in image mode.')
    parser.add_argument('--out_dir', default='.',
                        help='Output directory for PNG files (default: current directory)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoints is None:
        print("ERROR: --checkpoints is required")
        sys.exit(1)

    checkpoints = args.checkpoints
    labels = args.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in checkpoints]
    if len(labels) != len(checkpoints):
        print("ERROR: --labels must have the same number of entries as --checkpoints")
        sys.exit(1)

    if args.list_keys:
        print(f"Keys in {checkpoints[0]}:")
        state = extract_model_state(load_checkpoint(checkpoints[0]))
        list_keys(state)
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode in ('image', 'both'):
        print("\n--- Weight image mode ---")
        run_image_mode(checkpoints, labels, args.out_dir, args.linears)

    if args.mode in ('stats', 'both'):
        print("\n--- Statistics mode ---")
        run_stats_mode(checkpoints, labels, args.out_dir, args.lstms, args.linears,
                       no_plots=args.no_plots)

    print("\nDone.")


if __name__ == '__main__':
    main()
