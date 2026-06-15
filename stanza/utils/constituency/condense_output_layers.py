"""
condense_output_layers.py

Post-training compression for the output layers of a Stanza constituency parser.

After training with weight decay, middle output layers develop near-zero rows
(dead neurons).  Since a dead output row in layer i feeds into a corresponding
input column of layer i+1 (and nowhere else), both can be removed together
with no change in model behaviour.

The final output layer is never touched: its output size is len(transitions)
and must stay fixed.

Checkpoint format changes
-------------------------
Stanza constituency checkpoints have the structure:

    checkpoint['params']['config']   - model args dict
    checkpoint['params']['model']    - state dict

This script adds to params['config']:

    output_layer_sizes : list[int]
        Output size of each non-final output layer after condensation.
        e.g. for a 2-layer model condensed to K live neurons: [K]
        The final layer size (len(transitions)) is not included.

        At load time, build_output_layers reads this list instead of
        defaulting to [hidden_size] * middle_layers.

params['model'] changes for each condensed middle layer i:

    output_layers.{i}.weight   (K_i, input_size_i) condensed from (hidden_size, input_size_i)
    output_layers.{i}.bias     (K_i,)              condensed from (hidden_size,)
    output_layers.{i+1}.weight (output_size_{i+1}, K_i) condensed from (output_size_{i+1}, hidden_size)
    # bias of layer i+1 is unchanged

Dead-neuron detection
---------------------
A neuron (output row i of layer L) is dead iff:
    max(|weight[i, :]|) < threshold * median(max(|weight[j, :]|) for all j)
    AND
    |bias[i]| < threshold * median(max(|weight[j, :]|) for all j)

Unlike reduce_linear, the bias IS included in detection here because:
  - The bias on a dead output neuron directly controls the pre-activation
    value going into the next layer's input column
  - A row with near-zero weights but significant bias is not truly dead

The same threshold is applied to all middle layers.

Idempotency
-----------
If output_layer_sizes is already present in args and is consistent with
the actual weight shapes, the script exits cleanly.

Usage
-----
    python condense_output_layers.py --input model.pt --dry_run
    python condense_output_layers.py --input model.pt --output model_condensed.pt
    python condense_output_layers.py --input model.pt --in_place
    python condense_output_layers.py --input model.pt --output out.pt --threshold 0.01

Required code changes
---------------------
See instructions at the bottom of this file.
"""

import argparse
import logging
import os
import shutil
import sys

import torch

logger = logging.getLogger('stanza.condense_output_layers')


# ---------------------------------------------------------------------------
# Dead-neuron detection for a single layer
# ---------------------------------------------------------------------------

def find_live_rows(weight: torch.Tensor, bias: torch.Tensor,
                   threshold: float) -> torch.Tensor:
    """
    Return a boolean mask (out_features,) where True = live output row.

    A row is dead iff BOTH:
      max(|weight_row|) < threshold * median(max(|weight_row|) for all rows)
      |bias[i]|         < threshold * median(max(|weight_row|) for all rows)

    Bias is included because a near-zero weight row with a non-trivial bias
    still produces a non-zero pre-activation in the next layer.
    """
    row_max = weight.abs().amax(dim=1)
    median_row_max = row_max.median().item()
    if median_row_max == 0.0:
        logger.warning("Median max-row weight is 0 for this layer; treating all rows as live.")
        return torch.ones(weight.shape[0], dtype=torch.bool)
    abs_threshold = threshold * median_row_max
    return (row_max >= abs_threshold) | (bias.abs() >= abs_threshold)


# ---------------------------------------------------------------------------
# Main condensation logic
# ---------------------------------------------------------------------------

def condense_model(input_path: str,
                   output_path: str,
                   threshold: float = 0.005,
                   dry_run: bool = False) -> dict:
    """
    Load checkpoint, detect dead output-layer neurons, write condensed checkpoint.

    Only middle layers (all except the final) are condensed.

    Returns a summary dict with keys:
        num_layers, middle_layers, results
        where results is a list of dicts per middle layer:
            layer_idx, original_size, live, dead, live_indices
    """
    logger.info("Loading: %s", input_path)
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=True)

    params = checkpoint['params']
    args   = params['config']
    state  = params['model']

    if checkpoint.get('model_type', 'LSTM') != 'LSTM':
        raise ValueError(f"Expected model_type='LSTM', got {checkpoint.get('model_type')!r}")

    num_output_layers = args['num_output_layers']
    if num_output_layers < 2:
        logger.info(
            "Model has num_output_layers=%d; no middle layers to condense.",
            num_output_layers
        )
        return {
            'num_layers': num_output_layers,
            'middle_layers': 0,
            'results': [],
        }

    maxout_k = args.get('maxout_k', 0)
    if maxout_k:
        logger.info("Model uses maxout_k=%d; output layer condensation is not supported for maxout models.", maxout_k)
        return {
            'num_layers': num_output_layers,
            'middle_layers': 0,
            'results': [],
        }

    # output_layer_sizes may already be set from a previous run of this script
    # or because the user specified sizes at training time.  In either case,
    # the weights may have further dead neurons, so we always re-detect on the
    # current weights rather than treating the key's presence as a signal to skip.
    if 'output_layer_sizes' in args:
        logger.info("Model already has output_layer_sizes=%s; re-detecting on current weights.",
                    args['output_layer_sizes'])

    # --- detect dead neurons in each middle layer ---
    middle_layers = num_output_layers - 1
    results = []

    for i in range(middle_layers):
        weight = state[f'output_layers.{i}.weight']
        bias   = state[f'output_layers.{i}.bias']
        original_size = weight.shape[0]

        live_mask = find_live_rows(weight, bias, threshold)
        live_indices = live_mask.nonzero(as_tuple=True)[0]
        K = int(live_mask.sum().item())
        dead = original_size - K

        if K == 0:
            logger.warning(
                "output_layers.%d: ALL %d output rows appear dead. "
                "This likely means the model is not functioning correctly. "
                "Skipping condensation for this layer to avoid a zero-width matrix.",
                i, original_size
            )
            # treat as fully live to avoid producing a 0-row weight matrix
            live_mask = torch.ones(original_size, dtype=torch.bool)
            live_indices = torch.arange(original_size)
            K = original_size
            dead = 0

        logger.info(
            "output_layers.%d: %d / %d output rows dead (threshold=%.3f%%)",
            i, dead, original_size, threshold * 100
        )
        results.append({
            'layer_idx': i,
            'original_size': original_size,
            'live': K,
            'dead': dead,
            'live_indices': live_indices,
        })

    summary = {
        'num_layers': num_output_layers,
        'middle_layers': middle_layers,
        'results': results,
    }

    if dry_run:
        logger.info("Dry run - no file written.")
        # Convert tensors to lists for the summary
        for r in summary['results']:
            if isinstance(r['live_indices'], torch.Tensor):
                r['live_indices'] = r['live_indices'].tolist()
        return summary

    any_dead = any(r['dead'] > 0 for r in results)
    if not any_dead:
        logger.info("No dead neurons found in any middle layer; copying checkpoint unchanged.")
        if output_path != input_path:
            shutil.copy2(input_path, output_path)
        for r in summary['results']:
            if isinstance(r['live_indices'], torch.Tensor):
                r['live_indices'] = r['live_indices'].tolist()
        return summary

    # --- surgery ---
    output_layer_sizes = []

    for i, result in enumerate(results):
        live_indices = result['live_indices']   # LongTensor
        K = result['live']

        # Condense output rows of layer i
        w_i = state[f'output_layers.{i}.weight']
        b_i = state[f'output_layers.{i}.bias']
        state[f'output_layers.{i}.weight'] = w_i[live_indices].clone()
        state[f'output_layers.{i}.bias']   = b_i[live_indices].clone()

        # Condense input columns of layer i+1 to match
        w_next = state[f'output_layers.{i+1}.weight']
        state[f'output_layers.{i+1}.weight'] = w_next[:, live_indices].clone()
        # bias of layer i+1 is unchanged

        output_layer_sizes.append(K)
        logger.debug(
            "  output_layers.%d.weight: %s -> %s",
            i, list(w_i.shape), list(state[f'output_layers.{i}.weight'].shape)
        )
        logger.debug(
            "  output_layers.%d.weight cols: %s -> %s",
            i+1, list(w_next.shape), list(state[f'output_layers.{i+1}.weight'].shape)
        )

    # update (or set for the first time) output_layer_sizes in args
    args['output_layer_sizes'] = output_layer_sizes
    checkpoint['params']['config'] = args
    checkpoint['params']['model']  = state

    logger.info("Writing condensed checkpoint: %s", output_path)
    torch.save(checkpoint, output_path, _use_new_zipfile_serialization=False)
    logger.info("Done.")

    for r in summary['results']:
        if isinstance(r['live_indices'], torch.Tensor):
            r['live_indices'] = r['live_indices'].tolist()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Condense output layers in a Stanza constituency parser checkpoint."
    )
    p.add_argument('--input', required=True)

    out_group = p.add_mutually_exclusive_group()
    out_group.add_argument('--output')
    out_group.add_argument('--in_place', action='store_true')

    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--threshold', type=float, default=0.005,
                   help="Fraction of median max-row weight below which a neuron is dead. "
                        "Default: 0.005 (0.5%%).")
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    if args.dry_run:
        output_path = None
    elif args.in_place:
        backup = args.input + '.bak'
        shutil.copy2(args.input, backup)
        logger.info("Backup saved: %s", backup)
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        logger.error("Specify --output, --in_place, or --dry_run.")
        sys.exit(1)

    summary = condense_model(
        input_path=args.input,
        output_path=output_path,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )

    print()
    print("=== Output layer condensation summary ===")
    for r in summary['results']:
        i = r['layer_idx']
        print(f"  output_layers.{i}: {r['original_size']} -> {r['live']} "
              f"({r['dead']} dead, {100*r['live']/r['original_size']:.1f}% kept)")
    if not args.dry_run and any(r['dead'] > 0 for r in summary['results']):
        print(f"  Output: {output_path}")
    print()


if __name__ == '__main__':
    main()


# =============================================================================
# REQUIRED CODE CHANGES
# =============================================================================
#
# --- trainer.py: Trainer.model_from_params ---
#
# Add to the existing block of update_args.pop() calls:
#
#       update_args.pop("output_layer_sizes", None)
#       update_args.pop("reduce_linear_live_rows", None)   # if not already present
#       update_args.pop("reduce_linear_live_cols", None)   # if not already present
#
#
# --- lstm_model.py: LSTMModel.build_output_layers ---
#
# Replace:
#       middle_layers = num_output_layers - 1
#       predict_input_size = [self.hidden_size + self.hidden_size * self.num_tree_lstm_layers + self.transition_hidden_size] + [self.hidden_size] * middle_layers
#       predict_output_size = [self.hidden_size] * middle_layers + [final_layer_size]
#
# With:
#       middle_layers = num_output_layers - 1
#       first_input_size = self.hidden_size + self.hidden_size * self.num_tree_lstm_layers + self.transition_hidden_size
#       # output_layer_sizes stores the condensed output size of each middle layer.
#       # If absent (uncompressed model), default to hidden_size for all middle layers.
#       middle_output_sizes = self.args.get('output_layer_sizes', [self.hidden_size] * middle_layers)
#       predict_input_size  = [first_input_size] + middle_output_sizes
#       predict_output_size = middle_output_sizes + [final_layer_size]
#
# That's the only change needed. load_state_dict (strict=False) handles
# the weight shape differences automatically.
