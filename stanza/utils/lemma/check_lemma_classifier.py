"""
Analyze the weight matrices of one or more LemmaClassifierLSTM models,
reporting per-tensor sizes and SVD effective rank.

Handles three input formats:
  1. A standalone contextual lemmatizer .pt file (saved directly)
  2. A combined lemmatizer .pt file (contextual lemmatizers under checkpoint['contextual'])
  3. A get_save_dict() style dict already in memory (used internally)

Usage:
    # standalone contextual lemmatizer:
    python3 check_lemma_classifier.py path/to/contextual.pt

    # combined lemmatizer (reports all embedded contextual lemmatizers):
    python3 check_lemma_classifier.py path/to/sl_combined_charlm_lemmatizer.pt
"""

import sys
import torch


def effective_rank(W, thresholds=(0.90, 0.95, 0.99)):
    """
    Return the number of singular values needed to explain each threshold
    of total variance in W.  W must be a 2D tensor.
    """
    _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
    S = S / S.sum()
    cumvar = torch.cumsum(S, dim=0)
    return {t: int((cumvar < t).sum().item()) + 1 for t in thresholds}


def analyze(save_dict, label=""):
    params = save_dict["params"]
    args   = save_dict.get("args", {})

    print(f"{'='*70}")
    if label:
        print(f"  {label}")
    print(f"  hidden_dim={args.get('hidden_dim', '?')}  "
          f"known_words={len(save_dict.get('known_words', []))}  "
          f"target_words={save_dict.get('target_words', [])}  "
          f"label_decoder={save_dict.get('label_decoder', '?')}")
    print()

    # ---- size breakdown ----
    print(f"  {'Tensor':<50} {'MB':>7}  shape")
    print(f"  {'-'*70}")
    total = 0
    for k, v in sorted(params.items(), key=lambda x: -x[1].numel() * x[1].element_size()):
        mb = v.numel() * v.element_size() / 1024 / 1024
        total += mb
        print(f"  {k:<50} {mb:>7.2f}  {tuple(v.shape)}")
    print(f"  {'TOTAL':<50} {total:>7.2f}")
    print()

    # ---- effective rank on LSTM weight matrices ----
    lstm_keys = [k for k in params if k.startswith("lstm.weight_") and "ih" in k or
                 k.startswith("lstm.weight_") and "hh" in k]
    if lstm_keys:
        print(f"  {'Tensor':<45}  {'90%':>5}  {'95%':>5}  {'99%':>5}  dims")
        print(f"  {'-'*70}")
        for k in sorted(lstm_keys):
            W = params[k]
            if W.dim() != 2:
                print(f"  {k:<45}  (skipped, ndim={W.dim()})")
                continue
            ranks = effective_rank(W)
            hidden = args.get('hidden_dim', '?')
            print(f"  {k:<45}  {ranks[0.90]:>5}  {ranks[0.95]:>5}  {ranks[0.99]:>5}  "
                  f"(of {W.shape[0]}×{W.shape[1]})")
    else:
        print("  No LSTM weight matrices found (attention model?).")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}\n")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Case 1: combined lemmatizer — contextual lemmatizers under 'contextual'
    if "contextual" in checkpoint and isinstance(checkpoint["contextual"], list) and len(checkpoint.get("contextual", [])) > 0:
        contextuals = checkpoint["contextual"]
        print(f"Found {len(contextuals)} contextual lemmatizer(s) embedded in combined model.\n")
        for i, save_dict in enumerate(contextuals):
            analyze(save_dict, label=f"Contextual lemmatizer {i}")
    # Case 2: standalone contextual lemmatizer saved with torch.save
    elif "params" in checkpoint and isinstance(checkpoint["params"], dict):
        analyze(checkpoint, label=path)
    else:
        print("Unrecognized checkpoint format.")
        print(f"Top-level keys: {list(checkpoint.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
