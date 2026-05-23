# =============================================================================
# FILE: neuron_viz_pipeline/scripts/stage2_rank.py
#
# Purpose:
#   Thin runner for Stage 2 — ranking top-k images per neuron.
#
# Preconditions:
#   Stage 1 must have been run successfully so that
#     neuron_viz_pipeline/results/{model}/{xai}/{layer}/activations/
#       activations_{layer}_{target_type}_{pool_type}.safetensors
#   exists on disk. (Checked at startup.)
#
# Usage (from the repo root):
#   python neuron_viz_pipeline/scripts/stage2_rank.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml
#
# Smoke test with overrides (matches Stage 1 smoke test):
#   python neuron_viz_pipeline/scripts/stage2_rank.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml \
#       --override rank.top_k=50 \
#       --override rank.chunk_size=200
#
# Output (path from src/utils/paths.py):
#   neuron_viz_pipeline/results/{model}/{xai}/{layer}/top_k/
#       top_activations_{layer}_{target_type}_indices.npy   (num_neurons, top_k)
#       top_activations_{layer}_{target_type}_values.npy    (num_neurons, top_k)
#       top_activations_{layer}_{target_type}_metadata.json
# =============================================================================

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make our package importable when this script is run from the repo root.
# ---------------------------------------------------------------------------
_here     = Path(__file__).resolve()
_pkg_root = _here.parent.parent         # neuron_viz_pipeline/
sys.path.insert(0, str(_pkg_root))
_repo_root = _pkg_root.parent            # project root
sys.path.insert(0, str(_repo_root))

import torch

from src.utils.config import load_config
from src.utils.paths import (
    activations_file, top_k_dir, top_indices_file, top_values_file,
    top_k_metadata_file, ensure_dirs,
)
from src.utils.io_safetensors import open_safetensors_memmap
from src.rank import (
    aggregate_chunked, compute_top_activations, save_results, analyze_results,
)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 — Rank top-k images per neuron (neuron_viz_pipeline)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file, e.g. neuron_viz_pipeline/configs/rn152_ixg.yaml",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override a config value with dotted-key syntax. "
             "Repeat the flag to set multiple. "
             "Example: --override rank.top_k=50 --override rank.chunk_size=500",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply --override k1.k2.k3=VALUE flags to the config.
    VALUE is parsed as YAML (so '150' → int, 'true' → bool, etc.).
    Same implementation as scripts/stage1_extract.py.
    """
    import yaml
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(
                f"Invalid --override '{kv}' — expected format KEY.PATH=VALUE"
            )
        key_path, value_str = kv.split("=", 1)
        value = yaml.safe_load(value_str)
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        print(f"  [override] cfg.{key_path} = {value!r}")
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.override:
        print("Applying overrides:")
        cfg = apply_overrides(cfg, args.override)

    print("\n" + "=" * 70)
    print(f"STAGE 2 — RANK TOP-K IMAGES PER NEURON")
    print("=" * 70)
    print(f"  config         : {args.config}")
    print(f"  model          : {cfg['model']['name']}")
    print(f"  layer          : {cfg['model']['layer']}")
    print(f"  layer_type     : {cfg['model']['layer_type']}")
    print(f"  target_type    : {cfg['extract']['target_type']}")
    print(f"  aggregation    : {cfg['rank']['aggregation']}")
    if cfg["rank"]["aggregation"] == "top_mean":
        print(f"  top_percentile : {cfg['rank']['top_percentile']}%")
    print(f"  top_k          : {cfg['rank']['top_k']}")
    print(f"  chunk_size     : {cfg['rank']['chunk_size']}")
    print(f"  save_values    : {cfg['rank']['save_values']}")
    print()

    # ── Paths ────────────────────────────────────────────────────────
    act_file      = activations_file(cfg)
    indices_path  = top_indices_file(cfg)
    values_path   = top_values_file(cfg)
    metadata_path = top_k_metadata_file(cfg)

    print(f"  input  : {act_file}")
    print(f"  output : {top_k_dir(cfg)}")

    if not act_file.exists():
        print(
            f"\n  ERROR: Stage 1 output not found at:\n    {act_file}\n\n"
            f"  Run Stage 1 first:\n"
            f"    python neuron_viz_pipeline/scripts/stage1_extract.py "
            f"--config {args.config}"
        )
        sys.exit(1)

    act_size_gb = act_file.stat().st_size / (1024 ** 3)
    print(f"  input size: {act_size_gb:.2f} GB")

    # Warn if output exists (overwrite without prompt — topk is cheap)
    if indices_path.exists():
        print(f"  NOTE: output {indices_path.name} exists and will be overwritten.")

    ensure_dirs(cfg, cfg["neuron"]["channel_id"])

    # ── Open activation memmap (lazy — no RAM cost yet) ──────────────
    layer_name = cfg["model"]["layer"]
    print(f"\n  opening activations via lazy memmap...")
    memmap, resolved_layer_name = open_safetensors_memmap(
        str(act_file), layer_name=layer_name
    )
    print(f"  layer in file : '{resolved_layer_name}'")
    print(f"  memmap shape  : {tuple(memmap.shape)}")
    print(f"  memmap dtype  : {memmap.dtype}")

    if resolved_layer_name != layer_name:
        print(
            f"  WARNING: layer name in safetensors file "
            f"('{resolved_layer_name}') differs from config layer "
            f"('{layer_name}'). Using the one in the file."
        )

    # ── Chunked aggregation ──────────────────────────────────────────
    start_agg = time.time()
    aggregated = aggregate_chunked(
        memmap        = memmap,
        layer_type    = cfg["model"]["layer_type"],
        aggregation   = cfg["rank"]["aggregation"],
        top_percentile= cfg["rank"]["top_percentile"],
        chunk_size    = cfg["rank"]["chunk_size"],
    )
    agg_elapsed = time.time() - start_agg
    print(f"\n  aggregation done in {agg_elapsed:.1f}s")
    print(f"  aggregated tensor: shape={tuple(aggregated.shape)} "
          f"dtype={aggregated.dtype}  "
          f"size~{aggregated.element_size() * aggregated.nelement() / 1024**2:.0f} MB")

    # ── Top-k ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  running topk on device={device}...")
    start_topk = time.time()
    top_indices, top_values = compute_top_activations(
        activations   = aggregated,
        top_k         = cfg["rank"]["top_k"],
        aggregation   = cfg["rank"]["aggregation"],       # signature-compat
        percentile    = None,                             # not using percentile filter
        batch_size    = 1000,                             # neurons per GPU batch
        device        = device,
        top_percentile= cfg["rank"]["top_percentile"],    # signature-compat
    )

    # If Stage 1 saved only one channel, top_indices/top_values currently have
    # one row at index 0. Expand them so downstream stages can still address
    # the original model neuron id, e.g. neuron.channel_id=652.
    channel_id_only = cfg["extract"].get("channel_id_only")
    if channel_id_only is not None:
        channel_id_only = int(channel_id_only)
        if top_indices.shape[0] != 1:
            raise RuntimeError(
                f"extract.channel_id_only={channel_id_only} was set, but Stage 2 "
                f"saw {top_indices.shape[0]} extracted features instead of 1."
            )
        sparse_indices = torch.full(
            (channel_id_only + 1, top_indices.shape[1]),
            -1,
            dtype=top_indices.dtype,
        )
        sparse_values = torch.full(
            (channel_id_only + 1, top_values.shape[1]),
            float("-inf"),
            dtype=top_values.dtype,
        )
        sparse_indices[channel_id_only] = top_indices[0]
        sparse_values[channel_id_only] = top_values[0]
        top_indices, top_values = sparse_indices, sparse_values
        print(
            f"  channel_id_only mode: expanded one-column rankings to row "
            f"{channel_id_only}"
        )

    topk_elapsed = time.time() - start_topk
    print(f"  topk done in {topk_elapsed:.1f}s")
    print(f"  top_indices: shape={tuple(top_indices.shape)} dtype={top_indices.dtype}")
    print(f"  top_values : shape={tuple(top_values.shape)}  dtype={top_values.dtype}")

    # ── Diagnostic ───────────────────────────────────────────────────
    analyze_results(top_indices, top_values, layer_name)

    # ── Metadata — matches compute_top_activations.py schema ─────────
    metadata = {
        "layer_name":         layer_name,
        "input_file":         str(act_file),
        "target_type":        cfg["extract"]["target_type"],
        "original_shape":     list(memmap.shape),
        "top_k":              cfg["rank"]["top_k"],
        "aggregation":        cfg["rank"]["aggregation"],
        "top_percentile":     (cfg["rank"]["top_percentile"]
                               if cfg["rank"]["aggregation"] == "top_mean" else None),
        "percentile":         None,
        "num_samples":        int(memmap.shape[0]),
        "num_neurons":        int(top_indices.shape[0]),
        "valid_activations":  int((top_indices != -1).sum()),
        "value_range": (
            [float(top_values[top_indices != -1].min()),
             float(top_values[top_indices != -1].max())]
            if (top_indices != -1).any() else [0.0, 0.0]
        ),
    }

    # ── Save ─────────────────────────────────────────────────────────
    print(f"\n  saving results...")
    save_results(
        top_indices   = top_indices,
        top_values    = top_values,
        indices_path  = str(indices_path),
        values_path   = str(values_path),
        metadata_path = str(metadata_path),
        metadata      = metadata,
        save_values   = cfg["rank"]["save_values"],
    )

    # ── Verify ───────────────────────────────────────────────────────
    if not indices_path.exists():
        print(f"  ERROR: expected output not found: {indices_path}")
        sys.exit(1)

    total_elapsed = agg_elapsed + topk_elapsed
    print(f"\n  total time: {total_elapsed:.1f}s "
          f"(aggregate: {agg_elapsed:.1f}s, topk: {topk_elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print(f"STAGE 2 DONE — next step:")
    print(f"  Stage 3 (XAI maps / IxG) runner will be built in Step 5.")
    print(f"  You can inspect results with:")
    print(f"    python -c \"import numpy as np; "
          f"a = np.load('{indices_path}'); "
          f"print('shape:', a.shape); "
          f"print('channel {cfg['neuron']['channel_id']} top-5 sample indices:', "
          f"a[{cfg['neuron']['channel_id']}, :5])\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
