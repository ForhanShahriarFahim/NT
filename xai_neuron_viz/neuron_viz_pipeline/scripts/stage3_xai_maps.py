# =============================================================================
# FILE: neuron_viz_pipeline/scripts/stage3_xai_maps.py
#
# Purpose:
#   Thin runner for Stage 3 — compute XAI saliency maps for the top-k images
#   of one neuron (cfg.neuron.channel_id) at the target layer.
#
# Preconditions:
#   Stage 2 must have been run so that
#     neuron_viz_pipeline/results/{model}/{xai}/{layer}/top_k/
#       top_activations_{layer}_{target_type}_indices.npy
#   exists on disk. (Checked at startup.)
#
# Usage (from the repo root):
#   python neuron_viz_pipeline/scripts/stage3_xai_maps.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml
#
# Quick smoke test (matches existing 200-sample / top-50 state):
#   python neuron_viz_pipeline/scripts/stage3_xai_maps.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml \
#       --override rank.top_k=50
#
# ViT smoke test:
#   python neuron_viz_pipeline/scripts/stage3_xai_maps.py \
#       --config neuron_viz_pipeline/configs/vit_ixg.yaml \
#       --override rank.top_k=50
#
# Output (path from src/utils/paths.py):
#   neuron_viz_pipeline/results/{model}/{xai}/{layer}/xai_maps/neuron_{ccid:03d}/
#       {method}_maps.safetensors        shape [top_k, H, W] float32
#       {method}_maps_metadata.json      includes valid_ranks list
#
# Design notes:
#   - One image at a time (batch_size=1) — IxG backward retains the graph,
#     so memory grows with batch. On a 15 GB GPU, batch=1 gives plenty of
#     headroom for both ResNet-152 and ViT-B/16.
#   - Padding: if Stage 2 wrote -1 for ranks where num_samples < top_k,
#     those positions are filled with zeros AND recorded in valid_ranks
#     metadata so Stage 4 can skip them cleanly.
#   - Progress: logs every 25 maps processed.
# =============================================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make our package importable when this script is run from the repo root.
# Mirrors the pattern in scripts/stage1_extract.py and stage2_rank.py.
# ---------------------------------------------------------------------------
_here       = Path(__file__).resolve()
_pkg_root   = _here.parent.parent         # neuron_viz_pipeline/
sys.path.insert(0, str(_pkg_root))
_repo_root  = _pkg_root.parent            # project root
sys.path.insert(0, str(_repo_root))

from src.utils.config import load_config
from src.utils.paths import (
    top_indices_file, xai_maps_dir, xai_maps_file, ensure_dirs,
    assert_dataset_exists,
)
from src.utils.io_safetensors import save_safetensors_streaming
from src.data import build_preprocessed_dataset
from src.models import build_model
from src.models.builder import get_layer
from src.xai import get_xai_method, available_methods


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 — Compute XAI saliency maps (neuron_viz_pipeline)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file, e.g. neuron_viz_pipeline/configs/rn152_ixg.yaml",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override a config value with dotted-key syntax. "
             "Repeat the flag to set multiple. "
             "Example: --override rank.top_k=50 "
             "--override xai.ixg.vit_include_cls=true",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply --override k1.k2.k3=VALUE flags to the config. Same implementation
    as scripts/stage1_extract.py and scripts/stage2_rank.py.
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

    channel_id = cfg["neuron"]["channel_id"]
    top_k      = cfg["rank"]["top_k"]
    method_name = cfg["xai"]["method"]
    layer_name  = cfg["model"]["layer"]
    layer_type  = cfg["model"]["layer_type"]

    print("\n" + "=" * 70)
    print(f"STAGE 3 — COMPUTE XAI SALIENCY MAPS")
    print("=" * 70)
    print(f"  config          : {args.config}")
    print(f"  model           : {cfg['model']['name']}")
    print(f"  layer           : {layer_name}")
    print(f"  layer_type      : {layer_type}")
    print(f"  xai method      : {method_name}  "
          f"(available: {', '.join(available_methods())})")
    print(f"  target_channel  : {channel_id}")
    print(f"  top_k           : {top_k}")
    # Method-specific config is printed by each XAI method's own __init__
    # (which runs below in `get_xai_method`). That way the output is
    # always correct for whichever method was chosen.
    print()

    # ── Paths ────────────────────────────────────────────────────────
    indices_path = top_indices_file(cfg)
    out_dir      = xai_maps_dir(cfg, channel_id)
    out_file     = xai_maps_file(cfg, channel_id)

    print(f"  input  : {indices_path}")
    print(f"  output : {out_file}")

    if not indices_path.exists():
        print(
            f"\n  ERROR: Stage 2 output not found at:\n    {indices_path}\n\n"
            f"  Run Stage 2 first:\n"
            f"    python neuron_viz_pipeline/scripts/stage2_rank.py "
            f"--config {args.config}"
        )
        sys.exit(1)

    assert_dataset_exists(cfg)
    ensure_dirs(cfg, channel_id)

    # ── Load top indices ─────────────────────────────────────────────
    top_indices = np.load(indices_path)
    print(f"\n  top_indices shape: {top_indices.shape} (dtype={top_indices.dtype})")

    # Validate channel_id range
    num_neurons = top_indices.shape[0]
    if not (0 <= channel_id < num_neurons):
        print(
            f"\n  ERROR: channel_id={channel_id} out of range. "
            f"The top_indices file has {num_neurons} neurons (0..{num_neurons-1})."
        )
        sys.exit(1)

    # Determine effective top_k: may be capped by what Stage 2 produced
    stage2_top_k = top_indices.shape[1]
    if top_k > stage2_top_k:
        print(
            f"  NOTE: config.rank.top_k={top_k} > Stage 2 output top_k="
            f"{stage2_top_k}. Using {stage2_top_k} (Stage 2 is the limiter)."
        )
        top_k = stage2_top_k

    # Get this neuron's top-k sample indices
    neuron_indices = top_indices[channel_id, :top_k]
    valid_mask     = neuron_indices != -1
    n_valid        = int(valid_mask.sum())
    n_invalid      = top_k - n_valid
    print(f"  neuron {channel_id}: {n_valid} valid / {n_invalid} padded-with-zeros")

    if n_valid == 0:
        print(f"\n  ERROR: neuron {channel_id} has no valid samples in top-k.")
        sys.exit(1)

    # ── Build model and dataset ──────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  device: {device}")
    model = build_model(cfg, device=device, eval_mode=True)

    # Resolve the target layer module object
    target_layer = get_layer(model, layer_name)
    print(f"  target layer: {layer_name} → {type(target_layer).__name__}")

    print(f"\n  building preprocessed dataset...")
    dataset = build_preprocessed_dataset(cfg)
    print(f"  dataset size: {len(dataset)}")

    # ── Instantiate XAI method ───────────────────────────────────────
    xai_method = get_xai_method(cfg)
    print(f"  method instance: {type(xai_method).__name__}")

    # Fail-fast compatibility check — e.g. AttentionRollout raises if
    # the model is not a ViT (layer_type != 'linear').
    xai_method.validate_config(layer_type)

    # ── Allocate output tensor ───────────────────────────────────────
    # Determine H, W from the first valid image so we only have to look once
    first_valid_rank = int(np.argmax(valid_mask))
    first_idx        = int(neuron_indices[first_valid_rank])
    first_img, _    = dataset[first_idx]
    H, W             = first_img.shape[1], first_img.shape[2]
    print(f"  image spatial shape: {H} × {W}")

    all_maps = np.zeros((top_k, H, W), dtype=np.float32)  # zeros = padding for invalid ranks

    # ── Compute saliency for each valid rank ─────────────────────────
    print(f"\n  computing saliency for {n_valid} images...")
    valid_ranks = []
    start_time = time.time()

    for r in range(top_k):
        sample_idx = int(neuron_indices[r])
        if sample_idx == -1:
            # Padded rank from Stage 2 — skip, leave zeros
            continue

        image, _ = dataset[sample_idx]             # [3, H, W] normalized
        x = image.unsqueeze(0).to(device)           # [1, 3, H, W]

        try:
            saliency = xai_method.compute_saliency(
                model          = model,
                input_tensor   = x,
                target_layer   = target_layer,
                target_channel = channel_id,
                layer_type     = layer_type,
            )
        except Exception as e:
            # Fail hard on the first error — better to see it than to produce
            # silent zeros for a whole batch
            print(f"\n  ERROR at rank {r} (sample {sample_idx}): {e}")
            raise

        # Shape sanity check — first time only, then just the len
        if r == first_valid_rank:
            if saliency.shape != (H, W):
                print(
                    f"\n  ERROR: saliency shape {saliency.shape} does not match "
                    f"expected ({H}, {W}). The XAI method may be misconfigured "
                    f"(e.g. sum_channels=false)."
                )
                sys.exit(1)

        all_maps[r] = saliency
        valid_ranks.append(r)

        # Free per-image memory aggressively — ViT especially benefits
        del x, saliency, image
        if device == "cuda":
            torch.cuda.empty_cache()

        # Progress
        n_done = len(valid_ranks)
        if n_done % 25 == 0 or n_done == n_valid:
            elapsed = time.time() - start_time
            rate    = n_done / elapsed if elapsed > 0 else 0.0
            eta     = (n_valid - n_done) / rate if rate > 0 else 0.0
            print(
                f"    {n_done}/{n_valid}  "
                f"elapsed={elapsed:.1f}s  rate={rate:.2f} img/s  "
                f"eta={eta:.1f}s"
            )

    total_elapsed = time.time() - start_time
    print(f"\n  compute done in {total_elapsed:.1f}s "
          f"({n_valid / total_elapsed:.2f} images/s)")

    # Basic value sanity
    print(f"\n  saliency stats across all valid maps:")
    valid_maps = all_maps[valid_ranks]  # only the non-zero rows
    print(f"    shape        : {all_maps.shape}")
    print(f"    value range  : [{valid_maps.min():.6f}, {valid_maps.max():.6f}]")
    print(f"    mean / std   : {valid_maps.mean():.6f} / {valid_maps.std():.6f}")
    if valid_maps.min() < 0 and xai_method.metadata().get("abs_output", True):
        print(f"    WARNING: abs_output=true but saliency has negative values — "
              f"something is off.")

    # ── Save safetensors + metadata ──────────────────────────────────
    print(f"\n  saving {out_file} ...")
    # The key name in the safetensors file. Consistent with how we name
    # activation files (layer_name as key); for xai_maps we use the method name.
    save_safetensors_streaming(
        layer_name = f"{method_name}_maps",
        arr        = all_maps,
        save_path  = str(out_file),
        chunk_size = 50,  # small file; any chunk size works
    )

    # Metadata
    metadata = {
        "method"          : method_name,
        "model"           : cfg["model"]["name"],
        "layer"           : layer_name,
        "layer_type"      : layer_type,
        "channel_id"      : channel_id,
        "top_k"           : top_k,
        "num_valid"       : n_valid,
        "num_padded"      : n_invalid,
        "valid_ranks"     : valid_ranks,  # which rows of the tensor are non-zero
        "spatial_shape"   : [H, W],
        # Method-specific config (whatever the method wants to record).
        # IxG stores abs_output/sum_channels/vit_include_cls,
        # IG adds n_steps/baseline,
        # AttentionRollout stores include_identity/head_agg/weight_by_neuron/discard_ratio.
        "method_params"   : xai_method.metadata(),
        "value_range"     : [float(valid_maps.min()), float(valid_maps.max())],
        "top_indices_file": str(indices_path),
    }
    meta_path = Path(str(out_file).replace(".safetensors", "_metadata.json"))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata → {meta_path}")

    # Verify
    if not out_file.exists():
        print(f"  ERROR: output file not written: {out_file}")
        sys.exit(1)
    size_mb = out_file.stat().st_size / (1024 ** 2)

    print("\n" + "=" * 70)
    print(f"STAGE 3 DONE — output ({size_mb:.1f} MB):")
    print(f"  {out_file}")
    print(f"  {meta_path}")
    print(f"\n  Stage 4 (crop) runner will be built in Step 6.")
    print(f"  You can inspect the output with:")
    print(f"    python -c \"import numpy as np; import struct, json; "
          f"m = json.load(open('{meta_path}')); "
          f"print('method:', m['method'], 'shape:', m['spatial_shape'], "
          f"'valid:', m['num_valid'])\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
