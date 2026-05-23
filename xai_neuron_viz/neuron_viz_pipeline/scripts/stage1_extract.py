# =============================================================================
# FILE: neuron_viz_pipeline/scripts/stage1_extract.py
#
# Purpose:
#   Thin runner for Stage 1 — activation extraction.
#   Loads config → builds model → builds dataset → runs ActivationExtractor
#   → saves to the path from src/utils/paths.py.
#
# Usage (from the repo root):
#   python neuron_viz_pipeline/scripts/stage1_extract.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml
#
# For a smoke test (200 samples instead of 50 000):
#   python neuron_viz_pipeline/scripts/stage1_extract.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml \
#       --override extract.max_samples=200 \
#       --override extract.checkpoint_interval=5
#
# Output:
#   neuron_viz_pipeline/results/{model}/{xai_method}/{layer}/activations/
#       activations_{layer}_{target_type}_{pool_type}.safetensors
#       activations_{layer}_{target_type}_{pool_type}_metadata.txt
# =============================================================================

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Make our package importable when this script is run directly from the
# repo root. Scripts live at neuron_viz_pipeline/scripts/, and our package root
# is neuron_viz_pipeline/ — so we insert that into sys.path.
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve()
_pkg_root = _here.parent.parent      # neuron_viz_pipeline/
sys.path.insert(0, str(_pkg_root))
# We also need the repo root in sys.path so `from data.data_proces import ...`
# and `from models import build_models` (CoE imports) work.
_repo_root = _pkg_root.parent         # project root
sys.path.insert(0, str(_repo_root))

from src.utils.config import load_config
from src.utils.paths import (
    activations_dir, activations_file, assert_dataset_exists, ensure_dirs,
)
from src.data import build_preprocessed_dataset
from src.models import build_model
from src.extract import ActivationExtractor


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 — Extract activations (neuron_viz_pipeline)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file, e.g. neuron_viz_pipeline/configs/rn152_ixg.yaml",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override a config value with dotted-key syntax. "
             "Repeat the flag to set multiple. "
             "Example: --override extract.batch_size=16 "
             "--override extract.max_samples=200",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply --override k1.k2.k3=VALUE flags to the config.
    VALUE is parsed as YAML (so `10` becomes int, `True` becomes bool, etc.).
    """
    import yaml
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(
                f"Invalid --override '{kv}' — expected format KEY.PATH=VALUE"
            )
        key_path, value_str = kv.split("=", 1)
        # Parse value as YAML so '200' → int, 'true' → bool, etc.
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
    print(f"STAGE 1 — EXTRACT ACTIVATIONS")
    print("=" * 70)
    print(f"  config     : {args.config}")
    print(f"  model      : {cfg['model']['name']}")
    print(f"  layer      : {cfg['model']['layer']}")
    print(f"  target_type: {cfg['extract']['target_type']}")
    print(f"  pool_type  : {cfg['extract']['pool_type']}")
    print(f"  batch_size : {cfg['extract']['batch_size']}")
    print(f"  ckpt_every : {cfg['extract']['checkpoint_interval']} batches")
    if cfg["extract"].get("channel_id_only") is not None:
        print(f"  channel_id_only: {cfg['extract']['channel_id_only']}")
    if cfg["extract"].get("max_samples"):
        print(f"  max_samples: {cfg['extract']['max_samples']} "
              f"(SMOKE-TEST MODE)")
    print()

    # ── Sanity checks ────────────────────────────────────────────────
    assert_dataset_exists(cfg)
    ch = cfg["neuron"]["channel_id"]
    ensure_dirs(cfg, ch)

    out_dir = activations_dir(cfg)
    out_file = activations_file(cfg)
    print(f"  output dir : {out_dir}")
    print(f"  output file: {out_file}")

    if out_file.exists():
        size_gb = out_file.stat().st_size / (1024 ** 3)
        print(
            f"\n  WARNING: output file already exists ({size_gb:.2f} GB).\n"
            f"  Stage 1 will OVERWRITE it. Ctrl-C within 5 seconds to abort."
        )
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted by user.")
            sys.exit(0)

    # ── Build model ──────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  device: {device}")
    model = build_model(cfg, device=device, eval_mode=True)

    # ── Build dataset ────────────────────────────────────────────────
    print(f"\n  building preprocessed dataset...")
    dataset = build_preprocessed_dataset(cfg)
    print(f"  dataset size: {len(dataset)}")

    # Optionally subset for smoke-test runs
    max_samples = cfg["extract"].get("max_samples")
    if max_samples is not None and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
        print(f"  subsetted to {max_samples} samples for smoke test")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["extract"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    # ── Run extraction ───────────────────────────────────────────────
    layer_name  = cfg["model"]["layer"]
    target_type = cfg["extract"]["target_type"]
    pool_type   = cfg["extract"]["pool_type"]

    print(f"\n  registering hook on '{layer_name}' (target_type={target_type})...")
    extractor = ActivationExtractor(model, [(layer_name, target_type)])

    start = time.time()
    try:
        extractor.extract(
            data_loader=data_loader,
            save_dir=str(out_dir),
            save_intermediate=cfg["extract"]["save_intermediate"],
            pool_type=pool_type,
            checkpoint_interval=cfg["extract"]["checkpoint_interval"],
            # Optional single-neuron extraction, e.g.
            # --override extract.channel_id_only=652
            channel_id_only=cfg["extract"].get("channel_id_only"),
        )
    finally:
        extractor.cleanup()

    elapsed = time.time() - start
    print(f"\n  extraction complete in {elapsed / 60:.1f} min")

    # ── Verify output and print summary ──────────────────────────────
    if out_file.exists():
        size_gb = out_file.stat().st_size / (1024 ** 3)
        print(f"  output file: {out_file} ({size_gb:.2f} GB)  ✓")
    else:
        print(f"  WARNING: expected output file NOT found: {out_file}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"STAGE 1 DONE — run Stage 2 next:")
    print(f"  python neuron_viz_pipeline/scripts/stage2_rank.py --config {args.config}")
    print("=" * 70)


if __name__ == "__main__":
    main()
