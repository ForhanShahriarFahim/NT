# =============================================================================
# FILE: neuron_viz_pipeline/scripts/stage4_crop.py
#
# Purpose:
#   Thin runner for Stage 4 — produce cropped/masked images using the IxG
#   saliency maps from Stage 3 as the guide.
#
#   For each of the top-k ranks of cfg.neuron.channel_id:
#     1. Load the original image from dataset (raw, unnormalized)
#     2. Load the IxG saliency map for this rank from Stage 3 output
#     3. Compute crop bbox via get_crop_bbox
#     4. Produce alpha-masked crop (PNG) and/or no-mask crop (JPG)
#     5. Save alongside a JSON info file with run metadata
#
# Preconditions:
#   Stage 3 must have been run so that
#     neuron_viz_pipeline/results/{model}/{xai}/{layer}/xai_maps/neuron_{ccid:03d}/
#       {method}_maps.safetensors
#       {method}_maps_metadata.json
#   exist on disk. (Checked at startup.)
#
# Usage (from the repo root):
#   python neuron_viz_pipeline/scripts/stage4_crop.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml
#
# Smoke test (matches existing 50-rank state):
#   python neuron_viz_pipeline/scripts/stage4_crop.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml \
#       --override rank.top_k=50
#
#   python neuron_viz_pipeline/scripts/stage4_crop.py \
#       --config neuron_viz_pipeline/configs/vit_ixg.yaml \
#       --override rank.top_k=50
#
# Output (path from src/utils/paths.py):
#   neuron_viz_pipeline/results/{model}/{xai}/{layer}/crops/neuron_{ccid:03d}/
#     rank_{r:04d}_sample_{sample_idx}_crop.png                      (masked)
#     rank_{r:04d}_sample_{sample_idx}_crop_info.json                (metadata)
#     rank_{r:04d}_sample_{sample_idx}_crop_without_alpha_mask.jpg   (optional)
#
#   Filename convention matches crop_activation_regions.py main() exactly
#   (lines 1383-1394 of the reference) so outputs are byte-identical to
#   what pipeline_a produces for the same model+layer+neuron.
#
# Design notes:
#   - IxG maps are loaded lazily via numpy memmap — no full-file load, safe
#     for when we scale to top_150 × 50 000 images later
#   - Raw dataset (no normalize) is used here; the IxG map was computed on
#     the normalized tensor, but both variants share the same ImageFolder
#     → sample_idx i refers to the same underlying image
#   - Padding handling: Stage 3 marks invalid ranks in valid_ranks metadata.
#     We skip those ranks rather than trying to crop on an all-zero saliency.
# =============================================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Make our package importable when this script is run from the repo root.
# Mirrors the pattern in scripts/stage1_extract.py .. stage3_xai_maps.py.
# ---------------------------------------------------------------------------
_here       = Path(__file__).resolve()
_pkg_root   = _here.parent.parent         # neuron_viz_pipeline/
sys.path.insert(0, str(_pkg_root))
_repo_root  = _pkg_root.parent            # project root
sys.path.insert(0, str(_repo_root))

from src.utils.config import load_config
from src.utils.paths import (
    top_indices_file, top_values_file,
    xai_maps_file, xai_maps_dir,
    crops_dir, ensure_dirs, assert_dataset_exists,
)
from src.utils.io_safetensors import open_safetensors_memmap
from src.data import build_raw_dataset
from src.crop import (
    get_crop_bbox, crop_and_resize_image,
    create_alpha_mask_crop, create_activation_overlay,
    save_image_with_info,
)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4 — Crop + mask using XAI saliency maps (neuron_viz_pipeline)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file, e.g. neuron_viz_pipeline/configs/rn152_ixg.yaml",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override a config value with dotted-key syntax. "
             "Repeat the flag to set multiple. "
             "Examples: --override rank.top_k=50 "
             "--override crop.threshold_percentile=95",
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Same implementation as the other stage runners — see stage3_xai_maps.py."""
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
# Image conversion helper
# ---------------------------------------------------------------------------

def tensor_to_pil(image_tensor) -> Image.Image:
    """
    Convert a [3, H, W] float tensor in [0, 1] range (from build_raw_dataset)
    into a PIL RGB image.

    Mirrors the conversion done inline in pipeline_a/crop.py / the reference's
    get_cropped_images (lines 554-578 of crop_activation_regions.py):
        - CHW → HWC transpose
        - [0, 1] float → [0, 255] uint8
    """
    arr = image_tensor.numpy() if hasattr(image_tensor, "numpy") else image_tensor
    # CHW → HWC
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    # [0, 1] → [0, 255]
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.override:
        print("Applying overrides:")
        cfg = apply_overrides(cfg, args.override)

    channel_id   = cfg["neuron"]["channel_id"]
    top_k        = cfg["rank"]["top_k"]
    method_name  = cfg["xai"]["method"]
    layer_name   = cfg["model"]["layer"]
    crop_method  = cfg["crop"]["method"]
    threshold_p  = cfg["crop"]["threshold_percentile"]
    padding      = cfg["crop"]["padding"]
    alpha_mask   = cfg["crop"]["alpha_mask"]
    mask_threshold = cfg["crop"]["mask_threshold"]
    crop_size    = cfg["crop"]["crop_size"]
    save_no_mask = cfg["crop"]["save_no_mask"]
    save_overlay = cfg["crop"]["save_overlay"]

    print("\n" + "=" * 70)
    print(f"STAGE 4 — CROP + MASK USING XAI SALIENCY MAPS")
    print("=" * 70)
    print(f"  config               : {args.config}")
    print(f"  model                : {cfg['model']['name']}")
    print(f"  layer                : {layer_name}")
    print(f"  xai method           : {method_name}")
    print(f"  target_channel       : {channel_id}")
    print(f"  top_k                : {top_k}")
    print(f"  crop_method          : {crop_method}")
    print(f"  threshold_percentile : {threshold_p}")
    print(f"  padding              : {padding}")
    print(f"  alpha_mask           : {alpha_mask}")
    if alpha_mask:
        print(f"  mask_threshold       : {mask_threshold}")
    print(f"  crop_size            : {crop_size}")
    print(f"  save_no_mask         : {save_no_mask}")
    print(f"  save_overlay         : {save_overlay}")

    # ── Paths ────────────────────────────────────────────────────────
    indices_path = top_indices_file(cfg)
    values_path  = top_values_file(cfg)
    maps_path    = xai_maps_file(cfg, channel_id)
    maps_meta    = Path(str(maps_path).replace(".safetensors", "_metadata.json"))
    out_dir      = crops_dir(cfg, channel_id)

    print(f"\n  input top_indices : {indices_path}")
    print(f"  input top_values  : {values_path}")
    print(f"  input xai_maps    : {maps_path}")
    print(f"  input metadata    : {maps_meta}")
    print(f"  output dir        : {out_dir}")

    # ── Preconditions ────────────────────────────────────────────────
    if not indices_path.exists():
        print(f"\n  ERROR: Stage 2 output not found:\n    {indices_path}")
        sys.exit(1)
    if not maps_path.exists():
        print(
            f"\n  ERROR: Stage 3 output not found:\n    {maps_path}\n\n"
            f"  Run Stage 3 first:\n"
            f"    python neuron_viz_pipeline/scripts/stage3_xai_maps.py "
            f"--config {args.config}"
        )
        sys.exit(1)

    assert_dataset_exists(cfg)
    ensure_dirs(cfg, channel_id)

    # ── Load Stage 2 top indices and values ──────────────────────────
    top_indices = np.load(indices_path)
    print(f"\n  top_indices shape: {top_indices.shape}")

    neuron_values = None
    if values_path.exists():
        top_values = np.load(values_path)
        neuron_values = top_values[channel_id]  # [top_k] activation scores
        print(f"  top_values loaded for activation_value in info JSON")
    else:
        print(f"  top_values file not found — activation_value will be null in info JSON")

    # Validate channel_id range
    num_neurons = top_indices.shape[0]
    if not (0 <= channel_id < num_neurons):
        print(
            f"\n  ERROR: channel_id={channel_id} out of range. "
            f"top_indices has {num_neurons} neurons (0..{num_neurons-1})."
        )
        sys.exit(1)

    stage2_top_k = top_indices.shape[1]
    if top_k > stage2_top_k:
        print(
            f"  NOTE: config.rank.top_k={top_k} > Stage 2 output top_k={stage2_top_k}. "
            f"Using {stage2_top_k}."
        )
        top_k = stage2_top_k

    neuron_indices = top_indices[channel_id, :top_k]

    # ── Load Stage 3 metadata (for valid_ranks) ──────────────────────
    with open(maps_meta) as f:
        stage3_meta = json.load(f)
    valid_ranks = set(stage3_meta.get("valid_ranks", list(range(top_k))))
    stage3_top_k = stage3_meta.get("top_k", top_k)
    spatial_shape = stage3_meta.get("spatial_shape", None)
    print(f"\n  Stage 3 metadata: top_k={stage3_top_k}, "
          f"valid_ranks_count={len(valid_ranks)}, spatial_shape={spatial_shape}")

    if stage3_top_k < top_k:
        print(
            f"  NOTE: Stage 3 top_k={stage3_top_k} < config top_k={top_k}. "
            f"Using {stage3_top_k}."
        )
        top_k = stage3_top_k
        neuron_indices = top_indices[channel_id, :top_k]

    # ── Open IxG maps as memmap (lazy; one rank at a time) ────────────
    print(f"\n  opening IxG maps via lazy memmap...")
    maps_key = f"{method_name}_maps"   # how stage3 saved it
    maps_mm, resolved_key = open_safetensors_memmap(str(maps_path), layer_name=maps_key)
    print(f"  saved key in file : '{resolved_key}'")
    print(f"  memmap shape      : {maps_mm.shape}  dtype={maps_mm.dtype}")

    # Sanity: first dim should match top_k
    if maps_mm.shape[0] != stage3_top_k:
        print(
            f"  WARNING: IxG maps first dim ({maps_mm.shape[0]}) != stage3 top_k "
            f"({stage3_top_k}) — metadata and tensor disagree."
        )

    # ── Build raw dataset (for displayable pixels) ────────────────────
    print(f"\n  building raw dataset (no normalization)...")
    dataset = build_raw_dataset(cfg)
    print(f"  dataset size: {len(dataset)}")

    # ── Process each rank ────────────────────────────────────────────
    print(f"\n  processing {top_k} ranks (skipping {top_k - len(valid_ranks)} padded)...")
    processed = 0
    skipped_padded = 0
    errors = 0
    start_time = time.time()

    for r in range(top_k):
        if r not in valid_ranks:
            skipped_padded += 1
            continue

        sample_idx = int(neuron_indices[r])
        if sample_idx == -1:
            # Shouldn't happen if valid_ranks is correct, but double-guard
            skipped_padded += 1
            continue

        try:
            # ── Load original image ──────────────────────────────────
            image_tensor, label = dataset[sample_idx]
            image = tensor_to_pil(image_tensor)
            # image is PIL RGB. image.size is (W, H).

            # ── Load IxG saliency for this rank ──────────────────────
            # memmap read → numpy array [H, W] float32
            saliency = np.asarray(maps_mm[r], dtype=np.float32)

            # ── Compute crop bbox ────────────────────────────────────
            bbox = get_crop_bbox(
                saliency,
                method = crop_method,
                threshold_percentile = threshold_p,
                padding = padding,
            )
            # bbox is (x1, y1, x2, y2)

            # ── Crop with alpha mask (primary output) ────────────────
            if alpha_mask:
                cropped_masked = create_alpha_mask_crop(
                    image, saliency, bbox, crop_size, mask_threshold
                )
                # Primary file: PNG
                crop_filename = f"rank_{r:04d}_sample_{sample_idx}_crop.png"
            else:
                # If alpha_mask is off, the "primary" is the plain crop
                cropped_masked = crop_and_resize_image(image, bbox, crop_size)
                crop_filename = f"rank_{r:04d}_sample_{sample_idx}_crop.jpg"

            # ── Build info JSON ──────────────────────────────────────
            info = {
                # Fields in the same order as reference's main() (lines 1367-1381)
                "neuron_idx"            : int(channel_id),
                "sample_idx"            : int(sample_idx),
                "rank"                  : int(r),
                "activation_value"      : (
                    float(neuron_values[r]) if neuron_values is not None else None
                ),
                "label"                 : (
                    int(label) if isinstance(label, (int, np.integer)) else str(label)
                ),
                "layer_name"            : layer_name,
                "crop_bbox"             : [int(x) for x in bbox],
                "crop_method"           : crop_method,   # always spatial for IxG
                "has_spatial_activations": True,         # IxG always produces spatial maps
                "alpha_mask"            : alpha_mask,
                "mask_threshold"        : mask_threshold if alpha_mask else None,
                "original_size"         : [int(x) for x in image.size],
                "cropped_size"          : [crop_size, crop_size],
                # ── XAI context (NEW — not in original reference) ─────
                "xai_method"            : method_name,
                "saliency_source"       : str(maps_path),
            }

            # ── Save primary + info JSON ─────────────────────────────
            crop_path = out_dir / crop_filename
            save_image_with_info(cropped_masked, str(crop_path), info)

            # ── Save no-mask variant if requested ────────────────────
            if save_no_mask and alpha_mask:
                # Only meaningful when alpha_mask was applied above — we save
                # the plain crop as a companion for comparison.
                # Filename matches reference (crop_activation_regions.py line 1392)
                no_mask_filename = (
                    f"rank_{r:04d}_sample_{sample_idx}_crop_without_alpha_mask.jpg"
                )
                no_mask_crop = crop_and_resize_image(image, bbox, crop_size)
                (out_dir / no_mask_filename).parent.mkdir(parents=True, exist_ok=True)
                no_mask_crop.save(str(out_dir / no_mask_filename))

            # ── Save overlay if requested ────────────────────────────
            if save_overlay:
                overlay = create_activation_overlay(image, saliency)
                overlay_filename = f"rank_{r:04d}_sample_{sample_idx}_overlay.jpg"
                overlay.save(str(out_dir / overlay_filename))

                cropped_overlay = crop_and_resize_image(overlay, bbox, crop_size)
                cropped_overlay_filename = (
                    f"rank_{r:04d}_sample_{sample_idx}_crop_overlay.jpg"
                )
                cropped_overlay.save(str(out_dir / cropped_overlay_filename))

            processed += 1

            # Progress every 25 items
            if processed % 25 == 0 or processed == len(valid_ranks):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (len(valid_ranks) - processed) / rate if rate > 0 else 0.0
                print(
                    f"    {processed}/{len(valid_ranks)}  "
                    f"elapsed={elapsed:.1f}s  rate={rate:.2f} img/s  "
                    f"eta={eta:.1f}s"
                )

        except Exception as e:
            print(f"\n  ERROR at rank {r} (sample {sample_idx}): {e}")
            errors += 1
            # Continue with next rank rather than failing whole run — the
            # reference code (crop_activation_regions.py main() line 1409)
            # does the same thing.
            continue

    total_elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"STAGE 4 DONE")
    print("=" * 70)
    print(f"  processed : {processed}")
    print(f"  skipped   : {skipped_padded}  (padded / invalid ranks)")
    print(f"  errors    : {errors}")
    print(f"  elapsed   : {total_elapsed:.1f}s "
          f"({processed / total_elapsed:.2f} img/s)" if total_elapsed > 0 else "")
    print(f"  output    : {out_dir}")
    print(f"\n  Sample output files (first 3):")
    files = sorted(out_dir.iterdir())[:6]
    for f in files:
        print(f"    {f.name}")
    print(f"\n  Step 7 will add scripts/make_collage.py to build Collage_Part_*.jpg "
          f"from these crops.")


if __name__ == "__main__":
    main()
