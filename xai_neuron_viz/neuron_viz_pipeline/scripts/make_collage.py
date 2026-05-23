# =============================================================================
# FILE: neuron_viz_pipeline/scripts/make_collage.py
#
# Purpose:
#   Build collage grids from the per-neuron crop images produced by Stage 4.
#   Mirrors the visual layout of the original collage utility.
#   (3×5 grid, Collage_Part_NN.jpg, figsize=(15,9), dpi=150) so outputs from
#   this new pipeline look identical to pipeline_a/b collages that Fahim
#   is already used to reading.
#
# Relationship to the original make_collage.py:
#   The original supported three modes: --scan, CoE path construction, and
#   --direct_dir. This port keeps the grid rendering logic VERBATIM from
#   that file (lines that build the 3×5 figure, set titles, save JPG, and
#   handle missing-file fallbacks) and drops the modes we don't need here:
#
#   KEPT (verbatim):
#     - 3×5 plt.subplots(figsize=(15,9))
#     - per-tile title "Rank {global_rank + 1}"
#     - figure title "Channel {ch} | Ranks {start+1}–{end+1} | {label}"
#     - wspace=0.02, hspace=0.15, bbox_inches='tight', dpi=150
#     - Collage_Part_{i+1:02d}.jpg filename (1-indexed)
#     - missing-image fallback text behaviour
#     - rank_* filename detection: starts with 'rank_' AND has '_crop.png'
#       or '_crop.jpg' AND excludes _info / _overlay / _without_alpha_mask
#
#   REMOVED:
#     - --scan mode (our paths are deterministic from config)
#     - CoE path construction (we only generate our layout)
#     - Pipeline B xai-marker detection (we use cfg.xai.method as the label)
#
#   ADDED:
#     - --config mode: reads crops_dir + collages_dir from src/utils/paths.py
#     - auto-clamp: if found_crops < total_images, clamp to found_crops
#       and print a notice (matches Fahim's decision)
#
# The original reference excluded files containing '_no_mask' (from
# pipeline_a/b). Our Stage 4 produces '_crop_without_alpha_mask.jpg'
# (which does NOT contain '_crop.png' or '_crop.jpg' as a substring because
# '_crop' is followed by '_without', not by a file extension), so the
# original filter passes these files through unchanged by coincidence of
# string matching — BUT to be defensive and explicit, we also exclude
# '_without_alpha_mask' as a third keyword. Confirmed empirically before
# writing.
#
# Usage (from the repo root):
#
#   # Config-driven (most common — uses cfg.neuron.channel_id)
#   python neuron_viz_pipeline/scripts/make_collage.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml
#
#   # Direct-directory mode (for ad-hoc comparisons)
#   python neuron_viz_pipeline/scripts/make_collage.py \
#       --direct_dir neuron_viz_pipeline/results/rn152/ixg/layer4_2_conv3/crops/neuron_015 \
#       --output_dir neuron_viz_pipeline/results/rn152/ixg/layer4_2_conv3/collages/neuron_015 \
#       --channel_id 15
#
#   # Override total_images on the fly
#   python neuron_viz_pipeline/scripts/make_collage.py \
#       --config neuron_viz_pipeline/configs/rn152_ixg.yaml \
#       --override collage.total_images=50
# =============================================================================

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Make our package importable when this script is run from the repo root.
# Mirrors stage1..stage4.
# ---------------------------------------------------------------------------
_here      = Path(__file__).resolve()
_pkg_root  = _here.parent.parent          # neuron_viz_pipeline/
sys.path.insert(0, str(_pkg_root))
_repo_root = _pkg_root.parent             # project root
sys.path.insert(0, str(_repo_root))

from src.utils.config import load_config
from src.utils.paths  import crops_dir, collages_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build collage grids from Stage 4 crop images. "
            "Use --config OR --direct_dir."
        )
    )

    # Config-driven mode
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to YAML config, e.g. neuron_viz_pipeline/configs/rn152_ixg.yaml. "
            "The neuron, model, layer, and method determine the crops/collages "
            "directories automatically."
        ),
    )

    # Direct-directory mode (ad-hoc)
    parser.add_argument(
        "--direct_dir", type=str, default=None,
        help=(
            "Folder containing the crop images. Bypasses config-driven path "
            "construction. Requires --output_dir and --channel_id."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save collage JPGs. Required with --direct_dir.",
    )
    parser.add_argument(
        "--channel_id", type=int, default=None,
        help=(
            "Channel/neuron id used in figure titles. "
            "In --config mode this is read from cfg.neuron.channel_id."
        ),
    )

    # Common tuning (override config defaults)
    parser.add_argument(
        "--total_images", type=int, default=None,
        help="Override cfg.collage.total_images (default from config)",
    )
    parser.add_argument(
        "--images_per_grid", type=int, default=None,
        help="Override cfg.collage.images_per_grid (default from config)",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help=(
            "Override a config value with dotted-key syntax (config mode only). "
            "Example: --override collage.total_images=50"
        ),
    )
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Same implementation as the stage runners — see stage3_xai_maps.py."""
    import yaml
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(
                f"Invalid --override '{kv}' — expected KEY.PATH=VALUE"
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
# File discovery — identical filter logic to the reference make_collage.py
# (lines 256-264 of the uploaded file), with one additional defensive
# exclusion for '_without_alpha_mask' since that's what Stage 4 produces.
# ---------------------------------------------------------------------------

def find_crop_files(input_dir: Path) -> List[str]:
    """
    Return sorted list of primary crop filenames (alpha-masked PNG or plain JPG).

    Exclusion rules (matches reference file):
      - must start with 'rank_'
      - must contain '_crop.png' or '_crop.jpg' (the primary crop)
      - excludes '_without_alpha_mask' (our no-mask companion)
      - excludes '_no_mask'   (pipeline_a/b companion — kept for future-proof)
      - excludes '_overlay'   (heatmap overlay)
      - excludes '_info'      (JSON metadata)

    Zero-padded rank names (rank_0000, rank_0001, ...) sort correctly by
    lexicographic order, so plain sorted() is safe.
    """
    if not input_dir.is_dir():
        return []
    files = os.listdir(input_dir)
    filtered = sorted(
        f for f in files
        if f.startswith("rank_")
        and ("_crop.png" in f or "_crop.jpg" in f)
        and "_without_alpha_mask" not in f
        and "_no_mask" not in f
        and "_overlay"  not in f
        and "_info"     not in f
    )
    return filtered


# ---------------------------------------------------------------------------
# Collage rendering
# Grid / layout / filename conventions are VERBATIM from the reference
# make_collage.py create_collages() function (pipeline A/B branch).
# ---------------------------------------------------------------------------

def render_collages(
    input_dir: Path,
    output_dir: Path,
    channel_id: int,
    total_images: int,
    images_per_grid: int,
    source_label: str,
) -> int:
    """
    Render collages. Returns the number of collage files written.

    Input filename contract:
        rank_{r:04d}_sample_{sample_idx}_crop.png   (alpha-masked)
        rank_{r:04d}_sample_{sample_idx}_crop.jpg   (non-masked primary)

    Layout:
        - 3 rows × 5 columns (15 tiles per collage)
        - figsize=(15, 9), dpi=150
        - missing tiles show a grey "Rank N not found" placeholder
    """
    rank_files = find_crop_files(input_dir)

    if not rank_files:
        print(
            f"\n  ERROR: no crop files found in {input_dir}\n"
            f"  Expected: rank_XXXX_sample_YYYY_crop.png  (or .jpg)\n"
            f"  Did Stage 4 run successfully for this neuron?"
        )
        return 0

    found_n = len(rank_files)
    print(f"\n  crops found       : {found_n} files")
    print(f"  source            : {input_dir}")

    # ── Auto-clamp total_images to found count (Fahim's decision) ────────
    if found_n < total_images:
        print(
            f"  NOTE: only {found_n} crops on disk, config asked for "
            f"{total_images}. Clamping total_images to {found_n} "
            f"(fewer collages → no empty tiles)."
        )
        total_images = found_n

    # ── Compute number of collages ───────────────────────────────────────
    # Round up so a trailing partial grid still renders (e.g. 50 crops
    # with images_per_grid=15 → 4 collages, the last one with 5 tiles).
    num_collages = (total_images + images_per_grid - 1) // images_per_grid
    print(f"  collages to write : {num_collages}  "
          f"(images_per_grid={images_per_grid})")
    print(f"  output            : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pad list with None so partial last-grid draws correctly ──────────
    # Reference uses while-loop; we pad to num_collages * images_per_grid
    # exactly to avoid any out-of-range issues.
    target_len = num_collages * images_per_grid
    while len(rank_files) < target_len:
        rank_files.append(None)

    print("  " + "-" * 58)

    # ── Render each collage ──────────────────────────────────────────────
    for collage_idx in range(num_collages):
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))

        start_rank = collage_idx * images_per_grid
        end_rank   = start_rank + images_per_grid - 1

        fig.suptitle(
            f"Channel {channel_id} | "
            f"Ranks {start_rank + 1}\u2013{end_rank + 1} | {source_label}",
            fontsize=16, fontweight="bold", y=1.02,
        )

        for tile_i, ax in enumerate(axes.flatten()):
            global_rank = start_rank + tile_i
            fname = rank_files[global_rank] if global_rank < len(rank_files) else None

            if fname is not None:
                img_path = input_dir / fname
                if img_path.exists():
                    try:
                        ax.imshow(Image.open(img_path).convert("RGB"))
                    except Exception:
                        ax.text(0.5, 0.5, "Read\nerror",
                                ha="center", va="center",
                                color="orange", fontsize=9)
                else:
                    ax.text(0.5, 0.5, "Missing",
                            ha="center", va="center",
                            color="red", fontsize=9)
            else:
                ax.text(0.5, 0.5, f"Rank {global_rank + 1}\nnot found",
                        ha="center", va="center",
                        color="gray", fontsize=8)

            ax.set_title(f"Rank {global_rank + 1}", fontsize=10)
            ax.axis("off")

        plt.subplots_adjust(wspace=0.02, hspace=0.15)

        save_path = output_dir / f"Collage_Part_{collage_idx + 1:02d}.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {save_path.name}")

    return num_collages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.config is None and args.direct_dir is None:
        print("\nERROR: provide either --config or --direct_dir.\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 7 / MAKE_COLLAGE")
    print("=" * 70)

    # ── Config-driven mode ────────────────────────────────────────────────
    if args.config is not None:
        cfg = load_config(args.config)
        if args.override:
            print("Applying overrides:")
            cfg = apply_overrides(cfg, args.override)

        # Channel id
        channel_id = (
            args.channel_id
            if args.channel_id is not None
            else cfg["neuron"]["channel_id"]
        )

        # Input / output dirs from paths.py
        input_dir  = crops_dir(cfg, channel_id)
        output_dir = collages_dir(cfg, channel_id)

        # Collage sizing — CLI flags take priority over config
        total_images    = (
            args.total_images
            if args.total_images is not None
            else cfg["collage"]["total_images"]
        )
        images_per_grid = (
            args.images_per_grid
            if args.images_per_grid is not None
            else cfg["collage"]["images_per_grid"]
        )

        # Source label in figure titles = the XAI method name
        source_label = cfg["xai"]["method"].upper()

        print(f"  mode              : config")
        print(f"  config            : {args.config}")
        print(f"  model             : {cfg['model']['name']}")
        print(f"  layer             : {cfg['model']['layer']}")
        print(f"  xai method        : {cfg['xai']['method']}")
        print(f"  channel_id        : {channel_id}")

    # ── Direct-directory mode ────────────────────────────────────────────
    else:
        if args.output_dir is None or args.channel_id is None:
            print(
                "\nERROR: --direct_dir requires --output_dir and --channel_id.\n"
                "Example:\n"
                "  python neuron_viz_pipeline/scripts/make_collage.py \\\n"
                "      --direct_dir /path/to/crops/neuron_015 \\\n"
                "      --output_dir /path/to/collages/neuron_015 \\\n"
                "      --channel_id 15"
            )
            sys.exit(1)

        input_dir  = Path(args.direct_dir).resolve()
        output_dir = Path(args.output_dir).resolve()
        channel_id = args.channel_id

        total_images    = args.total_images    if args.total_images    is not None else 150
        images_per_grid = args.images_per_grid if args.images_per_grid is not None else 15
        source_label    = "CROPS"

        print(f"  mode              : direct_dir")
        print(f"  channel_id        : {channel_id}")

    # ── Render ───────────────────────────────────────────────────────────
    n = render_collages(
        input_dir       = Path(input_dir),
        output_dir      = Path(output_dir),
        channel_id      = channel_id,
        total_images    = total_images,
        images_per_grid = images_per_grid,
        source_label    = source_label,
    )

    print("\n" + "=" * 70)
    if n > 0:
        print(f"DONE. Wrote {n} collage(s) to:")
        print(f"  {output_dir}")
    else:
        print(f"DONE. No collages written.")
    print("=" * 70)


if __name__ == "__main__":
    main()
