# =============================================================================
# FILE: neuron_viz_pipeline/src/utils/paths.py
#
# Purpose:
#   Single source of truth for every path the pipeline writes to or reads
#   from. Every stage calls one of these helpers instead of hardcoding paths.
#
# This file has two responsibilities:
#   1. Build OUTPUT paths (results/, crops/, xai_maps/, ...) from the config
#   2. Resolve the INPUT dataset path and validate it exists on disk
#
# Filename patterns are verified against the original reference files:
#   preprocessing/extract_activations.py      (→ _save_layer_by_layer,
#                                                _save_safetensors_streaming)
#   preprocessing/compute_top_activations.py  (→ save_results)
# and against your pipeline_a/{extract,rank,crop}.py where you already
# adapted them. Keeping the patterns identical means outputs of this
# pipeline are directly comparable to pipeline_a / pipeline_b outputs
# for the same model+layer.
#
# Resolution rule (matches CoE's resolved model names from main.py __main__):
#   "vit" → "vit_base_patch16_224"   (timm's canonical name)
#   all others unchanged
#
# Dataset path policy:
#   Dataset is shared at
#   <repo_root>/dataset/imagenet/val/. We do not copy or symlink it.
#   data.path in configs is "./dataset" and all scripts are run from the
#   repo root.
#
# Directory layout produced:
#   {results_root}/{resolved_model}/{xai_method}/{layer_slug}/
#       activations/
#         activations_{layer_slug}_{target_type}_{pool_type}.safetensors
#         activations_{layer_slug}_{target_type}_{pool_type}_metadata.txt
#       top_k/
#         top_activations_{layer_slug}_{target_type}_indices.npy
#         top_activations_{layer_slug}_{target_type}_values.npy
#         top_activations_{layer_slug}_{target_type}_metadata.json
#       xai_maps/
#         neuron_{channel_id:03d}/
#           {xai_method}_maps.safetensors
#       crops/
#         neuron_{channel_id:03d}/
#           rank_{r:04d}_sample_{idx}_crop.png
#           rank_{r:04d}_sample_{idx}_crop_without_alpha_mask.jpg
#           rank_{r:04d}_sample_{idx}_crop_info.json
#       collages/
#         neuron_{channel_id:03d}/
#           Collage_Part_01.jpg ... Collage_Part_10.jpg
# =============================================================================

from pathlib import Path
from typing import Any, Dict


# Resolve short model names to canonical names used everywhere.
# Matches the mapping main.py __main__ uses in your existing CoE setup:
#   elif args.resume == 'vit':
#       args.model_name = 'vit_base_patch16_224'
# so output folders align with what you already have under output/ and results/.
MODEL_NAME_MAP = {
    "vit": "vit_base_patch16_224",
}


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------

def resolve_model_name(model_name: str) -> str:
    """`vit` → `vit_base_patch16_224`; others unchanged."""
    return MODEL_NAME_MAP.get(model_name, model_name)


def layer_to_slug(layer_name: str) -> str:
    """
    Convert a dotted layer name to a filesystem-safe slug.
    Matches the transform used by _save_layer_by_layer in extract.py:
        safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
    Examples:
        layer4.2.conv3     → layer4_2_conv3
        blocks.11.mlp.fc2  → blocks_11_mlp_fc2
    """
    return layer_name.replace(".", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Dataset path resolution & validation
# ---------------------------------------------------------------------------

def dataset_val_dir(cfg: Dict[str, Any]) -> Path:
    """
    Resolve the full path to the ImageFolder root that torchvision will read.

    Example:
        data.path      = "./dataset"
        data.dataset   = "imagenet-val"
        returns           Path("./dataset/imagenet/val")

    This reproduces the logic inside data/data_proces.py:
        root    = os.path.join(data_path, dataset_name)   # "./dataset/imagenet"
        val_dir = os.path.join(root, dataset_split)        # "./dataset/imagenet/val"
    """
    data_path       = cfg["data"]["path"]
    dataset_spec    = cfg["data"]["dataset"]          # e.g. "imagenet-val"
    dataset_name    = dataset_spec.split("-")[0]       # "imagenet"
    dataset_split   = dataset_spec.split("-")[1]       # "val"
    return Path(data_path) / dataset_name / dataset_split


def assert_dataset_exists(cfg: Dict[str, Any]) -> None:
    """
    Raise a clear error if the dataset directory is missing or empty.
    Called at the start of every pipeline stage so misconfiguration is
    caught immediately, not 10 seconds into a forward pass.
    """
    val_dir = dataset_val_dir(cfg).resolve()
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at: {val_dir}\n"
            f"  data.path    = {cfg['data']['path']!r}  (resolved from cwd={Path.cwd()})\n"
            f"  data.dataset = {cfg['data']['dataset']!r}\n"
            f"\n"
            f"Expected layout: <data.path>/imagenet/val/<1000 class folders>/<images>\n"
            f"\n"
            f"Fixes:\n"
            f"  1. Run scripts from the project root:\n"
            f"       cd NeuronTree-AI/Task-12/xai_neuron_viz\n"
            f"       python neuron_viz_pipeline/scripts/run_stage.py \\\n"
            f"           --config neuron_viz_pipeline/configs/rn152_ixg.yaml \\\n"
            f"           --stage all\n"
            f"  2. Or override with an absolute path:\n"
            f"       --override data.path=/absolute/path/to/dataset"
        )
    # Quick sanity check — at least one class folder should exist
    subdirs = [p for p in val_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 0:
        raise FileNotFoundError(
            f"Dataset directory exists but is empty: {val_dir}\n"
            f"Expected 1000 class folders (n01440764, n01443537, ...)"
        )


# ---------------------------------------------------------------------------
# Base directory (shared by every stage)
# ---------------------------------------------------------------------------

def _base_dir(cfg: Dict[str, Any]) -> Path:
    """
    Root of all results for this (model, xai_method, layer) combination.
    Example: neuron_viz_pipeline/results/rn152/ixg/layer4_2_conv3/
    """
    return (
        Path(cfg["results_root"])
        / resolve_model_name(cfg["model"]["name"])
        / cfg["xai"]["method"]
        / layer_to_slug(cfg["model"]["layer"])
    )


# ---------------------------------------------------------------------------
# Stage 1 — Activations
# Filename pattern from extract_activations.py / pipeline_a/extract.py:
#   activations_{safe_layer_name}_{target_type}_{pool_type}.safetensors
# ---------------------------------------------------------------------------

def activations_dir(cfg: Dict[str, Any]) -> Path:
    return _base_dir(cfg) / "activations"


def activations_file(cfg: Dict[str, Any]) -> Path:
    """
    Final .safetensors file written by Stage 1.
    Name matches _save_layer_by_layer / _save_safetensors_streaming in
    extract.py exactly:
        f"activations_{safe_layer_name}_{target_type}_{pool_type}.safetensors"
    """
    slug        = layer_to_slug(cfg["model"]["layer"])
    pool_type   = cfg["extract"]["pool_type"]
    target_type = cfg["extract"]["target_type"]
    fname = f"activations_{slug}_{target_type}_{pool_type}.safetensors"
    return activations_dir(cfg) / fname


def activations_metadata_file(cfg: Dict[str, Any]) -> Path:
    """
    Companion metadata .txt file.
    Pattern from extract.py: save_path.replace(".safetensors", "_metadata.txt")
    produces:
        activations_{layer_slug}_{target_type}_{pool_type}_metadata.txt
    """
    safetensors_path = activations_file(cfg)
    # Reproduce the exact string replacement used in the original code
    # to guarantee byte-identical filenames.
    return Path(str(safetensors_path).replace(".safetensors", "_metadata.txt"))


# ---------------------------------------------------------------------------
# Stage 2 — Top-k rankings
# Filename pattern from compute_top_activations.py / pipeline_a/rank.py:
#   top_activations_{safe_layer_name}{target_type_suffix}_indices.npy
#   where target_type_suffix = "_output" or "_input"
# ---------------------------------------------------------------------------

def top_k_dir(cfg: Dict[str, Any]) -> Path:
    return _base_dir(cfg) / "top_k"


def _target_type_suffix(cfg: Dict[str, Any]) -> str:
    """
    Matches rank.py save_results logic:
        if "_input_"  in fn: target_type_suffix = "_input"
        elif "_output_" in fn: target_type_suffix = "_output"
    Since we control the config explicitly, we just use the value directly.
    """
    return f"_{cfg['extract']['target_type']}"


def top_indices_file(cfg: Dict[str, Any]) -> Path:
    slug   = layer_to_slug(cfg["model"]["layer"])
    suffix = _target_type_suffix(cfg)
    return top_k_dir(cfg) / f"top_activations_{slug}{suffix}_indices.npy"


def top_values_file(cfg: Dict[str, Any]) -> Path:
    slug   = layer_to_slug(cfg["model"]["layer"])
    suffix = _target_type_suffix(cfg)
    return top_k_dir(cfg) / f"top_activations_{slug}{suffix}_values.npy"


def top_k_metadata_file(cfg: Dict[str, Any]) -> Path:
    slug   = layer_to_slug(cfg["model"]["layer"])
    suffix = _target_type_suffix(cfg)
    return top_k_dir(cfg) / f"top_activations_{slug}{suffix}_metadata.json"


# ---------------------------------------------------------------------------
# Stage 3 — XAI saliency maps (NEW — not in reference files)
# Filename pattern (our own convention):
#   {xai_method}_maps.safetensors   inside neuron_{id:03d}/ folder
# ---------------------------------------------------------------------------

def xai_maps_dir(cfg: Dict[str, Any], channel_id: int) -> Path:
    return _base_dir(cfg) / "xai_maps" / f"neuron_{channel_id:03d}"


def xai_maps_file(cfg: Dict[str, Any], channel_id: int) -> Path:
    """
    One safetensors file per neuron containing [top_k, H, W] saliency maps.
    Method is encoded in the filename so multiple methods can coexist
    for the same neuron (e.g. ixg_maps.safetensors, ig_maps.safetensors).
    """
    method = cfg["xai"]["method"]
    fname = f"{method}_maps.safetensors"
    return xai_maps_dir(cfg, channel_id) / fname


# ---------------------------------------------------------------------------
# Stage 4 — Cropped / masked images
# Filename pattern from crop_activation_regions.py main() (lines 1385, 1387,
# 1392) / pipeline_a/crop.py:
#   rank_{r:04d}_sample_{sample_idx}_crop.png                    (alpha-masked)
#   rank_{r:04d}_sample_{sample_idx}_crop.jpg                    (if alpha_mask=False)
#   rank_{r:04d}_sample_{sample_idx}_crop_without_alpha_mask.jpg (optional companion)
#   rank_{r:04d}_sample_{sample_idx}_crop_info.json              (JSON metadata —
#       save_image_with_info replaces .png/.jpg with _info.json)
# ---------------------------------------------------------------------------

def crops_dir(cfg: Dict[str, Any], channel_id: int) -> Path:
    return _base_dir(cfg) / "crops" / f"neuron_{channel_id:03d}"


# ---------------------------------------------------------------------------
# Collages
# Filename pattern from your make_collage.py:
#   Collage_Part_{i:02d}.jpg
# ---------------------------------------------------------------------------

def collages_dir(cfg: Dict[str, Any], channel_id: int) -> Path:
    return _base_dir(cfg) / "collages" / f"neuron_{channel_id:03d}"


# ---------------------------------------------------------------------------
# Bulk creation helper
# ---------------------------------------------------------------------------

def ensure_dirs(cfg: Dict[str, Any], channel_id: int) -> None:
    """
    Create every output directory we will need for this run.
    Safe to call multiple times (idempotent — uses exist_ok=True).
    """
    for path in [
        activations_dir(cfg),
        top_k_dir(cfg),
        xai_maps_dir(cfg, channel_id),
        crops_dir(cfg, channel_id),
        collages_dir(cfg, channel_id),
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Smoke test — run from repo root:
#   python -m neuron_viz_pipeline.src.utils.paths
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running from either repo root or inside neuron_viz_pipeline/
    import sys
    from pathlib import Path as _Path
    here = _Path(__file__).resolve()
    # Try inserting parent neuron_viz_pipeline/ for `from src.utils.config import ...`
    for up in (3, 2):
        cand = here.parents[up - 1] if up <= len(here.parents) else None
        if cand and (cand / "configs" / "base.yaml").is_file():
            sys.path.insert(0, str(cand))
            break

    from src.utils.config import load_config

    # Resolve config paths relative to cwd
    configs_dir = _Path("configs")
    if not configs_dir.is_dir():
        configs_dir = _Path("neuron_viz_pipeline/configs")

    for config_name in (configs_dir / "rn152_ixg.yaml",
                        configs_dir / "vit_ixg.yaml"):
        if not config_name.is_file():
            continue
        print(f"\n=== Paths for {config_name} ===")
        cfg = load_config(str(config_name))
        ch = cfg["neuron"]["channel_id"]

        print(f"  resolved_model       : {resolve_model_name(cfg['model']['name'])}")
        print(f"  layer_slug           : {layer_to_slug(cfg['model']['layer'])}")
        print(f"  target_type_suffix   : {_target_type_suffix(cfg)}")
        print(f"  base_dir             : {_base_dir(cfg)}")
        print(f"  dataset_val_dir      : {dataset_val_dir(cfg)}")
        print(f"                       → {dataset_val_dir(cfg).resolve()}")
        print(f"  activations_file     : {activations_file(cfg)}")
        print(f"  activations_metadata : {activations_metadata_file(cfg)}")
        print(f"  top_indices_file     : {top_indices_file(cfg)}")
        print(f"  top_values_file      : {top_values_file(cfg)}")
        print(f"  top_k_metadata_file  : {top_k_metadata_file(cfg)}")
        print(f"  xai_maps_file        : {xai_maps_file(cfg, ch)}")
        print(f"  crops_dir            : {crops_dir(cfg, ch)}")
        print(f"  collages_dir         : {collages_dir(cfg, ch)}")

    print("\n=== Dataset existence check ===")
    cfg = load_config(str(configs_dir / "rn152_ixg.yaml"))
    try:
        assert_dataset_exists(cfg)
        print(f"  ✓ Dataset found at: {dataset_val_dir(cfg).resolve()}")
    except FileNotFoundError as e:
        # Expected when running from a machine without the dataset present
        print(f"  (expected on dev machines without dataset):")
        for line in str(e).splitlines()[:3]:
            print(f"    {line}")
