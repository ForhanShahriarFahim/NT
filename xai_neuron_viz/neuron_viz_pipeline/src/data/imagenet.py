# =============================================================================
# FILE: neuron_viz_pipeline/src/data/imagenet.py
#
# Purpose:
#   Thin wrapper around CoE's MakeDataset (data/data_proces.py) that produces
#   two variants of the same dataset:
#
#     build_preprocessed_dataset(cfg)
#         Returns dataset with FULL transform:
#             Resize(256) → CenterCrop(224) → ToTensor() → Normalize(mean,std)
#         Used by: Stage 1 (extract) — model needs normalized input
#                  Stage 3 (xai_maps) — same, plus requires_grad on input
#
#     build_raw_dataset(cfg)
#         Returns dataset with NO normalize:
#             Resize(256) → CenterCrop(224) → ToTensor()
#         Used by: Stage 4 (crop) — we save these pixels as visible images,
#                                   so no ImageNet normalization statistics
#
# Why two variants and not one + un-normalize at save time:
#   - Un-normalization introduces tiny floating-point errors
#   - Simpler code path: Stage 4 reads raw tensors and casts to uint8 directly
#   - Matches the pattern in main.py NT LAB PATCH 7 where CoE already uses
#     two MakeDataset instances for exactly this reason
#
# Both variants use the SAME underlying ImageFolder so `dataset[i]` returns
# the same IMAGE in both — guaranteeing sample-index parity.
#
# Reference:
#   - data/data_proces.py (CoE, UNTOUCHED)   — provides MakeDataset class
#   - main.py NT LAB PATCH 7                  — pattern of two datasets
#   - pipeline_a/extract.py _make_args        — how we build the args namespace
# =============================================================================

import os
import sys
import types
from typing import Any, Dict

import torch.utils.data as data
import torchvision.transforms as T

# CoE's MakeDataset lives at repo_root/data/data_proces.py
# All scripts run from repo root (see base.yaml WORKING DIRECTORY ASSUMPTION),
# so we can import directly.
#
# The wrapper below also works when CoE's `data/` package is available —
# same as your existing pipeline_a and pipeline_b code.
from data.data_proces import MakeDataset


# ---------------------------------------------------------------------------
# Args-namespace builder (MakeDataset needs an `args` object with certain
# attributes; we build it from the config dict rather than requiring the
# caller to pass an argparse Namespace)
# ---------------------------------------------------------------------------

def _make_coe_args(cfg: Dict[str, Any]) -> types.SimpleNamespace:
    """
    Build a minimal SimpleNamespace with exactly the attributes MakeDataset
    reads from its `args` parameter. Matches the pattern used in
    pipeline_a/extract.py → _make_args().

    Attributes MakeDataset requires:
        args.data_dir        — {data.path}/{dataset_name}, e.g. "./dataset/imagenet"
        args.model_name      — used only inside MakeDataset to detect CLIP models;
                               we pass the config value so non-CLIP models work
        args.resume          — unused for our case (only CLIP uses it)
        args.dataset         — original spec, e.g. "imagenet-val"
        args.dataset_name    — "imagenet"
        args.dataset_split   — "val"
    """
    dataset_spec  = cfg["data"]["dataset"]              # e.g. "imagenet-val"
    dataset_name  = dataset_spec.split("-")[0]           # "imagenet"
    dataset_split = dataset_spec.split("-")[1]           # "val"

    a = types.SimpleNamespace()
    a.data_dir      = os.path.join(cfg["data"]["path"], dataset_name)
    a.model_name    = cfg["model"]["name"]
    a.resume        = None                                # not used for ImageFolder
    a.dataset       = dataset_spec
    a.dataset_name  = dataset_name
    a.dataset_split = dataset_split
    return a


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _preprocessed_transform(cfg: Dict[str, Any]) -> T.Compose:
    """
    Full transform: Resize → CenterCrop → ToTensor → Normalize.
    Matches the transform used everywhere CoE and pipeline_a run inference.
    """
    return T.Compose([
        T.Resize(cfg["data"]["resize"]),
        T.CenterCrop(cfg["data"]["image_size"]),
        T.ToTensor(),
        T.Normalize(
            mean=cfg["data"]["normalize_mean"],
            std=cfg["data"]["normalize_std"],
        ),
    ])


def _raw_transform(cfg: Dict[str, Any]) -> T.Compose:
    """
    No-normalize transform: Resize → CenterCrop → ToTensor.
    Produces tensors in [0, 1] ready for saving as uint8 images after
    ×255 rescale.
    """
    return T.Compose([
        T.Resize(cfg["data"]["resize"]),
        T.CenterCrop(cfg["data"]["image_size"]),
        T.ToTensor(),
    ])


def build_preprocessed_dataset(cfg: Dict[str, Any]) -> data.Dataset:
    """
    Build the ImageNet-val dataset with FULL preprocessing applied.

    Used as model input by Stage 1 (extract) and Stage 3 (xai_maps).
    """
    coe_args = _make_coe_args(cfg)
    transform = _preprocessed_transform(cfg)
    return MakeDataset(
        coe_args,
        transform=transform,
        dataset_split=coe_args.dataset_split,
    )


def build_raw_dataset(cfg: Dict[str, Any]) -> data.Dataset:
    """
    Build the ImageNet-val dataset WITHOUT normalization.

    Used by Stage 4 (crop) to get visible pixel values for saving as PNGs/JPGs.
    """
    coe_args = _make_coe_args(cfg)
    transform = _raw_transform(cfg)
    return MakeDataset(
        coe_args,
        transform=transform,
        dataset_split=coe_args.dataset_split,
    )


# ---------------------------------------------------------------------------
# Smoke test — run from repo root:
#   python -m neuron_viz_pipeline.src.data.imagenet
# Requires the dataset to actually exist; prints size and first-sample shape.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Make 'src.utils.config' importable when run from repo root
    here = Path(__file__).resolve()
    # here = .../neuron_viz_pipeline/src/data/imagenet.py
    # parents[0] = src/data/, parents[1] = src/, parents[2] = neuron_viz_pipeline/
    sys.path.insert(0, str(here.parents[2]))

    from src.utils.config import load_config
    from src.utils.paths import dataset_val_dir, assert_dataset_exists

    # Try loading rn152 config (paths are relative to cwd)
    config_candidates = [
        "neuron_viz_pipeline/configs/rn152_ixg.yaml",  # from repo root
        "configs/rn152_ixg.yaml",                   # from inside neuron_viz_pipeline/
    ]
    cfg = None
    for path in config_candidates:
        if Path(path).is_file():
            cfg = load_config(path)
            print(f"Loaded config: {path}")
            break
    if cfg is None:
        print("Could not find rn152_ixg.yaml — run from repo root or neuron_viz_pipeline/")
        sys.exit(1)

    print(f"Dataset will be loaded from: {dataset_val_dir(cfg).resolve()}")

    try:
        assert_dataset_exists(cfg)
    except FileNotFoundError as e:
        print("Dataset missing — smoke test skipped")
        print(str(e).splitlines()[0])
        sys.exit(0)

    # Build and inspect both variants
    print("\n=== Building preprocessed dataset ===")
    ds_pre = build_preprocessed_dataset(cfg)
    print(f"  length: {len(ds_pre)}")
    img_pre, label_pre = ds_pre[0]
    print(f"  sample 0: tensor {tuple(img_pre.shape)}, label {label_pre}, "
          f"range=[{img_pre.min():.3f}, {img_pre.max():.3f}]")

    print("\n=== Building raw dataset ===")
    ds_raw = build_raw_dataset(cfg)
    print(f"  length: {len(ds_raw)}")
    img_raw, label_raw = ds_raw[0]
    print(f"  sample 0: tensor {tuple(img_raw.shape)}, label {label_raw}, "
          f"range=[{img_raw.min():.3f}, {img_raw.max():.3f}]")

    print("\n=== Parity check — same label for same index ===")
    assert label_pre == label_raw, "Dataset variants disagree on labels!"
    print(f"  sample 0 label: {label_pre} (both variants agree)  ✓")
    print("\nSmoke test passed.")