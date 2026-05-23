# =============================================================================
# FILE: neuron_viz_pipeline/src/models/builder.py
#
# Purpose:
#   Thin wrapper around CoE's models/__init__.py build_models() function.
#   Takes a config dict and returns a ready-to-use model on the target device.
#
# Why a wrapper:
#   1. CoE's build_models() expects an argparse-style Namespace, not a dict.
#      This wrapper does the conversion so pipeline code only touches dicts.
#   2. CoE's build_models() returns a 5-tuple (model, criterion, preprocess,
#      tokenizer, msg). For our Phase 1 use case we only need `model` — the
#      wrapper extracts it.
#   3. Centralizes the eval() + .to(device) boilerplate.
#   4. Does ONE important thing: resolves "vit" → "vit_base_patch16_224"
#      before passing to CoE so CoE's internal dispatch picks the right path
#      (same as main.py __main__ does).
#
# Reference:
#   - models/__init__.py (CoE, UNTOUCHED) — the build_models() function
#   - main.py __main__   (CoE)            — the 'vit' → 'vit_base_patch16_224'
#                                           mapping we reproduce here
# =============================================================================

import sys
import types
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# CoE's build_models lives at repo_root/models/__init__.py
# Scripts are run from repo root (see base.yaml WORKING DIRECTORY ASSUMPTION).
from models import build_models as _coe_build_models


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Model name resolution table.
# Must match src/utils/paths.py MODEL_NAME_MAP and main.py __main__ mapping,
# otherwise Stage 1 activation files would be saved under a different model
# name than what build_models actually produces.
MODEL_NAME_MAP = {
    "vit": "vit_base_patch16_224",
}


def _resolve_model_name(name: str) -> str:
    """`vit` → `vit_base_patch16_224`; others unchanged."""
    return MODEL_NAME_MAP.get(name, name)


def _make_coe_args(cfg: Dict[str, Any]) -> types.SimpleNamespace:
    """
    Build a SimpleNamespace with the attributes CoE's build_models needs.
    Matches the args-namespace pattern used throughout CoE.

    Attributes build_models reads:
        args.model_name  — e.g. "rn152", "vit_base_patch16_224"
        args.resume      — path to a checkpoint, or None for pretrained
        args.dataset     — e.g. "imagenet-val"; used only in 'rn152 +
                           imagenet' branch to guard the rn152 loader
    """
    a = types.SimpleNamespace()
    a.model_name = _resolve_model_name(cfg["model"]["name"])
    a.resume     = cfg.get("model", {}).get("resume", None)
    a.dataset    = cfg["data"]["dataset"]
    return a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(
    cfg: Dict[str, Any],
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> nn.Module:
    """
    Build the model specified in `cfg` and move it to `device`.

    Args:
        cfg        : resolved config dict (from src.utils.config.load_config)
        device     : "cuda" | "cpu" | None. If None, auto-detect.
        eval_mode  : if True, sets model.eval() before returning. Default True
                     because every pipeline stage uses inference mode.

    Returns:
        torch.nn.Module ready for use (on device, in eval mode)

    Raises:
        ValueError if build_models returns None for the requested name
                   (CoE's gatekeeper path — means the name isn't recognized)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    coe_args = _make_coe_args(cfg)
    print(
        f"[build_model] model_name='{coe_args.model_name}' "
        f"(resolved from cfg.model.name='{cfg['model']['name']}'), "
        f"device='{device}'"
    )

    model, _criterion, _preprocess, _tokenizer, _msg = _coe_build_models(coe_args)

    if model is None:
        raise ValueError(
            f"CoE build_models returned None for model_name="
            f"'{coe_args.model_name}'.\n"
            f"This means CoE's gatekeeper does not recognize the name.\n"
            f"Supported names: rn50, rn152, vgg16, vit_base_patch16_224 (alias: vit)"
        )

    model = model.to(device)
    if eval_mode:
        model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build_model] loaded {coe_args.model_name}: "
          f"{n_params / 1e6:.1f}M parameters, on {device}")

    return model


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Retrieve a submodule by its dotted path name.
    Mirrors the logic in ActivationExtractor._get_layer_by_name() from
    pipeline_a/extract.py — handles mixed attribute-access and numeric indices.

    Examples:
        get_layer(resnet152, "layer4.2.conv3")
        get_layer(vit,       "blocks.11.mlp.fc2")

    Raises:
        AttributeError if the path doesn't resolve to a leaf module.
    """
    parts = layer_name.split(".")
    current: Any = model
    for part in parts:
        if part.isdigit():
            # Sequential / ModuleList indexing
            idx = int(part)
            if hasattr(current, "__getitem__"):
                current = current[idx]
            else:
                current = getattr(current, part)
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise AttributeError(
            f"Layer path '{layer_name}' resolved to {type(current).__name__}, "
            f"not an nn.Module. Check your config's model.layer value."
        )
    return current


# ---------------------------------------------------------------------------
# Smoke test — run from repo root:
#   python -m neuron_viz_pipeline.src.models.builder
# Downloads weights on first run (pretrained=True inside CoE).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    # Make 'src.utils.config' importable when run directly
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[2]))

    from src.utils.config import load_config

    config_candidates = [
        "neuron_viz_pipeline/configs/rn152_ixg.yaml",
        "configs/rn152_ixg.yaml",
    ]
    cfg = None
    for path in config_candidates:
        if Path(path).is_file():
            cfg = load_config(path)
            print(f"Loaded config: {path}\n")
            break
    if cfg is None:
        print("Could not find rn152_ixg.yaml — run from repo root")
        sys.exit(1)

    print("=== Building model (this may download weights on first run) ===")
    model = build_model(cfg, device="cpu")    # use CPU for the smoke test
    print(f"Model type: {type(model).__name__}")

    print("\n=== Getting target layer ===")
    layer_name = cfg["model"]["layer"]
    layer = get_layer(model, layer_name)
    print(f"  layer '{layer_name}' → {type(layer).__name__}")

    # Quick forward pass with a dummy tensor
    print("\n=== Dummy forward pass ===")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"  input: {tuple(x.shape)} → output: {tuple(y.shape)}")
    print("\nSmoke test passed.")