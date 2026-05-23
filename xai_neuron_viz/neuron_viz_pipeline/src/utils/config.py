# =============================================================================
# FILE: neuron_viz_pipeline/src/utils/config.py
#
# Purpose:
#   Load a YAML config file and resolve its `inherits:` chain into a single
#   dict. Deep-merge: child keys override parent keys recursively.
#
# Usage:
#   from src.utils.config import load_config
#   cfg = load_config("neuron_viz_pipeline/configs/rn152_ixg.yaml")
#   print(cfg["model"]["name"])         # "rn152"
#   print(cfg["extract"]["batch_size"]) # 32  (from base.yaml)
#
# Why `inherits:` instead of OmegaConf/Hydra:
#   - Zero extra dependencies beyond PyYAML
#   - One-level inheritance is all we need (base → model-specific)
#   - Transparent — you can read the resolved config by printing the dict
# =============================================================================

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `override` into `base`. Override keys win.
    Nested dicts are merged; scalars and lists are replaced wholesale.

    Example:
      base     = {"extract": {"batch_size": 32, "pool_type": "raw"}}
      override = {"extract": {"batch_size": 16}}
      result   = {"extract": {"batch_size": 16, "pool_type": "raw"}}
    """
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config and resolve its `inherits:` chain.

    Args:
        config_path: path to a YAML config file, e.g.
                     "neuron_viz_pipeline/configs/rn152_ixg.yaml"

    Returns:
        Fully resolved config dict with all inherited defaults applied.
        The `inherits:` key is removed from the returned dict.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    # Resolve inheritance (one level, loaded recursively).
    # `inherits:` is always resolved relative to the child config's directory.
    if "inherits" in cfg:
        parent_name = cfg.pop("inherits")
        parent_path = config_path.parent / parent_name
        parent_cfg = load_config(str(parent_path))
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg


def print_config(cfg: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty-print a resolved config. Useful for sanity-checking `inherits:`.
    """
    pad = "  " * indent
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{pad}{key}: {value}")


if __name__ == "__main__":
    # Smoke test — run from repo root:
    #   python -m neuron_viz_pipeline.src.utils.config neuron_viz_pipeline/configs/rn152_ixg.yaml
    # or from inside neuron_viz_pipeline/:
    #   python -m src.utils.config configs/rn152_ixg.yaml
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/rn152_ixg.yaml"
    resolved = load_config(path)
    print(f"Resolved config: {path}\n")
    print_config(resolved)