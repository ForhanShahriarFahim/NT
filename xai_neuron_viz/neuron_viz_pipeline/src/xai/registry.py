# =============================================================================
# FILE: neuron_viz_pipeline/src/xai/registry.py
#
# Purpose:
#   Method lookup — given a config dict, return an instantiated XAIMethod.
#
#   The Stage 3 runner calls get_xai_method(cfg) which:
#     1. Reads cfg.xai.method (e.g. "ixg")
#     2. Looks up the matching class in METHODS
#     3. Instantiates it with the full config
#     4. Returns the ready-to-use method object
#
#   Adding a new method (e.g. IntegratedGradients) = 2 lines here (register
#   the class in METHODS) plus the new method file — nothing else changes.
# =============================================================================

from typing import Any, Dict, Type

from .base import XAIMethod
from .ixg import IxG
from .ig  import IntegratedGradients
from .attention_rollout import AttentionRollout


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# Keys MUST match the method's `name` attribute AND cfg.xai.method values
# in the YAML configs. Keeping all three in sync is a validation invariant.

METHODS: Dict[str, Type[XAIMethod]] = {
    IxG.name:                 IxG,                    # "ixg"
    IntegratedGradients.name: IntegratedGradients,    # "ig"
    AttentionRollout.name:    AttentionRollout,       # "attention_rollout" (ViT-only)
    # Future:
    # "smoothgrad": SmoothGrad,
    # "crp"       : CRP,
}


def get_xai_method(cfg: Dict[str, Any]) -> XAIMethod:
    """
    Instantiate the XAI method named in cfg.xai.method.

    Args:
        cfg: resolved config dict

    Returns:
        XAIMethod subclass instance, ready for compute_saliency() calls

    Raises:
        KeyError         if cfg.xai.method is missing from the config
        NotImplementedError if the method name is not in METHODS
    """
    if "xai" not in cfg or "method" not in cfg["xai"]:
        raise KeyError(
            "cfg.xai.method is required. Add `xai: {method: 'ixg'}` to "
            "your config, or check that base.yaml was inherited correctly."
        )

    method_name = cfg["xai"]["method"]

    if method_name not in METHODS:
        available = ", ".join(sorted(METHODS.keys()))
        raise NotImplementedError(
            f"XAI method '{method_name}' is not implemented.\n"
            f"Available methods: {available}\n"
            f"To add a new method: create src/xai/<method>.py with a subclass "
            f"of XAIMethod, then register it in src/xai/registry.py."
        )

    method_cls = METHODS[method_name]
    return method_cls(cfg)


def available_methods() -> list:
    """Return a sorted list of registered method names (useful for --help)."""
    return sorted(METHODS.keys())