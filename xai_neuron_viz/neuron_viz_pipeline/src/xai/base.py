# =============================================================================
# FILE: neuron_viz_pipeline/src/xai/base.py
#
# Purpose:
#   Abstract base class every XAI method in this project implements.
#
#   Having a common interface means:
#     - Stage 3 runner (scripts/stage3_xai_maps.py) is method-agnostic
#     - Adding IntegratedGradients/SmoothGrad/CRP in the future = one new file
#       subclassing XAIMethod, no changes anywhere else
#
# Design notes:
#   - compute_saliency takes a SINGLE preprocessed image (batch_size=1 for IxG
#     because backward pass retains the computation graph → memory-heavy).
#     We keep the interface simple and let Stage 3 iterate over top-k images.
#   - Returns np.ndarray shaped [H, W] to match what Stage 4's cropping
#     code expects (get_crop_bbox expects 2D saliency).
#   - Higher value = more attribution (positive is "this pixel contributes to
#     the neuron firing"). For signed methods, this means abs() was applied.
#
# What each method must implement:
#   name       → short string, e.g. "ixg" — matches the cfg.xai.method key
#   compute_saliency(...) → the actual attribution math
# =============================================================================

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


class XAIMethod(ABC):
    """
    Abstract base for all XAI saliency methods in neuron_viz_pipeline.

    Contract:
        Given a model, a preprocessed input image, a target layer name, and
        a target channel/neuron id, produce a 2D spatial saliency map [H, W]
        where higher values indicate pixels contributing more strongly to the
        neuron's activation on this specific image.
    """

    #: Short identifier — must match cfg.xai.method for registry lookup.
    #: Subclasses override this.
    name: str = "base"

    def __init__(self, cfg: Dict[str, Any]):
        """
        Args:
            cfg: full resolved config dict. Method-specific options live under
                 cfg["xai"][self.name] (e.g. cfg["xai"]["ixg"]).
        """
        self.cfg = cfg
        # Pull method-specific options. Absence is acceptable — methods can
        # have sensible defaults.
        self.method_cfg = cfg.get("xai", {}).get(self.name, {}) or {}

    # ------------------------------------------------------------------
    # Optional hook 1: config validation
    # ------------------------------------------------------------------
    def validate_config(self, layer_type: str) -> None:
        """
        Called by Stage 3 at startup, before any compute_saliency() call.
        Default implementation does nothing. Override to fail fast on
        incompatible model/method combinations.

        Example use:
            Attention Rollout is ViT-only, so its override raises
            ValueError when layer_type != "linear".

        Args:
            layer_type : "conv" | "linear" — resolved from cfg.model.layer_type
        """
        return None

    # ------------------------------------------------------------------
    # Optional hook 2: per-method metadata
    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        """
        Return a dict of method-specific parameters to record in the
        Stage 3 output metadata JSON. Keeps Stage 3 method-agnostic:
        IxG returns {abs_output, sum_channels, vit_include_cls}, IG
        returns {n_steps, baseline}, Attention Rollout returns
        {include_identity, head_agg, weight_by_neuron, ...}.

        Default returns an empty dict. Override to populate.
        """
        return {}

    # ------------------------------------------------------------------
    # Core: the actual attribution math — REQUIRED
    # ------------------------------------------------------------------
    @abstractmethod
    def compute_saliency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        target_channel: int,
        layer_type: str,
    ) -> np.ndarray:
        """
        Compute the saliency map for a single image.

        Args:
            model          : nn.Module, in eval mode, on the correct device.
                             NOT reset — caller is responsible for model.zero_grad()
                             etc. between calls if needed.
            input_tensor   : [1, 3, H, W] preprocessed image on the SAME device
                             as the model. Caller should set requires_grad=True
                             before passing in if the method needs gradients.
            target_layer   : the nn.Module whose activations we want to attribute
                             from (resolved from cfg.model.layer by the runner).
            target_channel : int, the neuron/channel id within `target_layer`'s
                             output (cfg.neuron.channel_id).
            layer_type     : "conv" | "linear" — tells the method how to reduce
                             the target layer's output to a scalar.

        Returns:
            [H, W] np.float32 spatial saliency map. Same H, W as input_tensor's
            last two dims. Higher value = higher attribution.
        """
        raise NotImplementedError


# Convenience: methods that need gradient computation can inherit from this
# instead of XAIMethod to signal their characteristic.
class GradientBasedMethod(XAIMethod):
    """
    Marker subclass for gradient-based methods (IxG, IntegratedGradients,
    SmoothGrad, ...). Doesn't add behavior — just makes the registry /
    debugging output clearer about what family a method belongs to.
    """
    pass