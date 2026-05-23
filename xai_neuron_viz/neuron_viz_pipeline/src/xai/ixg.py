# =============================================================================
# FILE: neuron_viz_pipeline/src/xai/ixg.py
#
# Purpose:
#   Input × Gradient (IxG) attribution method.
#   Computes a 2D spatial saliency map showing which input pixels contribute
#   most to a specific neuron's activation on a specific image.
#
# Backend: pure torch autograd (no Captum dependency).
#   Reasons for choosing pure autograd over Captum:
#     1. The algorithm is ~15 lines — a Captum wrapper adds abstraction without
#        saving code
#     2. Sidesteps Captum/ViT edge cases (the user previously hit GBP/ViT issues
#        because GBP hooks nn.ReLU, which ViT doesn't have)
#     3. Full transparency — every step is visible for debugging
#   A Captum-backed IxG may be added as a cross-check in a future step.
#
# The Math:
#   For input x ∈ R^(3×H×W) and target neuron activation a(x):
#     gradient = ∂a(x) / ∂x          shape [3, H, W]
#     ixg      = x * gradient         shape [3, H, W]   (element-wise product)
#     ixg      = ixg.sum(dim=0)       shape [H, W]      (if sum_channels=True)
#     ixg      = ixg.abs()            shape [H, W]      (if abs_output=True)
#
# Target-scalar reduction (the part that differs between CNN and ViT):
#
#   layer_type = "conv"    — target layer output shape [1, C, H', W']
#     scalar = activations[0, channel_id, :, :].sum()
#     Sums over ALL spatial positions of the target channel.
#     Matches the paper's formula (Eq. 1 context, p.3):
#       "aggregate these values into a single scalar representing the total
#        contribution by summing over all spatial locations"
#
#   layer_type = "linear"  — target layer output shape [1, seq_len, hidden_dim]
#     ── vit_include_cls = False (default, recommended):
#         scalar = activations[0, 1:, channel_id].sum()
#         Sums over the 196 PATCH tokens only (skips the CLS token at index 0).
#         Cleaner spatial attribution — CLS has no spatial location, so
#         including it blurs the IxG map across the entire image.
#     ── vit_include_cls = True:
#         scalar = activations[0, :, channel_id].sum()
#         Sums over ALL 197 tokens (CLS + 196 patches).
#         Matches CoE's max_target="sum" behavior; may produce blurrier crops
#         because CLS attention has been mixed globally across the image by
#         block 11.
#
# Why the ViT default is False:
#   The paper's formula says "spatial locations"; the CLS token is not spatial.
#   With vit_include_cls=False, ViT attribution is conceptually equivalent to
#   CNN attribution (both sum only over positions that correspond to image
#   regions), so the final crops are directly comparable across architectures.
#
# GPU memory budget per image:
#   ~1.5-2.5 GB peak during backward (graph retention for both CNN/ViT at
#   batch_size=1). Well within the user's 15 GB VRAM budget.
# =============================================================================

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .base import GradientBasedMethod


class IxG(GradientBasedMethod):
    """
    Input × Gradient attribution for a specific neuron at a specific layer.

    Config schema read:
        cfg.xai.ixg.abs_output       (bool, default True)
        cfg.xai.ixg.sum_channels     (bool, default True)
        cfg.xai.ixg.vit_include_cls  (bool, default False) — ONLY used when
                                     layer_type="linear" (ViT). Ignored for
                                     "conv" layers.

    One instance of this class is reused across all top-k images of a neuron.
    """

    name = "ixg"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        # Unpack options from cfg.xai.ixg with safe defaults
        self.abs_output      = self.method_cfg.get("abs_output",      True)
        self.sum_channels    = self.method_cfg.get("sum_channels",    True)
        # ViT-specific option; conv path ignores it at computation time
        self.vit_include_cls = self.method_cfg.get("vit_include_cls", False)

        print(
            f"[IxG] initialized: abs_output={self.abs_output}, "
            f"sum_channels={self.sum_channels}, "
            f"vit_include_cls={self.vit_include_cls}"
        )

    # ------------------------------------------------------------------
    # Method-specific metadata for Stage 3 output JSON
    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        """
        Return the IxG-specific parameters so Stage 3 can embed them
        in the xai_maps_metadata.json without having to know what
        keys each method cares about.

        vit_include_cls is only semantically meaningful for layer_type
        == "linear" (ViT). It's still recorded for conv models but
        the value has no effect on output.
        """
        return {
            "abs_output":      self.abs_output,
            "sum_channels":    self.sum_channels,
            "vit_include_cls": self.vit_include_cls,
        }

    # ------------------------------------------------------------------
    # Scalar reduction — the one place ViT vs CNN behavior differs
    # ------------------------------------------------------------------

    def _reduce_to_scalar(
        self,
        activations: torch.Tensor,
        target_channel: int,
        layer_type: str,
    ) -> torch.Tensor:
        """
        Reduce target-layer activations to a single scalar for backward pass.

        Args:
            activations    : tensor captured by the forward hook on the target
                              layer. Shape depends on layer_type:
                                conv   → [1, C, H', W']
                                linear → [1, seq_len, hidden_dim]
            target_channel : neuron/channel index within that layer
            layer_type     : "conv" | "linear"

        Returns:
            0-dim tensor (scalar) suitable for .backward()

        Raises:
            ValueError if layer_type is not "conv" or "linear"
            IndexError if target_channel is out of range (clearer error than
                       the cryptic one torch would give)
        """
        if layer_type == "conv":
            # Expected shape: [1, C, H', W']
            if activations.dim() != 4:
                raise ValueError(
                    f"conv layer_type expects 4D activation [1, C, H', W'], "
                    f"got {activations.dim()}D shape {tuple(activations.shape)}"
                )
            C = activations.shape[1]
            if not (0 <= target_channel < C):
                raise IndexError(
                    f"target_channel={target_channel} out of range for conv "
                    f"layer with {C} channels"
                )
            # Sum over ALL spatial positions of the target channel
            # Shape: activations[0, target_channel, :, :] is [H', W']
            scalar = activations[0, target_channel, :, :].sum()

        elif layer_type == "linear":
            # Expected shape: [1, seq_len, hidden_dim] (ViT) or
            # [1, hidden_dim] for a post-CLS-only model (rare)
            if activations.dim() == 3:
                seq_len  = activations.shape[1]
                hidden_d = activations.shape[2]
                if not (0 <= target_channel < hidden_d):
                    raise IndexError(
                        f"target_channel={target_channel} out of range for "
                        f"linear layer with hidden_dim={hidden_d}"
                    )

                if self.vit_include_cls:
                    # Sum over ALL tokens (CLS + patches) = all seq_len positions
                    # activations[0, :, target_channel] has shape [seq_len]
                    scalar = activations[0, :, target_channel].sum()
                else:
                    # Skip CLS (index 0), sum over patch tokens only
                    # activations[0, 1:, target_channel] has shape [seq_len-1]
                    # For ViT-B/16: 197 - 1 = 196 patches
                    if seq_len < 2:
                        raise ValueError(
                            f"vit_include_cls=False but seq_len={seq_len} < 2 "
                            f"— no patch tokens to sum over. Check your layer "
                            f"choice or set xai.ixg.vit_include_cls=true."
                        )
                    scalar = activations[0, 1:, target_channel].sum()

            elif activations.dim() == 2:
                # Shape [1, hidden_dim] — already reduced across sequence.
                # No CLS choice to make here.
                hidden_d = activations.shape[1]
                if not (0 <= target_channel < hidden_d):
                    raise IndexError(
                        f"target_channel={target_channel} out of range for "
                        f"linear layer with hidden_dim={hidden_d}"
                    )
                scalar = activations[0, target_channel]
            else:
                raise ValueError(
                    f"linear layer_type expects 2D or 3D activation, "
                    f"got {activations.dim()}D shape {tuple(activations.shape)}"
                )

        else:
            raise ValueError(
                f"Unknown layer_type: {layer_type!r}. "
                f"Expected 'conv' or 'linear'."
            )

        return scalar

    # ------------------------------------------------------------------
    # Main saliency computation
    # ------------------------------------------------------------------

    def compute_saliency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_layer: nn.Module,
        target_channel: int,
        layer_type: str,
    ) -> np.ndarray:
        """
        Compute IxG spatial saliency map for a single preprocessed image.

        Args:
            model          : nn.Module in eval mode on correct device
            input_tensor   : [1, 3, H, W] preprocessed, on same device as model
                             Caller MUST NOT set requires_grad — we handle it.
            target_layer   : nn.Module whose activations we attribute from
            target_channel : int, channel id within target_layer's output
            layer_type     : "conv" | "linear"

        Returns:
            [H, W] float32 numpy array. Higher = more attribution.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be [1, 3, H, W], got {tuple(input_tensor.shape)}"
            )

        # ── Set up autograd ─────────────────────────────────────────
        # Clone so the caller's tensor is not mutated; detach first to strip
        # any graph the caller may have built, then set requires_grad=True.
        x = input_tensor.detach().clone().requires_grad_(True)

        # Make sure no stale gradients linger from a previous call
        model.zero_grad(set_to_none=True)

        # ── Register forward hook to capture target layer output ─────
        # We capture the tensor WITH the graph (no .detach() here — we need
        # the graph for backward). This is different from ActivationExtractor
        # which detaches because it only needs the numeric values.
        captured: Dict[str, torch.Tensor] = {}

        def _hook(module, inputs, output):
            # If output is a tuple (some modules return that), take first element.
            if isinstance(output, (tuple, list)):
                captured["act"] = output[0]
            else:
                captured["act"] = output

        handle = target_layer.register_forward_hook(_hook)

        try:
            # ── Forward pass ────────────────────────────────────────
            # We use torch.enable_grad() explicitly in case caller is inside
            # a no_grad context. Model is in eval mode — that only affects
            # BatchNorm/Dropout behavior, NOT autograd.
            with torch.enable_grad():
                _ = model(x)

            if "act" not in captured:
                raise RuntimeError(
                    f"Forward hook on target layer did not fire. "
                    f"This usually means the layer is not a leaf module or "
                    f"the forward pass took a branch that skipped it."
                )

            # ── Reduce to scalar ─────────────────────────────────────
            scalar = self._reduce_to_scalar(
                captured["act"], target_channel, layer_type,
            )

            # ── Backward pass ────────────────────────────────────────
            scalar.backward()

            if x.grad is None:
                raise RuntimeError(
                    "Backward pass produced no gradient on input. "
                    "This should not happen if requires_grad=True was set."
                )

            # ── Compute IxG ──────────────────────────────────────────
            # Detach everything from the graph before further numpy conversion
            ixg = (x * x.grad).detach()  # shape [1, 3, H, W]

            if self.sum_channels:
                ixg = ixg.sum(dim=1)     # shape [1, H, W]
            # else: leave as [1, 3, H, W] — but then [H, W] return contract breaks.
            # So we enforce sum_channels=True in practice for this pipeline:
            if ixg.dim() != 3 or ixg.shape[0] != 1:
                raise RuntimeError(
                    f"IxG tensor has unexpected shape {tuple(ixg.shape)} after "
                    f"channel reduction. For this pipeline, xai.ixg.sum_channels "
                    f"must be true (the [H, W] return contract requires it)."
                )

            if self.abs_output:
                ixg = ixg.abs()

            # [1, H, W] → [H, W] numpy
            saliency = ixg[0].cpu().numpy().astype(np.float32)

        finally:
            handle.remove()
            # Clean up input grad so it doesn't linger on the caller's tensor
            # (we cloned it, so actually this is clean already — but be explicit)
            if x.grad is not None:
                x.grad = None

        return saliency