# =============================================================================
# FILE: neuron_viz_pipeline/src/xai/ig.py
#
# Purpose:
#   Integrated Gradients (IG) attribution method — Sundararajan, Taly, Yan
#   "Axiomatic Attribution for Deep Networks" (ICML 2017).
#
#   Produces a 2D spatial saliency map showing which input pixels contribute
#   most to a specific neuron's activation, with stronger theoretical
#   guarantees than IxG alone (completeness + sensitivity axioms).
#
# Relationship to IxG:
#   IG = IxG averaged along a straight-line path from a baseline (zero image)
#   to the real input, then multiplied by (input - baseline).
#
#   Because the scalar reduction (conv vs linear, CLS handling, etc.) is
#   identical to IxG, this class INHERITS FROM IxG and reuses:
#     - __init__ config unpacking (plus IG-specific keys)
#     - _reduce_to_scalar (unchanged — same math)
#     - the forward-hook pattern
#
#   What IG adds:
#     - a baseline tensor x'
#     - a path-integration loop over n_steps interpolated inputs
#     - the final multiplication by (input - baseline)
#
# The Math (straight-line path integration):
#
#     IG_i(x) = (x_i - x'_i) × ∫[α=0 to 1] ∂f(x' + α(x-x')) / ∂x_i  dα
#
#   Riemann approximation (what we compute):
#
#     IG_i(x) ≈ (x_i - x'_i) × (1/n) × Σ[k=1 to n] ∂f(x_interp_k) / ∂x_i
#
#   where x_interp_k = x' + (k/n)·(x - x').
#
#   Note: some implementations use midpoint (α=(k-0.5)/n) instead of right-
#   endpoint (α=k/n). We use right-endpoint to match Sundararajan et al.'s
#   pseudocode. With n=50 steps the difference is visually imperceptible.
#
# Config schema (cfg.xai.ig.*):
#   n_steps         : int, default 50 — number of Riemann steps.
#                      Paper recommends 20-300; 50 is a good balance.
#   baseline        : str, default "zero" — "zero" | "uniform_noise"
#                      "zero" means all-zero image in pixel space.
#                      For normalized inputs this means a grey-ish blur
#                      (because subtracting mean/std makes zero → grey).
#   abs_output      : inherited from IxG, default True
#   sum_channels    : inherited from IxG, default True
#   vit_include_cls : inherited from IxG, default False
#
# GPU memory budget per image:
#   IG is ~n_steps × IxG cost. With n_steps=50, that's 50× the forward +
#   backward work per image. For ViT (~2s/image IxG) this is ~100s/image
#   for IG. A 50-image smoke test on ViT ≈ 80 minutes. For the 150-image
#   full run it's ~4 hours — use a smaller n_steps on ViT if time matters.
#
# Why not use Captum's IntegratedGradients?
#   Same reason as IxG: pure torch keeps every line visible. Captum wraps
#   this exact formula in 3 layers of indirection. If Captum cross-check
#   is wanted later, it's easy to add as a separate class.
# =============================================================================

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .ixg import IxG


class IntegratedGradients(IxG):
    """
    Integrated Gradients attribution for a specific neuron at a specific layer.

    Inherits everything from IxG and overrides `compute_saliency` to
    integrate along a straight-line path from baseline to input.

    One instance is reused across all top-k images of a neuron.
    """

    name = "ig"

    def __init__(self, cfg: Dict[str, Any]):
        # Parent IxG.__init__ reads cfg["xai"]["ixg"] — we want it to read
        # cfg["xai"]["ig"] instead. We handle this explicitly:
        #   - Call the grandparent XAIMethod.__init__ manually so self.method_cfg
        #     points at cfg["xai"]["ig"], not cfg["xai"]["ixg"].
        #   - Then duplicate IxG's __init__ body.
        #
        # This is intentional: IG's config block may diverge from IxG's as
        # the project grows (e.g., IG adds n_steps/baseline that IxG doesn't
        # have), so we don't want IG to silently inherit IxG's config.
        from .base import XAIMethod
        XAIMethod.__init__(self, cfg)

        # Shared options (same defaults as IxG)
        self.abs_output      = self.method_cfg.get("abs_output",      True)
        self.sum_channels    = self.method_cfg.get("sum_channels",    True)
        self.vit_include_cls = self.method_cfg.get("vit_include_cls", False)

        # IG-specific options
        self.n_steps         = int(self.method_cfg.get("n_steps",  50))
        self.baseline_type   = str(self.method_cfg.get("baseline", "zero"))

        # Validate IG options
        if self.n_steps < 2:
            raise ValueError(
                f"xai.ig.n_steps must be >= 2 (got {self.n_steps}). "
                f"Smaller values make the Riemann sum meaningless."
            )
        if self.baseline_type not in ("zero", "uniform_noise"):
            raise ValueError(
                f"xai.ig.baseline must be 'zero' or 'uniform_noise' "
                f"(got {self.baseline_type!r})."
            )

        print(
            f"[IG]  initialized: n_steps={self.n_steps}, "
            f"baseline={self.baseline_type!r}, "
            f"abs_output={self.abs_output}, "
            f"sum_channels={self.sum_channels}, "
            f"vit_include_cls={self.vit_include_cls}"
        )

    # ------------------------------------------------------------------
    # Method-specific metadata
    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        """IG config fields for the Stage 3 metadata JSON."""
        return {
            "n_steps":         self.n_steps,
            "baseline":        self.baseline_type,
            "abs_output":      self.abs_output,
            "sum_channels":    self.sum_channels,
            "vit_include_cls": self.vit_include_cls,
        }

    # ------------------------------------------------------------------
    # Baseline construction
    # ------------------------------------------------------------------
    def _make_baseline(self, reference: torch.Tensor) -> torch.Tensor:
        """
        Build a baseline tensor x' with the same shape, dtype, and device
        as `reference`. No grad — this is the start of the integration path,
        not a variable.

        "zero": all zeros in the preprocessed-space tensor (black-ish /
                normalized-grey, depending on mean/std normalization).
                This is the canonical choice from Sundararajan et al.
        "uniform_noise": uniform random in [0, 1] range. Useful as a
                         sanity-check alternative baseline; results should
                         be qualitatively similar to "zero".
        """
        if self.baseline_type == "zero":
            return torch.zeros_like(reference)
        # uniform_noise
        return torch.rand_like(reference)

    # ------------------------------------------------------------------
    # Main saliency computation — overrides IxG.compute_saliency
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
        Compute Integrated Gradients saliency for a single preprocessed image.

        Args:
            model          : nn.Module in eval mode on correct device
            input_tensor   : [1, 3, H, W] preprocessed, on same device as model
                             Caller should NOT set requires_grad — we handle it.
            target_layer   : nn.Module whose activations we attribute from
            target_channel : channel id within target_layer's output
            layer_type     : "conv" | "linear"

        Returns:
            [H, W] float32 numpy array. Higher = more attribution.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be [1, 3, H, W], got {tuple(input_tensor.shape)}"
            )

        # ── Prepare integration path ────────────────────────────────
        # Detach the input so we can freely build new tensors from it.
        x_real     = input_tensor.detach().clone()
        x_baseline = self._make_baseline(x_real)
        # delta = x - x' — used in two places: interpolation AND final multiply
        delta      = x_real - x_baseline

        # ── Accumulator for averaged gradients ──────────────────────
        # IG averages ∂f/∂x over n_steps interpolated inputs.
        # We accumulate on the same device as x_real.
        grad_sum = torch.zeros_like(x_real)

        # ── Path integration loop ───────────────────────────────────
        # Use right-endpoint Riemann: α_k = k/n  for k=1..n
        # (matches Sundararajan et al. 2017 pseudocode)
        for k in range(1, self.n_steps + 1):
            alpha = k / self.n_steps

            # Interpolated input x_α = x' + α·(x - x')
            # We clone and set requires_grad=True so autograd can differentiate
            # w.r.t. this specific step's input.
            x_alpha = (x_baseline + alpha * delta).detach().clone().requires_grad_(True)

            # Make sure no stale gradients linger from a previous step
            model.zero_grad(set_to_none=True)

            # Register forward hook ONCE per step — hook captures target
            # layer's output WITH the graph so backward can flow through it.
            captured: Dict[str, torch.Tensor] = {}

            def _hook(module, inputs, output):
                if isinstance(output, (tuple, list)):
                    captured["act"] = output[0]
                else:
                    captured["act"] = output

            handle = target_layer.register_forward_hook(_hook)
            try:
                # ── Forward pass ───────────────────────────────────
                with torch.enable_grad():
                    _ = model(x_alpha)

                if "act" not in captured:
                    raise RuntimeError(
                        "Forward hook on target layer did not fire inside IG. "
                        "Same root cause as in IxG — check target_layer is a leaf."
                    )

                # ── Reduce to scalar (inherited from IxG) ──────────
                scalar = self._reduce_to_scalar(
                    captured["act"], target_channel, layer_type,
                )

                # ── Backward pass ──────────────────────────────────
                scalar.backward()

                if x_alpha.grad is None:
                    raise RuntimeError(
                        "Backward pass produced no gradient on x_alpha inside IG."
                    )

                # Accumulate this step's gradient. Detach — we only need
                # numeric values from here on; grad_sum doesn't need a graph.
                grad_sum += x_alpha.grad.detach()

            finally:
                handle.remove()
                # Clear graph/grad from this step before next iter
                if x_alpha.grad is not None:
                    x_alpha.grad = None
                del x_alpha, scalar

        # ── Riemann average ─────────────────────────────────────────
        # The 1/n scaling turns the sum into the average over α ∈ (0, 1].
        grad_avg = grad_sum / float(self.n_steps)

        # ── Multiply by (input - baseline) — the "integrated" factor ─
        # This is what makes IG "(x-x') × ∫ gradient" rather than just
        # the average gradient. It gives IG its axiomatic properties.
        ig = (delta * grad_avg)  # shape [1, 3, H, W]

        # ── Reduce to [H, W] — same as IxG: sum channels, optional abs ──
        if self.sum_channels:
            ig = ig.sum(dim=1)   # [1, H, W]
        if ig.dim() != 3 or ig.shape[0] != 1:
            raise RuntimeError(
                f"IG tensor has unexpected shape {tuple(ig.shape)} after "
                f"channel reduction. xai.ig.sum_channels must be true."
            )

        if self.abs_output:
            ig = ig.abs()

        saliency = ig[0].detach().cpu().numpy().astype(np.float32)
        return saliency