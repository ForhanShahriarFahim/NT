# =============================================================================
# FILE: neuron_viz_pipeline/src/xai/attention_rollout.py
#
# Purpose:
#   Attention Rollout — Abnar & Zuidema, "Quantifying Attention Flow in
#   Transformers" (ACL 2020), as applied to ViT by Dosovitskiy et al.
#
#   Uses ViT's own attention weights (not gradients) to trace which input
#   patches contribute information to a target token's representation after
#   passing through all transformer blocks. Completely different family from
#   IxG/IG — attention-based rather than gradient-based.
#
#   THIS METHOD IS VIT-ONLY. It raises ValueError on conv models in its
#   validate_config override.
#
# The Math (from Abnar & Zuidema 2020, Section 3):
#
#   For each transformer block k ∈ {0, ..., L-1}:
#     1. W_k = softmax(Q_k K_k^T / sqrt(d_head))     per-head attention
#                shape [num_heads, N, N]  where N = seq_len = 197 for ViT-B/16
#     2. A_k = mean over heads of W_k                shape [N, N]
#     3. Account for residual connections: since x_{k+1} = x_k + Attn(x_k),
#        the effective mixing is (A_k + I), renormalized so rows sum to 1:
#                Ā_k = (A_k + I) / 2   (equivalently: row-normalize(A_k + I))
#
#   Rollout up through layer L-1:
#     R = Ā_{L-1} @ Ā_{L-2} @ ... @ Ā_0              shape [N, N]
#
#   Row i of R tells you "after all L layers, how much does token i's
#   representation depend on token j's input embedding" for each j.
#
# Two aggregation modes for our Phase-1 research use case:
#
#   weight_by_neuron=False  (vanilla rollout — Abnar & Zuidema default):
#     Take row 0 (the CLS row) of R → [N] vector showing which patches
#     flow information into CLS. Same output regardless of target neuron.
#     Good baseline but NOT neuron-specific.
#
#   weight_by_neuron=True   (neuron-weighted — our default, more comparable to IxG):
#     Use the target neuron's per-token activations at the target layer
#     as row-weights over R. This produces a neuron-specific map:
#         a_n ∈ R^N = activations[0, :, channel_id] at target_layer
#         per_patch = a_n @ R  ∈ R^N   (each patch's contribution to
#                                        ALL tokens, weighted by this
#                                        neuron's per-token importance)
#     Matches how CRP/CRP-based methods give per-neuron attributions.
#     Because IxG is also neuron-specific, using this mode makes
#     visual comparisons meaningful.
#
# After aggregation, the N-vector is:
#   - dropped of the CLS token (index 0, since CLS is not spatial)
#   - reshaped from [196] → [14, 14]
#   - upsampled (bilinear) → [224, 224]
#
# timm-specific implementation note:
#
#   timm.models.vision_transformer.Attention uses F.scaled_dot_product_attention
#   (via self.fused_attn = True) in recent versions. That path skips any
#   intermediate softmax tensor, so forward hooks can't capture the weights.
#
#   The fix (recipe from huggingface/pytorch-image-models Discussion #2141):
#     1. For each block, set blocks[k].attn.fused_attn = False so the
#        non-fused code path runs — which explicitly computes
#        (q @ k.transpose(-2,-1) / sqrt(d_head)).softmax(dim=-1)
#     2. Monkey-patch blocks[k].attn.forward with a verbatim copy of that
#        non-fused path, AND stash the post-softmax attention tensor on
#        attn_block.last_attn for us to read out after the forward pass.
#
#   We do this INSIDE compute_saliency so the patching is scoped to each
#   call — the model is restored to its original state after (no permanent
#   mutation of the shared model object).
#
# Notes on numerical stability:
#   - We work in float32 throughout. The N=197 matrix products are cheap
#     (micro-seconds) even for 12 blocks.
#   - There's no backward pass — this method is ~30× faster than IxG.
# =============================================================================

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import XAIMethod


class AttentionRollout(XAIMethod):
    """
    Abnar & Zuidema 2020 attention rollout for ViT.

    Config schema (cfg.xai.attention_rollout.*):
        include_identity  : bool, default True
                            Whether to add I to each layer's attention before
                            the matrix product (models residual connections).
                            Set False to get "raw rollout" without the
                            residual correction (not recommended — breaks
                            the paper's contract).
        head_agg          : "mean" | "max", default "mean"
                            How to combine per-head attention into a single
                            [N, N] matrix per layer. Paper uses mean.
        weight_by_neuron  : bool, default True
                            See top-of-file notes. False gives vanilla
                            CLS-row rollout; True gives neuron-specific.
        discard_ratio     : float in [0, 1), default 0.0
                            Optional — zero out this fraction of the lowest
                            attention weights per layer before composition.
                            Used in some ViT rollout variants to denoise.
                            0.0 = disabled (paper's original formulation).

    This class is ViT-only. validate_config raises on conv models.
    """

    name = "attention_rollout"

    # ------------------------------------------------------------------
    # Construction — read config, store options
    # ------------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        self.include_identity = bool(self.method_cfg.get("include_identity", True))
        self.head_agg         = str(self.method_cfg.get("head_agg", "mean"))
        self.weight_by_neuron = bool(self.method_cfg.get("weight_by_neuron", True))
        self.discard_ratio    = float(self.method_cfg.get("discard_ratio", 0.0))

        if self.head_agg not in ("mean", "max"):
            raise ValueError(
                f"xai.attention_rollout.head_agg must be 'mean' or 'max' "
                f"(got {self.head_agg!r})."
            )
        if not (0.0 <= self.discard_ratio < 1.0):
            raise ValueError(
                f"xai.attention_rollout.discard_ratio must be in [0, 1) "
                f"(got {self.discard_ratio})."
            )

        print(
            f"[AttentionRollout] initialized: "
            f"include_identity={self.include_identity}, "
            f"head_agg={self.head_agg!r}, "
            f"weight_by_neuron={self.weight_by_neuron}, "
            f"discard_ratio={self.discard_ratio}"
        )

    # ------------------------------------------------------------------
    # Guard: ViT only
    # ------------------------------------------------------------------
    def validate_config(self, layer_type: str) -> None:
        if layer_type != "linear":
            raise ValueError(
                f"AttentionRollout is ViT-only: it requires cfg.model.layer_type "
                f"== 'linear' (got {layer_type!r}). For CNNs (rn152, vgg16, ...) "
                f"use xai.method='ixg' or 'ig' instead."
            )

    # ------------------------------------------------------------------
    # Method-specific metadata for Stage 3 output JSON
    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        return {
            "include_identity": self.include_identity,
            "head_agg":         self.head_agg,
            "weight_by_neuron": self.weight_by_neuron,
            "discard_ratio":    self.discard_ratio,
        }

    # ------------------------------------------------------------------
    # Monkey-patch helper
    # ------------------------------------------------------------------
    def _install_attention_capture(self, model: nn.Module) -> List[Any]:
        """
        Install a monkey-patched forward on every `blocks[k].attn` so that
        after a forward pass, the post-softmax attention tensor is stored
        on `attn_block.last_attn` with shape [B, num_heads, N, N].

        Returns:
            A list of (attn_block, original_forward, original_fused_flag)
            tuples so the caller can restore state in a finally block.

        Recipe source: huggingface/pytorch-image-models Discussion #2141
        (maintainers' own workaround for extracting attention in fused mode).
        The non-fused path we re-implement is a direct copy of the code in
        timm/models/vision_transformer.py's Attention.forward.
        """
        if not hasattr(model, "blocks"):
            raise RuntimeError(
                "AttentionRollout expected the model to have a `.blocks` "
                "attribute (timm ViT convention). Got "
                f"{type(model).__name__} instead."
            )

        state: List[Any] = []

        def _make_capturing_forward(attn_block):
            # Save originals so we can restore
            orig_forward = attn_block.forward
            orig_fused   = getattr(attn_block, "fused_attn", False)

            # Force non-fused so we can inject our custom forward
            attn_block.fused_attn = False

            # Ensure a nn.Softmax submodule exists (timm's non-fused path
            # uses `attn.softmax(dim=-1)` as a tensor method — we replace
            # with an explicit module so we COULD also hook it if wanted).
            # This is harmless even if the attribute already existed.
            attn_block.softmax = nn.Softmax(dim=-1)

            # Placeholder to hold the captured weights after forward runs.
            attn_block.last_attn = None

            def capturing_forward(x: torch.Tensor, **kwargs) -> torch.Tensor:
                """
                Verbatim copy of timm's non-fused Attention.forward, plus:
                  - stash the post-softmax attention matrix on
                    attn_block.last_attn for the rollout to read.
                """
                B, N, C = x.shape
                # qkv: [B, N, 3*C] -> [3, B, heads, N, head_dim]
                qkv = attn_block.qkv(x).reshape(
                    B, N, 3, attn_block.num_heads, attn_block.head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                # q_norm / k_norm — present in newer timm, may be Identity.
                q_norm = getattr(attn_block, "q_norm", nn.Identity())
                k_norm = getattr(attn_block, "k_norm", nn.Identity())
                q = q_norm(q)
                k = k_norm(k)

                # Scaled dot product
                q = q * attn_block.scale
                attn = q @ k.transpose(-2, -1)           # [B, heads, N, N]

                # --- FIX: Handle attn_mask passed by newer timm versions ---
                attn_mask = kwargs.get("attn_mask", None)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn.masked_fill_(~attn_mask, float("-inf"))
                    else:
                        attn += attn_mask
                # -----------------------------------------------------------

                attn = attn_block.softmax(attn)          # [B, heads, N, N]

                # ── stash BEFORE dropout so we get clean probabilities ──
                attn_block.last_attn = attn.detach()

                attn = attn_block.attn_drop(attn)
                x_out = attn @ v                          # [B, heads, N, head_dim]
                x_out = x_out.transpose(1, 2).reshape(B, N, C)
                x_out = attn_block.proj(x_out)
                x_out = attn_block.proj_drop(x_out)
                return x_out

            attn_block.forward = capturing_forward
            state.append((attn_block, orig_forward, orig_fused))

        # Walk each block and install capturing forward
        for blk in model.blocks:
            if not hasattr(blk, "attn"):
                continue  # not a standard ViT block; skip quietly
            _make_capturing_forward(blk.attn)

        if not state:
            raise RuntimeError(
                "AttentionRollout installed capturing forwards on 0 blocks. "
                "Is this really a ViT? model.blocks exists but no .attn "
                "submodules were found."
            )
        return state

    def _uninstall_attention_capture(self, state: List[Any]) -> None:
        """Restore the original `forward` and `fused_attn` on each attn block."""
        for attn_block, orig_forward, orig_fused in state:
            attn_block.forward    = orig_forward
            attn_block.fused_attn = orig_fused
            if hasattr(attn_block, "last_attn"):
                attn_block.last_attn = None  # drop reference to free memory

    # ------------------------------------------------------------------
    # Rollout math helpers
    # ------------------------------------------------------------------
    def _aggregate_heads(self, attn_heads: torch.Tensor) -> torch.Tensor:
        """
        Collapse the heads dimension of [num_heads, N, N] → [N, N].
        """
        if self.head_agg == "mean":
            return attn_heads.mean(dim=0)
        # head_agg == "max"
        return attn_heads.max(dim=0).values

    def _apply_discard_ratio(self, A: torch.Tensor) -> torch.Tensor:
        """
        Zero out the `discard_ratio` fraction of smallest attention weights
        per row, then RE-normalize each row to sum to 1.
        Diagonal entries (self-attention) are preserved — zeroing the
        diagonal breaks the residual-connection interpretation.
        """
        if self.discard_ratio <= 0.0:
            return A
        N = A.shape[0]
        k_drop = int(N * self.discard_ratio)
        if k_drop == 0:
            return A

        A = A.clone()
        # Work out the smallest-k indices per row (excluding the diagonal).
        # Save the diagonal, sort, then restore.
        for i in range(N):
            row = A[i].clone()
            row[i] = float("inf")  # protect the diagonal from being zeroed
            _, small_idx = row.topk(k_drop, largest=False)
            A[i, small_idx] = 0.0

        # Re-normalize rows so they still sum to ~1 (residual-preserving).
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
        return A

    def _compute_rollout(
        self,
        attn_maps: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Given per-layer attention matrices (already head-aggregated, [N, N]
        each), compute the rollout matrix through all layers.

        Order matters: rollout traverses from input → output, so if attn_maps
        is [A_0, A_1, ..., A_{L-1}] we want:
            R = Ā_{L-1} @ Ā_{L-2} @ ... @ Ā_0

        Returns:
            [N, N] rollout matrix on the same device as the inputs.
        """
        N = attn_maps[0].shape[0]
        device = attn_maps[0].device
        dtype  = attn_maps[0].dtype

        I = torch.eye(N, device=device, dtype=dtype)

        def _residual_correct(A):
            if self.include_identity:
                A_res = A + I
                # Row-normalize so each row sums to 1 again.
                A_res = A_res / (A_res.sum(dim=-1, keepdim=True) + 1e-12)
            else:
                A_res = A
            # Optional sparsification
            A_res = self._apply_discard_ratio(A_res)
            return A_res

        # Start with the first layer's corrected attention
        rollout = _residual_correct(attn_maps[0])
        # Compose later layers on top: rollout = Ā_k @ rollout
        for A in attn_maps[1:]:
            A_res  = _residual_correct(A)
            rollout = A_res @ rollout

        return rollout

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
        Compute attention rollout saliency for a single preprocessed image.

        Args:
            model          : timm ViT, in eval mode, on correct device
            input_tensor   : [1, 3, H, W] preprocessed (typically [1, 3, 224, 224])
            target_layer   : the nn.Module for the target layer. Used ONLY
                             when weight_by_neuron=True, to extract per-token
                             activations for the target channel.
            target_channel : neuron id within target_layer's output
            layer_type     : must be "linear" (guarded by validate_config)

        Returns:
            [H, W] float32 numpy array, same H×W as input. Higher = more
            attention-flow-based attribution for this neuron/image.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be [1, 3, H, W], got {tuple(input_tensor.shape)}"
            )

        H_img, W_img = input_tensor.shape[2], input_tensor.shape[3]
        device = input_tensor.device

        # Install attention capturing on every block. Restored in finally.
        capture_state = self._install_attention_capture(model)

        # Hook the target layer (only needed if weight_by_neuron=True),
        # but we install it unconditionally — it's cheap and simplifies
        # control flow. The captured tensor is detached in the hook.
        target_captured: Dict[str, torch.Tensor] = {}

        def _target_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                target_captured["act"] = output[0].detach()
            else:
                target_captured["act"] = output.detach()

        target_handle = target_layer.register_forward_hook(_target_hook)

        try:
            # ── Forward pass only; no gradients needed ──────────────
            with torch.no_grad():
                _ = model(input_tensor)

            # ── Gather the captured per-layer attention matrices ────
            per_layer_attn: List[torch.Tensor] = []
            for blk in model.blocks:
                if not hasattr(blk, "attn"):
                    continue
                A_heads = blk.attn.last_attn  # [B, heads, N, N]
                if A_heads is None:
                    raise RuntimeError(
                        "A block's capturing forward never fired. The model's "
                        "forward pass may have skipped that block."
                    )
                # Batch dim is 1 — squeeze it out, then aggregate heads
                A = A_heads[0]                # [heads, N, N]
                A = self._aggregate_heads(A)  # [N, N]
                per_layer_attn.append(A.to(torch.float32))

            # ── Rollout matrix ─────────────────────────────────────
            rollout = self._compute_rollout(per_layer_attn)  # [N, N]
            N = rollout.shape[0]

            # ── Aggregate to per-token scores ──────────────────────
            if self.weight_by_neuron:
                # Pull this neuron's per-token activation from the target layer
                if "act" not in target_captured:
                    raise RuntimeError(
                        "target_layer hook did not fire — check that target_layer "
                        "was resolved correctly."
                    )
                act = target_captured["act"]  # [1, N, hidden_dim]
                if act.dim() != 3:
                    raise RuntimeError(
                        f"AttentionRollout expected target activations of "
                        f"shape [1, N, hidden_dim]; got {tuple(act.shape)}."
                    )
                # Per-token scalar for this neuron
                weights = act[0, :, target_channel].to(torch.float32)  # [N]
                # Each output token's importance is weights[i]; its dependence
                # on input token j is rollout[i, j]. Per-patch importance is:
                #       per_patch[j] = Σ_i weights[i] * rollout[i, j]
                # = weights @ rollout   (row-vector times matrix)
                per_patch = weights @ rollout  # [N]
            else:
                # Vanilla CLS-row rollout (Abnar & Zuidema default)
                per_patch = rollout[0]  # row 0 = CLS → all tokens  [N]

            # ── Drop CLS (index 0) and reshape to spatial grid ─────
            # ViT-B/16: 197 tokens → 1 CLS + 196 patches (14×14)
            patch_scores = per_patch[1:]
            num_patches  = patch_scores.shape[0]
            import math
            grid_size = int(math.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                raise RuntimeError(
                    f"AttentionRollout: num_patches={num_patches} is not a "
                    f"perfect square. Non-square patch grids are not supported."
                )

            grid = patch_scores.reshape(grid_size, grid_size)  # [14, 14]

            # ── Upsample to input resolution ────────────────────────
            # F.interpolate expects [N, C, H, W], so add two leading dims
            grid_4d = grid.unsqueeze(0).unsqueeze(0)            # [1, 1, 14, 14]
            upsampled = F.interpolate(
                grid_4d, size=(H_img, W_img),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)                              # [H, W]

            # ── ABS — Attention rollout can produce negative values
            # only when weight_by_neuron=True AND the neuron's per-token
            # activations contain negatives (possible pre-GELU in fc2 output,
            # depends on what layer was chosen). Take abs so downstream
            # get_crop_bbox / alpha-mask logic sees a non-negative saliency
            # consistent with IxG abs_output=True.
            saliency = upsampled.abs().cpu().numpy().astype(np.float32)

        finally:
            target_handle.remove()
            self._uninstall_attention_capture(capture_state)

        return saliency