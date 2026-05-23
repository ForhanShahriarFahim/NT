# =============================================================================
# FILE: neuron_viz_pipeline/src/extract/activations.py
#
# Purpose:
#   ActivationExtractor — registers forward hooks on target layer(s), runs
#   the model over a DataLoader, captures activations per batch, and saves
#   them to a single .safetensors file.
#
# Source:
#   VERBATIM port of ActivationExtractor from pipeline_a/extract.py.
#   Original logic was adapted from preprocessing/extract_activations.py
#   (your three reference files).
#
# Changes vs pipeline_a/extract.py:
#   1. Removed inline `_save_safetensors_streaming` method — now imports
#      from src.utils.io_safetensors.save_safetensors_streaming (shared
#      across Stages 1, 2, 3 to avoid duplication).
#   2. Removed argparse/main() section — the runner is now a separate file
#      (scripts/stage1_extract.py) for cleaner separation of concerns.
#   3. Everything else (hook logic, pooling, checkpoint combine via memmap)
#      is VERBATIM from your working code.
#
# What is VERBATIM (no algorithmic changes):
#   - _register_hooks, _get_layer_by_name  — hook registration
#   - extract()                              — main DataLoader loop
#   - _save_intermediate_checkpoint          — per-N-batch checkpoint saves
#   - _apply_pooling                         — raw / gap / gmp modes
#   - _save_layer_by_layer                   — fallback path (non-streaming)
#   - _combine_checkpoint_files              — memmap-based final combine
#   - cleanup, __del__                       — hook cleanup
# =============================================================================

import os
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Shared RAM-safe helper from our utils package
from src.utils.io_safetensors import save_safetensors_streaming


class ActivationExtractor:
    """
    Extract activations from one or more layers of a model.

    Workflow:
        extractor = ActivationExtractor(model, [("layer4.2.conv3", "output")])
        extractor.extract(
            data_loader,
            save_dir="results/.../activations/",
            save_intermediate=True,
            pool_type="raw",
            checkpoint_interval=50
        )
        extractor.cleanup()

    What it produces:
        {save_dir}/activations_{layer}_{target_type}_{pool_type}.safetensors
        {save_dir}/activations_{layer}_{target_type}_{pool_type}_metadata.txt
        + intermediate checkpoints under {save_dir}/checkpoints/ (auto-cleaned
          after combine).
    """

    def __init__(self, model: nn.Module, layers_to_hook: List[Tuple[str, str]]):
        """
        Args:
            model          : nn.Module (will be set to eval mode during extract)
            layers_to_hook : list of (layer_name, target_type) tuples
                             target_type is "output" or "input" — which side
                             of the layer the forward hook captures.
                             Typical: [("layer4.2.conv3", "output")]
        """
        self.model          = model
        self.layers_to_hook = layers_to_hook
        self.hooks: Dict[Tuple[str, str], torch.utils.hooks.RemovableHandle] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration — VERBATIM from pipeline_a/extract.py
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Register a forward hook on every (layer_name, target_type) pair."""
        def get_activation(name, target_type="output"):
            def hook(module, input, output):
                tensor_to_save = None
                if target_type == "output":
                    if isinstance(output, torch.Tensor):
                        tensor_to_save = output
                    elif isinstance(output, (tuple, list)):
                        tensor_to_save = output[0]
                elif target_type == "input":
                    if isinstance(input, (tuple, list)) and len(input) > 0:
                        tensor_to_save = input[0]
                if tensor_to_save is not None:
                    self.activations[name] = tensor_to_save.detach().cpu()
            return hook

        for layer_name, target_type in self.layers_to_hook:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                handle = layer.register_forward_hook(
                    get_activation(layer_name, target_type)
                )
                self.hooks[(layer_name, target_type)] = handle
            else:
                raise ValueError(f"Layer '{layer_name}' not found in model")

    def _get_layer_by_name(self, layer_name: str):
        """
        Resolve a dotted layer path to a submodule.
        Supports mixed attribute access and numeric indices (for nn.Sequential):
            "layer4.2.conv3"     → model.layer4[2].conv3
            "blocks.11.mlp.fc2"  → model.blocks[11].mlp.fc2
        """
        parts = layer_name.split(".")
        print(f"  [hook] resolving '{layer_name}' → parts {parts}")
        layer = self.model
        try:
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    if hasattr(layer, "__getitem__"):
                        layer = layer[idx]
                    else:
                        try:
                            layer = getattr(layer, part)
                        except AttributeError:
                            print(f"  [hook] could not access index {idx}")
                            return None
                else:
                    if hasattr(layer, part):
                        layer = getattr(layer, part)
                    else:
                        print(f"  [hook] attribute '{part}' not found")
                        return None
            return layer
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            print(f"  [hook] error accessing layer '{layer_name}': {e}")
            return None

    # ------------------------------------------------------------------
    # Main extract loop — VERBATIM from pipeline_a/extract.py
    # ------------------------------------------------------------------

    def extract(
        self,
        data_loader: DataLoader,
        save_dir: str = None,
        save_intermediate: bool = False,
        pool_type: str = "raw",
        checkpoint_interval: int = 100,
        channel_id_only: int = None,
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """
        Run the model over the DataLoader and capture activations.

        Three modes based on `save_dir` + `save_intermediate`:
          save_dir is None → keeps everything in RAM (small datasets only)
          save_dir + save_intermediate=False → accumulates in RAM, saves once
          save_dir + save_intermediate=True  → streams to disk in checkpoints
                                                (the recommended mode)

        pool_type:
          "raw" — keep full spatial activations (required for IxG)
          "gap" — global average pool over spatial dims
          "gmp" — global max pool over spatial dims

        channel_id_only:
          Optional memory/disk optimization for single-neuron runs. If set,
          only that channel/hidden dimension is saved while preserving the
          sequence/spatial axes needed for ranking. For ViT [B, tokens, hidden]
          activations, channel_id_only=652 saves [B, tokens, 1].
        """
        all_activations = {
            (name, target_type): []
            for name, target_type in self.layers_to_hook
        }

        self.model.eval()
        checkpoint_counter = 0

        with torch.no_grad():
            for batch_idx, (inputs, _targets) in enumerate(
                tqdm(data_loader, desc="Extracting activations")
            ):
                inputs = inputs.to(next(self.model.parameters()).device)
                self.activations.clear()
                _ = self.model(inputs)

                for j, (layer_name, target_type) in enumerate(self.layers_to_hook):
                    if layer_name in self.activations:
                        if batch_idx == 0 and j == 0:
                            print(f"  layer {layer_name} activation shape: "
                                  f"{self.activations[layer_name].shape}")
                        act = self.activations[layer_name]

                        # Single-neuron extraction path. This keeps the real
                        # channel id's activation only, reducing ViT tensors
                        # from [B, tokens, hidden_dim] to [B, tokens, 1].
                        if channel_id_only is not None:
                            ch = int(channel_id_only)
                            if act.dim() == 4:
                                if not (0 <= ch < act.shape[1]):
                                    raise IndexError(
                                        f"channel_id_only={ch} out of range for "
                                        f"4D activation with {act.shape[1]} channels"
                                    )
                                act = act[:, ch:ch + 1, ...]
                            elif act.dim() == 3:
                                if not (0 <= ch < act.shape[2]):
                                    raise IndexError(
                                        f"channel_id_only={ch} out of range for "
                                        f"3D activation with hidden_dim={act.shape[2]}"
                                    )
                                act = act[:, :, ch:ch + 1]
                            elif act.dim() == 2:
                                if not (0 <= ch < act.shape[1]):
                                    raise IndexError(
                                        f"channel_id_only={ch} out of range for "
                                        f"2D activation with {act.shape[1]} features"
                                    )
                                act = act[:, ch:ch + 1]
                            else:
                                raise ValueError(
                                    f"channel_id_only requires 2D/3D/4D activations, "
                                    f"got shape {tuple(act.shape)}"
                                )
                            if batch_idx == 0 and j == 0:
                                print(f"  saving only channel {ch}: shape {act.shape}")

                        all_activations[(layer_name, target_type)].append(
                            act
                        )

                # Checkpoint save every N batches
                if save_intermediate and save_dir and \
                        (batch_idx + 1) % checkpoint_interval == 0:
                    checkpoint_counter += 1
                    self._save_intermediate_checkpoint(
                        all_activations, save_dir, checkpoint_counter, pool_type
                    )
                    for key in all_activations:
                        all_activations[key].clear()

        # Save any remaining batches in final checkpoint
        if save_intermediate and save_dir and any(all_activations.values()):
            checkpoint_counter += 1
            self._save_intermediate_checkpoint(
                all_activations, save_dir, checkpoint_counter, pool_type
            )
            for key in all_activations:
                all_activations[key].clear()

        if save_dir:
            if save_intermediate:
                self._combine_checkpoint_files(save_dir, checkpoint_counter, pool_type)
            else:
                self._save_layer_by_layer(all_activations, save_dir, pool_type)
            return {}
        else:
            # Return in-memory activations for small-dataset cases
            result = {}
            for layer_name, target_type in self.layers_to_hook:
                if all_activations[(layer_name, target_type)]:
                    result[(layer_name, target_type)] = torch.cat(
                        all_activations[(layer_name, target_type)], dim=0
                    )
            return result

    # ------------------------------------------------------------------
    # Checkpoint handling — VERBATIM from pipeline_a/extract.py
    # ------------------------------------------------------------------

    def _save_intermediate_checkpoint(self, all_activations, save_dir,
                                       checkpoint_num, pool_type):
        """Save each layer's accumulated batches as one .pt file per checkpoint."""
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        for layer_name, target_type in self.layers_to_hook:
            if all_activations[(layer_name, target_type)]:
                acts = torch.cat(all_activations[(layer_name, target_type)], dim=0)
                cp_path = os.path.join(
                    checkpoint_dir, f"{layer_name}_batch_{checkpoint_num}.pt"
                )
                torch.save(acts, cp_path)
                print(f"  checkpoint {checkpoint_num}: {layer_name} "
                      f"shape {acts.shape} → {cp_path}")
                del acts

    def _apply_pooling(self, activation: torch.Tensor, pool_type: str) -> torch.Tensor:
        """Apply raw / gap / gmp pooling over spatial dims."""
        if activation.dim() <= 2:
            return activation
        if pool_type == "gap":
            return activation.mean(dim=list(range(2, activation.dim())))
        elif pool_type == "gmp":
            for dim in range(activation.dim() - 1, 1, -1):
                activation = activation.max(dim=dim)[0]
            return activation
        elif pool_type == "raw":
            return activation
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

    def _save_layer_by_layer(self, all_activations, save_dir, pool_type):
        """
        In-RAM save path (used when save_intermediate=False).
        Needs the full tensor to fit in RAM — only safe for small datasets.
        """
        os.makedirs(save_dir, exist_ok=True)
        for layer_name, target_type in self.layers_to_hook:
            if all_activations[(layer_name, target_type)]:
                print(f"  in-memory save: {layer_name}")
                acts = torch.cat(all_activations[(layer_name, target_type)], dim=0)
                acts = self._apply_pooling(acts, pool_type)
                print(f"    final shape: {acts.shape}")

                safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                filename = (f"activations_{safe_layer_name}"
                            f"_{target_type}_{pool_type}.safetensors")
                save_path = os.path.join(save_dir, filename)

                # Use shared streaming writer even in the in-RAM case —
                # it still works and gives consistent file format.
                save_safetensors_streaming(
                    layer_name=layer_name,
                    arr=acts.numpy(),
                    save_path=save_path,
                    chunk_size=1000,
                )

                self._write_metadata(save_path, layer_name, acts.shape,
                                      pool_type, acts.dtype)
                del acts
                all_activations[(layer_name, target_type)].clear()

    def _combine_checkpoint_files(self, save_dir, checkpoint_counter, pool_type):
        """
        Two-pass memmap combine:
          Pass 1: scan checkpoints to count total samples and discover shape
          Pass 2: write each checkpoint into a disk-backed memmap one at a time
          Final:  stream from memmap → safetensors in chunks (no full-RAM copy)

        Peak RAM = one checkpoint at a time (~1.3 GB for rn152).
        """
        checkpoint_dir = os.path.join(save_dir, "checkpoints")

        for layer_name, target_type in self.layers_to_hook:

            # ── Pass 1: count total samples and get feature shape ──
            total_samples = 0
            feature_shape = None
            for i in range(1, checkpoint_counter + 1):
                cp_path = os.path.join(
                    checkpoint_dir, f"{layer_name}_batch_{i}.pt"
                )
                if not os.path.exists(cp_path):
                    continue
                try:
                    cp = torch.load(cp_path, map_location="cpu")
                    cp = self._apply_pooling(cp, pool_type)
                    if feature_shape is None:
                        feature_shape = list(cp.shape[1:])
                    total_samples += cp.shape[0]
                    del cp
                except Exception as e:
                    print(f"  cannot read {cp_path}: {e}")

            if feature_shape is None or total_samples == 0:
                print(f"  no valid checkpoints for '{layer_name}' — skipping")
                continue

            full_shape = tuple([total_samples] + feature_shape)
            print(f"\n  combining {checkpoint_counter} checkpoints for "
                  f"'{layer_name}'")
            print(f"    total shape: {full_shape}")

            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            filename = (f"activations_{safe_layer_name}"
                        f"_{target_type}_{pool_type}.safetensors")
            save_path = os.path.join(save_dir, filename)
            tmp_npy   = save_path + ".tmp.npy"

            try:
                # ── Pass 2: write each checkpoint into disk memmap ──
                memmap_arr = np.lib.format.open_memmap(
                    tmp_npy, mode="w+", dtype=np.float32, shape=full_shape
                )

                offset = 0
                for i in range(1, checkpoint_counter + 1):
                    cp_path = os.path.join(
                        checkpoint_dir, f"{layer_name}_batch_{i}.pt"
                    )
                    if not os.path.exists(cp_path):
                        continue
                    try:
                        cp = torch.load(cp_path, map_location="cpu")
                        cp = self._apply_pooling(cp, pool_type)
                        n  = cp.shape[0]
                        memmap_arr[offset: offset + n] = cp.numpy()
                        memmap_arr.flush()
                        offset += n
                        del cp
                        print(f"    written {offset}/{total_samples} "
                              f"(checkpoint {i}/{checkpoint_counter})")
                    except Exception as e:
                        print(f"    error in checkpoint {i}: {e}")

                # ── Final: stream memmap → safetensors in chunks ──
                save_safetensors_streaming(
                    layer_name=layer_name,
                    arr=memmap_arr,
                    save_path=save_path,
                    chunk_size=1000,
                )

                # Save metadata alongside
                self._write_metadata(
                    save_path, layer_name, list(full_shape), pool_type,
                    "float32", num_checkpoints=checkpoint_counter
                )

                # Cleanup temp memmap
                del memmap_arr
                if os.path.exists(tmp_npy):
                    os.remove(tmp_npy)
                    print(f"    temp memmap file removed")

                # Cleanup checkpoint .pt files
                removed = 0
                for i in range(1, checkpoint_counter + 1):
                    cp = os.path.join(
                        checkpoint_dir, f"{layer_name}_batch_{i}.pt"
                    )
                    if os.path.exists(cp):
                        try:
                            os.remove(cp)
                            removed += 1
                        except OSError:
                            pass
                print(f"    removed {removed} checkpoint .pt files")

            except Exception as e:
                print(f"  error during combine for '{layer_name}': {e}")
                if os.path.exists(tmp_npy):
                    try:
                        os.remove(tmp_npy)
                    except OSError:
                        pass
                raise

        # Remove checkpoints directory if empty
        try:
            if os.path.exists(checkpoint_dir) and not os.listdir(checkpoint_dir):
                os.rmdir(checkpoint_dir)
        except OSError:
            pass

    @staticmethod
    def _write_metadata(save_path, layer_name, shape, pool_type, dtype,
                         num_checkpoints=None):
        """Write the companion _metadata.txt file (matches pipeline_a format)."""
        metadata = {
            "layer_name": layer_name,
            "shape":      str(list(shape)),
            "pool_type":  pool_type,
            "dtype":      str(dtype),
        }
        if num_checkpoints is not None:
            metadata["num_checkpoints"] = str(num_checkpoints)
        meta_path = save_path.replace(".safetensors", "_metadata.txt")
        with open(meta_path, "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove all registered forward hooks. Safe to call multiple times."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

    def __del__(self):
        self.cleanup()
