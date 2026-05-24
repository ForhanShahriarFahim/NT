# =============================================================================
# Activation Viz: activation_viz/extract.py
#
# ORIGIN: Adapted from preprocessing/extract_activations.py
#
# WHAT CHANGED vs original:
#   1. Removed: from dsets import get_dataset
#               from models import get_fn_model_loader
#               from utils.helper import load_config, get_layer_names_model
#               from timm.data import resolve_data_config
#               from timm.data.transforms_factory import create_transform
#   2. Added:   from data.data_proces import MakeDataset  (CoE dataset loader)
#               from models import build_models            (CoE model loader)
#   3. Added:   _make_args() helper to create minimal namespace for CoE loaders
#   4. Added:   LAYER_DEFAULTS dict for each model
#   5. main():  replaced config-based loading with direct CLI args
#               replaced get_dataset() with MakeDataset()
#               replaced get_fn_model_loader() with build_models()
#   6. Removed: --config_file, --print_model_structure, --structure_depth args
#   7. Added:   --model_name, --data_path, --dataset args
#
# WHAT IS IDENTICAL to original:
#   - ActivationExtractor class (verbatim — hooks, pooling, checkpointing)
#   - All pooling logic (gap, gmp, raw)
#   - All checkpoint/combine logic for --save_intermediate
#   - safetensors output format
#   - All CLI args except those listed above
# =============================================================================

import argparse
import os
import types
from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from safetensors.torch import save_file

# Activation Viz CHANGE: use CoE loaders instead of dsets/models
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.data_proces import MakeDataset
from models import build_models

import torchvision.transforms as T


# =============================================================================
# Activation Viz: layer defaults per model
# These match the examples in the original extract_activations.py docstring.
# =============================================================================
LAYER_DEFAULTS = {
    'rn152':                'layer4.2.conv3',
    'rn50':                 'layer4.2.conv3',
    'vgg16':                'features.28',
    'vit':                  'blocks.11',
    'vit_base_patch16_224': 'blocks.11',
}


def _make_args(model_name, data_path, dataset='imagenet-val', resume=None):
    """
    Activation Viz: Create minimal args namespace for CoE's build_models()
    and MakeDataset(). CoE loaders expect an args object — this creates one
    from plain strings without needing the full argparse setup.
    """
    a = types.SimpleNamespace()
    a.model_name    = model_name
    a.resume        = resume
    a.dataset       = dataset
    a.dataset_name  = dataset.split('-')[0]          # 'imagenet'
    a.dataset_split = dataset.split('-')[1]           # 'val'
    a.data_dir      = os.path.join(data_path, a.dataset_name)
    return a


# =============================================================================
# ActivationExtractor — VERBATIM from extract_activations.py
# No changes to this class. All hook logic, pooling, checkpointing preserved.
# =============================================================================

class ActivationExtractor:
    """Extract activations from multiple layers of a model"""

    def __init__(self, model: nn.Module, layers_to_hook: List):
        self.model          = model
        self.layers_to_hook = layers_to_hook
        self.hooks          = {}
        self.activations    = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for specified layers"""
        def get_activation(name, target_type='output'):
            def hook(model, input, output):
                tensor_to_save = None
                if target_type == 'output':
                    if isinstance(output, torch.Tensor):
                        tensor_to_save = output
                    elif isinstance(output, (tuple, list)):
                        tensor_to_save = output[0]
                elif target_type == 'input':
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
                print(f"Warning: Layer '{layer_name}' not found in model")
                raise ValueError(f"Layer '{layer_name}' not found in model")

    def _get_layer_by_name(self, layer_name: str, model=None):
        """Get layer object by name, supporting nested paths like 'encoder.blocks.0'"""
        parts = layer_name.split('.')
        print(f"Getting layer by name: {layer_name} - {parts}")
        layer = self.model
        try:
            for i, part in enumerate(parts):
                if part.isdigit():
                    idx = int(part)
                    if hasattr(layer, '__getitem__'):
                        layer = layer[idx]
                    else:
                        try:
                            layer = getattr(layer, part)
                        except AttributeError:
                            print(f"Warning: Could not access index {idx}")
                            return None
                else:
                    if hasattr(layer, part):
                        layer = getattr(layer, part)
                    else:
                        print(f"Warning: Attribute '{part}' not found")
                        return None
            return layer
        except (AttributeError, IndexError, TypeError, KeyError) as e:
            print(f"Error accessing layer '{layer_name}': {e}")
            return None

    def extract(self, data_loader: DataLoader, save_dir: str = None,
                save_intermediate: bool = False, pool_type: str = "raw",
                checkpoint_interval: int = 100) -> Dict[str, torch.Tensor]:
        """Extract activations for all samples in the data loader"""
        all_activations = {
            (name, target_type): []
            for name, target_type in self.layers_to_hook
        }

        self.model.eval()
        checkpoint_counter = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(
                tqdm(data_loader, desc="Extracting activations")
            ):
                inputs = inputs.to(next(self.model.parameters()).device)
                self.activations.clear()
                _ = self.model(inputs)

                for j, (layer_name, target_type) in enumerate(self.layers_to_hook):
                    if layer_name in self.activations:
                        if j == 0:
                            print(f"Layer {layer_name} has "
                                  f"{self.activations[layer_name].shape} activations")
                        all_activations[(layer_name, target_type)].append(
                            self.activations[layer_name]
                        )

                if save_intermediate and save_dir and \
                        (batch_idx + 1) % checkpoint_interval == 0:
                    checkpoint_counter += 1
                    self._save_intermediate_checkpoint(
                        all_activations, save_dir, checkpoint_counter, pool_type
                    )
                    for key in all_activations:
                        all_activations[key].clear()

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
            result = {}
            for layer_name, target_type in self.layers_to_hook:
                if all_activations[(layer_name, target_type)]:
                    result[(layer_name, target_type)] = torch.cat(
                        all_activations[(layer_name, target_type)], dim=0
                    )
                else:
                    print(f"Warning: No activations for '{layer_name}'")
            return result

    def _save_intermediate_checkpoint(self, all_activations, save_dir,
                                       checkpoint_num, pool_type):
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        for layer_name, target_type in self.layers_to_hook:
            if all_activations[(layer_name, target_type)]:
                activations = torch.cat(
                    all_activations[(layer_name, target_type)], dim=0
                )
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"{layer_name}_batch_{checkpoint_num}.pt"
                )
                torch.save(activations, checkpoint_path)
                print(f"Saved checkpoint {checkpoint_num} for {layer_name} "
                      f"shape {activations.shape}")
                del activations

    def _apply_pooling(self, activation: torch.Tensor, pool_type: str) -> torch.Tensor:
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
            raise ValueError(f"Unknown pooling type: {pool_type}")

    def _save_layer_by_layer(self, all_activations, save_dir, pool_type):
        os.makedirs(save_dir, exist_ok=True)
        for layer_name, target_type in self.layers_to_hook:
            if all_activations[(layer_name, target_type)]:
                print(f"Processing and saving layer: {layer_name}")
                activations = torch.cat(
                    all_activations[(layer_name, target_type)], dim=0
                )
                activations = self._apply_pooling(activations, pool_type)
                print(f"  Final shape: {activations.shape}")
                safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
                filename = (f"activations_{safe_layer_name}"
                            f"_{target_type}_{pool_type}.safetensors")
                save_path = os.path.join(save_dir, filename)
                save_file({layer_name: activations}, save_path)
                print(f"  Saved to: {save_path}")
                metadata = {
                    "layer_name": layer_name,
                    "shape":      str(list(activations.shape)),
                    "pool_type":  pool_type,
                    "dtype":      str(activations.dtype)
                }
                meta_path = save_path.replace(".safetensors", "_metadata.txt")
                with open(meta_path, "w") as f:
                    for k, v in metadata.items():
                        f.write(f"{k}: {v}\n")
                del activations
                all_activations[(layer_name, target_type)].clear()
            else:
                print(f"Warning: No activations for layer '{layer_name}'")
    
    @staticmethod
    def _save_safetensors_streaming(layer_name, memmap_arr, save_path,
                                    chunk_size=1000):
        """
        Activation Viz FIX: Write safetensors file by streaming chunks from memmap.

        WHY THIS EXISTS:
        safetensors.torch.save_file() calls data.tobytes() internally which
        copies the entire tensor into RAM at once. For a (50000,2048,7,7)
        float32 tensor that is ~20 GB → MemoryError.

        THIS FIX:
        Writes the safetensors binary format manually:
            [8 bytes uint64: header_len][JSON header][raw bytes streamed in chunks]
        Only chunk_size samples (~1000) are in RAM at any time.
        Peak RAM = chunk_size × 2048 × 49 × 4 bytes ≈ 400 MB for rn152.

        safetensors format spec:
        https://github.com/huggingface/safetensors#format
        """
        import struct
        import json

        dtype_str_map = {
            'float32': 'F32',
            'float16': 'F16',
            'bfloat16': 'BF16',
            'int64':   'I64',
            'int32':   'I32',
        }
        dtype_str  = dtype_str_map.get(str(memmap_arr.dtype), 'F32')
        total_bytes = int(memmap_arr.nbytes)
        shape       = list(memmap_arr.shape)

        # Build JSON header — data_offsets are byte ranges in the data section
        header_dict = {
            layer_name: {
                "dtype":        dtype_str,
                "shape":        shape,
                "data_offsets": [0, total_bytes]
            }
        }
        header_json  = json.dumps(header_dict, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')

        # safetensors requires header to be padded to 8-byte boundary with spaces
        pad = (8 - len(header_bytes) % 8) % 8
        header_bytes += b' ' * pad

        n_samples = memmap_arr.shape[0]

        print(f"  Streaming write: {n_samples} samples in chunks of {chunk_size}")
        print(f"  Output: {save_path}")

        with open(save_path, 'wb') as f:
            # Write 8-byte little-endian uint64 = length of header
            f.write(struct.pack('<Q', len(header_bytes)))
            # Write JSON header
            f.write(header_bytes)
            # Stream data chunks — never more than chunk_size samples in RAM
            written = 0
            for start in range(0, n_samples, chunk_size):
                end   = min(start + chunk_size, n_samples)
                chunk = memmap_arr[start:end]  # reads from disk into RAM
                f.write(chunk.tobytes())       # writes to file
                del chunk                      # immediately free RAM
                written += (end - start)
                if written % 10000 == 0 or end == n_samples:
                    print(f"    Streamed {written}/{n_samples} samples")

        # Verify file size
        actual_size = os.path.getsize(save_path)
        expected_size = 8 + len(header_bytes) + total_bytes
        if actual_size != expected_size:
            raise RuntimeError(
                f"Activation Viz: File size mismatch!\n"
                f"  Expected: {expected_size} bytes\n"
                f"  Actual:   {actual_size} bytes\n"
                f"  Possible incomplete write."
            )
        print(f"  File verified: {actual_size / (1024**3):.2f} GB")

    def _combine_checkpoint_files(self, save_dir, checkpoint_counter, pool_type):
        """
        Activation Viz FIX: Two-pass memmap combine with streaming safetensors write.
        Pass 1: count total samples and get shape (each checkpoint read briefly)
        Pass 2: write each checkpoint into disk-backed memmap one at a time
        Final:  stream safetensors from memmap in chunks (no full RAM copy)
        Peak RAM = one checkpoint at a time (~1.3 GB for rn152)
        """
        import numpy as np
        checkpoint_dir = os.path.join(save_dir, "checkpoints")

        for layer_name, target_type in self.layers_to_hook:

            # ── Pass 1: count total samples and get feature shape ──────────
            total_samples = 0
            feature_shape = None

            for i in range(1, checkpoint_counter + 1):
                cp_path = os.path.join(
                    checkpoint_dir, f"{layer_name}_batch_{i}.pt"
                )
                if not os.path.exists(cp_path):
                    continue
                try:
                    cp = torch.load(cp_path, map_location='cpu')
                    cp = self._apply_pooling(cp, pool_type)
                    if feature_shape is None:
                        feature_shape = list(cp.shape[1:])
                    total_samples += cp.shape[0]
                    del cp
                except Exception as e:
                    print(f"  Warning: cannot read {cp_path}: {e}")

            if feature_shape is None or total_samples == 0:
                print(f"  Warning: no valid checkpoints for '{layer_name}'")
                continue

            full_shape = tuple([total_samples] + feature_shape)
            print(f"\nActivation Viz: Combining {checkpoint_counter} checkpoints "
                f"for '{layer_name}'")
            print(f"  Total shape : {full_shape}")

            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            filename  = (f"activations_{safe_layer_name}"
                        f"_{target_type}_{pool_type}.safetensors")
            save_path = os.path.join(save_dir, filename)
            tmp_npy   = save_path + ".tmp.npy"

            try:
                # ── Pass 2: write each checkpoint into disk memmap ─────────
                memmap_arr = np.lib.format.open_memmap(
                    tmp_npy, mode='w+',
                    dtype=np.float32,
                    shape=full_shape
                )

                offset = 0
                for i in range(1, checkpoint_counter + 1):
                    cp_path = os.path.join(
                        checkpoint_dir, f"{layer_name}_batch_{i}.pt"
                    )
                    if not os.path.exists(cp_path):
                        continue
                    try:
                        cp = torch.load(cp_path, map_location='cpu')
                        cp = self._apply_pooling(cp, pool_type)
                        n  = cp.shape[0]
                        memmap_arr[offset: offset + n] = cp.numpy()
                        memmap_arr.flush()
                        offset += n
                        del cp
                        print(f"  Written {offset}/{total_samples} "
                            f"(checkpoint {i}/{checkpoint_counter})")
                    except Exception as e:
                        print(f"  Warning: error in checkpoint {i}: {e}")

                # ── Final: stream from memmap → safetensors ────────────────
                # Activation Viz FIX: do NOT call save_file() here.
                # save_file() calls .tobytes() = 20 GB RAM copy → MemoryError.
                # Use streaming writer instead.
                self._save_safetensors_streaming(
                    layer_name, memmap_arr, save_path, chunk_size=1000
                )

                # save metadata
                metadata = {
                    "layer_name":      layer_name,
                    "shape":           str(list(full_shape)),
                    "pool_type":       pool_type,
                    "dtype":           "float32",
                    "num_checkpoints": str(checkpoint_counter)
                }
                meta_path = save_path.replace(".safetensors", "_metadata.txt")
                with open(meta_path, "w") as f:
                    for k, v in metadata.items():
                        f.write(f"{k}: {v}\n")

                # cleanup
                del memmap_arr
                if os.path.exists(tmp_npy):
                    os.remove(tmp_npy)
                    print(f"  Temp file removed")

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
                print(f"  Removed {removed} checkpoint .pt files")

            except Exception as e:
                print(f"  Error during combine for '{layer_name}': {e}")
                if os.path.exists(tmp_npy):
                    try:
                        os.remove(tmp_npy)
                    except OSError:
                        pass
                raise

        try:
            if os.path.exists(checkpoint_dir) and not os.listdir(checkpoint_dir):
                os.rmdir(checkpoint_dir)
        except OSError:
            pass

    def cleanup(self):
        """Remove all registered forward hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

    def __del__(self):
        self.cleanup()

# =============================================================================
# Activation Viz: get_args — same as original minus config_file args,
# plus CoE-style model_name / data_path / dataset args.
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Activation Viz - Step 1: Extract activations"
    )

    # Activation Viz CHANGE: direct model/data args instead of --config_file
    parser.add_argument('--model_name', type=str, default='rn152',
                        choices=['rn152', 'rn50', 'vgg16', 'vit',
                                 'vit_base_patch16_224'],
                        help="Model to use")
    parser.add_argument('--data_path', type=str, default='./dataset',
                        help="Root data directory (imagenet subfolder expected)")
    parser.add_argument('--dataset', type=str, default='imagenet-val',
                        help="Dataset name-split, e.g. imagenet-val")
    parser.add_argument('--resume', type=str, default=None,
                        help="Checkpoint path (None = pretrained)")

    # Activation Viz: layer hooking — same as original
    parser.add_argument(
        '--layers_to_hook', type=str, nargs='+',
        default=None,
        help=(
            "Pairs of layer_name and target_type. "
            "E.g.: --layers_to_hook blocks.11 output layer4.2.conv3 output. "
            "If not provided, uses LAYER_DEFAULTS for the chosen model."
        )
    )
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--max_samples',   type=int,   default=None)
    parser.add_argument('--save_dir',      type=str,
                        default='results/activation_viz/activations')
    parser.add_argument('--pool_type',     type=str,   default='raw',
                        choices=['gap', 'gmp', 'raw'])
    parser.add_argument('--save_intermediate', action='store_true',
                        help="Save checkpoints every N batches")
    parser.add_argument('--checkpoint_interval', type=int, default=500)

    args = parser.parse_args()

    # Activation Viz: parse layers_to_hook pairs
    if args.layers_to_hook is not None:
        flat = args.layers_to_hook
        if len(flat) % 2 != 0:
            raise ValueError(
                "--layers_to_hook must be name/type pairs. "
                "E.g.: --layers_to_hook blocks.11 output"
            )
        args.layers_to_hook = list(zip(flat[::2], flat[1::2]))
    else:
        # Activation Viz: use default layer for model
        default_layer = LAYER_DEFAULTS.get(args.model_name, 'layer4.2.conv3')
        args.layers_to_hook = [(default_layer, 'output')]
        print(f"Activation Viz: Using default layer '{default_layer}' for "
              f"model '{args.model_name}'")

    return args


# =============================================================================
# Activation Viz: main() — replaces config-based loading with CoE loaders
# =============================================================================

def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Activation Viz [extract]: device = {device}")
    print(f"  model      = {args.model_name}")
    print(f"  layers     = {args.layers_to_hook}")
    print(f"  pool_type  = {args.pool_type}")
    print(f"  save_dir   = {args.save_dir}")

    # Activation Viz CHANGE: load model via CoE's build_models()
    coe_args = _make_args(args.model_name, args.data_path,
                          args.dataset, args.resume)
    model, _, _, _, _ = build_models(coe_args)
    if model is None:
        raise ValueError(
            f"Activation Viz: build_models() returned None for "
            f"model_name='{args.model_name}'"
        )
    model = model.to(device).eval()
    print(f"Activation Viz: Model '{args.model_name}' loaded")

    # Activation Viz CHANGE: load dataset via CoE's MakeDataset
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    dataset = MakeDataset(
        coe_args,
        transform=transform,
        dataset_split=coe_args.dataset_split
    )
    print(f"Activation Viz: Dataset loaded with {len(dataset)} samples")

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        print(f"Activation Viz: Limited to {args.max_samples} samples")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    # Activation Viz: save directory per model
    save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Activation Viz: run extraction — ActivationExtractor unchanged from original
    extractor = ActivationExtractor(model, args.layers_to_hook)
    try:
        extractor.extract(
            data_loader,
            save_dir=save_dir,
            save_intermediate=args.save_intermediate,
            pool_type=args.pool_type,
            checkpoint_interval=args.checkpoint_interval
        )

        # save overall metadata
        metadata = {
            "model_name":    args.model_name,
            "dataset":       args.dataset,
            "layers_to_hook": str(args.layers_to_hook),
            "num_samples":   len(dataset),
            "pool_type":     args.pool_type,
            "batch_size":    args.batch_size,
        }
        meta_path = os.path.join(save_dir, "extraction_metadata.txt")
        with open(meta_path, "w") as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

        print(f"\nActivation Viz [extract]: Complete. Results -> {save_dir}")
        for fname in os.listdir(save_dir):
            if fname.endswith('.safetensors'):
                fpath = os.path.join(save_dir, fname)
                fsize = os.path.getsize(fpath) / (1024 ** 3)
                print(f"  {fname}  ({fsize:.2f} GB)")

    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
