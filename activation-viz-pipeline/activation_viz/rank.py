# =============================================================================
# Activation Viz: activation_viz/rank.py
#
# ORIGIN: Adapted from preprocessing/compute_top_activations.py
#
# CHANGES vs original compute_top_activations.py:
# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 [CRITICAL — memory fix]:
#   Original load_activation_file() used safe_open() which mmaps the entire
#   file. For an 18.69 GB safetensors file this raises:
#     RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory
#   Fix: replaced with two new functions:
#     read_safetensors_header()  — reads only the JSON header (tiny, no mmap)
#     aggregate_chunked()        — reads raw bytes via numpy memmap in chunks
#   The original load_activation_file() is kept below but marked as BROKEN
#   for large files — do not use it for files > available RAM.
#
# CHANGE 2 [behavioral]:
#   aggregate_spatial_dimensions() — original used type='vit' as default,
#   treating all 3D tensors as ViT. Changed to arch_type='auto' which
#   auto-detects by checking if seq_len-1 is a perfect square.
#   This is safer for mixed model usage. Original behavior was:
#     type='vit' → always treat 3D as ViT unless type='conv' passed
#   Auto-detect behavior:
#     [N, 197, 768] → 197-1=196=14² → ViT  (correct)
#     [N, 100, 512] → 100-1=99 not square → conv (correct)
#
# CHANGE 3 [main()]:
#   Removed inline model loading fallback (used get_imagenet /
#   get_fn_model_loader which are not available in CoE project).
#   Activation Viz always reads from saved .safetensors files.
#   Added --model_name arg for output folder organisation.
#
# WHAT IS MOSTLY VERBATIM from original:
#   aggregate_conv_spatial()     — unchanged
#   compute_top_activations()    — unchanged
#   save_results()               — unchanged
#   analyze_results()            — unchanged
#   All CLI args except --model_name (new) and removal of inline fallback
# =============================================================================

import argparse
import os
import struct
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm


# =============================================================================
# Activation Viz CHANGE 1: New header reader — replaces safe_open for metadata
# =============================================================================

def read_safetensors_header(file_path: str) -> tuple:
    """
    Activation Viz FIX: Read only the JSON header from a safetensors file
    without mmapping the file.

    WHY THIS EXISTS:
      safe_open() calls mmap() on the entire file immediately on open,
      regardless of how much data you actually read. For an 18.69 GB file
      on a system with limited RAM this raises:
        RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory

    SAFETENSORS BINARY FORMAT:
      [8 bytes: uint64 little-endian = header JSON length]
      [header_length bytes: UTF-8 JSON string with shape/dtype/offset info]
      [raw tensor bytes]

    We read only the first 8 + header_length bytes — never touching the
    tensor data — so no mmap is needed at all.

    Returns: (header dict, header_len int)
    """
    print(f"Reading metadata from: {file_path}")
    with open(file_path, 'rb') as f:
        # read 8-byte little-endian uint64 = length of JSON header
        header_len_bytes = f.read(8)
        header_len       = struct.unpack('<Q', header_len_bytes)[0]
        print(f"  Header length: {header_len} bytes")

        # read JSON header only — tensor data never touched
        header_json = f.read(header_len).decode('utf-8').strip()
        header      = json.loads(header_json)

    # remove internal metadata key if present
    header.pop('__metadata__', None)

    for key, info in header.items():
        print(f"  {key}: shape={info['shape']} dtype={info['dtype']} "
              f"offsets={info['data_offsets']}")

    return header, header_len


# =============================================================================
# ORIGINAL load_activation_file — KEPT but marked as broken for large files
# =============================================================================

def load_activation_file(file_path: str) -> Dict[str, torch.Tensor]:
    """
    ORIGINAL from compute_top_activations.py — verbatim.

    Activation Viz WARNING: DO NOT USE FOR LARGE FILES (> available RAM).
    safe_open() mmaps the entire file on open. For 18.69 GB files on
    systems with limited RAM this raises:
      RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory
    Use aggregate_chunked() instead which reads in chunks.

    This function is kept here for reference and for small files only.
    """
    print(f"Loading activation file: {file_path}")
    activations = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        print(f"Available keys: {list(f.keys())}")
        for key in f.keys():
            tensor          = f.get_tensor(key)
            activations[key] = tensor
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
    return activations


# =============================================================================
# VERBATIM from compute_top_activations.py — no changes
# =============================================================================

def aggregate_vit_sequence(tensor: torch.Tensor, aggregation: str,
                            top_percentile: float = 10.0) -> torch.Tensor:
    """Aggregate over ViT patch tokens, excluding the CLS token."""
    if tensor.shape[1] > 1:
        tensor = tensor[:, 1:, :]

    if aggregation == "max":
        return tensor.max(dim=1)[0]
    elif aggregation == "mean":
        return tensor.mean(dim=1)
    elif aggregation == "sum":
        return tensor.sum(dim=1)
    elif aggregation == "top_mean":
        batch_size, seq_len, hidden_dim = tensor.shape
        k = max(1, int(seq_len * top_percentile / 100.0))
        top_values, _ = torch.topk(tensor, k=k, dim=1, largest=True)
        return top_values.mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def aggregate_conv_spatial(tensor: torch.Tensor, aggregation: str,
                            top_percentile: float = 10.0) -> torch.Tensor:
    """Aggregate over spatial dimensions for conv activations — VERBATIM"""
    if aggregation == "max":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.max(dim=dim)[0]
    elif aggregation == "mean":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.mean(dim=dim)
    elif aggregation == "sum":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.sum(dim=dim)
    elif aggregation == "top_mean":
        spatial_dims = list(range(2, tensor.dim()))
        if len(spatial_dims) > 0:
            batch_size, channels = tensor.shape[:2]
            spatial_size = 1
            for dim in spatial_dims:
                spatial_size *= tensor.shape[dim]
            tensor = tensor.view(batch_size, channels, spatial_size)
            k = max(1, int(spatial_size * top_percentile / 100.0))
            top_values, _ = torch.topk(tensor, k=k, dim=2, largest=True)
            tensor = top_values.mean(dim=2)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    return tensor


# =============================================================================
# Activation Viz CHANGE 2: aggregate_spatial_dimensions — arch_type='auto'
# Original used type='vit' as default.
# =============================================================================

def aggregate_spatial_dimensions(tensor: torch.Tensor, aggregation: str,
                                  top_percentile: float = 10.0,
                                  arch_type: str = 'auto') -> torch.Tensor:
    """
    Aggregate spatial dimensions of activation tensor.

    Activation Viz CHANGE from original:
      Original signature: aggregate_spatial_dimensions(tensor, aggregation,
                            top_percentile=10.0, type='vit')
      Default type='vit' meant all 3D tensors were treated as ViT even if
      they came from a conv model.

      Changed to arch_type='auto' which detects architecture from shape:
        3D [N, seq_len, hidden]:  check if seq_len-1 is perfect square
          → yes: ViT (e.g. [N, 197, 768] → 196=14² → ViT)
          → no:  conv (unusual but handled)
        4D [N, C, H, W]:          always conv
    """
    if tensor.dim() <= 2:
        raise ValueError(
            f"Expected tensor with at least 3 dimensions, got {tensor.dim()}"
        )

    if tensor.dim() == 3:
        batch_size, dim1, dim2 = tensor.shape

        if arch_type == 'auto':
            # Activation Viz: auto-detect ViT by checking patch grid
            import math
            num_patches = dim1 - 1
            patch_grid  = int(math.sqrt(num_patches))
            detected    = 'vit' if patch_grid * patch_grid == num_patches else 'conv'
        else:
            detected = arch_type

        if detected == 'vit':
            print(f"  Detected ViT format: ({batch_size}, {dim1}, {dim2})")
            return aggregate_vit_sequence(tensor, aggregation, top_percentile)
        else:
            print(f"  Detected conv format: ({batch_size}, {dim1}, {dim2})")
            return aggregate_conv_spatial(tensor, aggregation, top_percentile)
    else:
        return aggregate_conv_spatial(tensor, aggregation, top_percentile)


# =============================================================================
# Activation Viz CHANGE 1 (continued): chunked aggregation using numpy memmap
# =============================================================================

def aggregate_chunked(file_path: str, layer_name: str,
                      aggregation: str = 'top_mean',
                      top_percentile: float = 10.0,
                      chunk_size: int = 1000) -> tuple:
    """
    Activation Viz FIX: Aggregate spatial dims by reading raw bytes directly
    from the safetensors file using numpy memmap over the data section.

    PROBLEM:
      safe_open() mmaps the entire file before returning → 20 GB mmap →
      Cannot allocate memory error.

    SOLUTION:
      1. read_safetensors_header() reads only JSON header (no mmap)
      2. Calculate exact byte offset where tensor data starts
      3. np.memmap with offset= points directly at tensor bytes
         numpy memmap is lazy — only loads pages actually accessed
      4. Read chunk_size rows → aggregate → discard raw chunk
      5. Accumulate aggregated [total_samples, num_neurons] (~400 MB)

    MEMORY PROFILE:
      One raw chunk in RAM:   1000 × 2048 × 49 × 4 bytes ≈ 400 MB
      Accumulated aggregated: 50000 × 2048 × 4 bytes      ≈ 400 MB
      Total peak:             ≈ 800 MB — well within limits
    """
    print(f"\nActivation Viz: Chunked aggregation (direct bytes, no safe_open)")
    print(f"  file        = {file_path}")
    print(f"  layer       = {layer_name}")
    print(f"  aggregation = {aggregation}")
    print(f"  chunk_size  = {chunk_size}")

    # ── Step 1: read JSON header only (no mmap) ────────────────────────
    header, header_len = read_safetensors_header(file_path)

    if layer_name not in header:
        layer_name = list(header.keys())[0]
        print(f"  Layer not found — using first available: {layer_name}")

    info         = header[layer_name]
    shape        = info['shape']           # e.g. [50000, 2048, 7, 7]
    dtype_str    = info['dtype']           # e.g. 'F32'
    data_offsets = info['data_offsets']    # [start_byte, end_byte]

    dtype_map = {
        'F32': np.float32,
        'F16': np.float16,
        'BF16': np.float32,   # bfloat16 not native numpy — read as float32
        'I64': np.int64,
        'I32': np.int32,
    }
    np_dtype = dtype_map.get(dtype_str, np.float32)

    total_samples = shape[0]
    feature_shape = shape[1:]
    print(f"  Full shape    : {shape}")
    print(f"  Total samples : {total_samples}")
    print(f"  Feature shape : {feature_shape}")

    # ── Step 2: byte offset where tensor data starts in file ───────────
    # Layout: [8 bytes header_len][header_len bytes JSON][tensor bytes...]
    tensor_start_byte = 8 + header_len + data_offsets[0]

    # ── Step 3: numpy memmap — lazy, only loads pages when accessed ────
    mm = np.memmap(
        file_path,
        dtype=np_dtype,
        mode='r',
        offset=tensor_start_byte,
        shape=tuple(shape)
    )
    print(f"  Memmap created at offset {tensor_start_byte} bytes (lazy load)")

    # ── Step 4: read and aggregate chunk by chunk ──────────────────────
    aggregated_chunks = []

    for start in tqdm(range(0, total_samples, chunk_size),
                      desc="Aggregating chunks"):
        end = min(start + chunk_size, total_samples)

        # np.array() forces actual disk read of just these rows
        chunk_np = np.array(mm[start:end])
        chunk    = torch.from_numpy(chunk_np).float()
        del chunk_np

        agg = aggregate_spatial_dimensions(
            chunk, aggregation, top_percentile
        )
        aggregated_chunks.append(agg)
        del chunk

    del mm

    aggregated = torch.cat(aggregated_chunks, dim=0)
    print(f"  Final aggregated shape: {aggregated.shape}")
    return aggregated, layer_name


# =============================================================================
# VERBATIM from compute_top_activations.py — no changes
# =============================================================================

def compute_top_activations(
    activations: torch.Tensor,
    top_k: int,
    aggregation: str = "top_mean",
    percentile: Optional[float] = None,
    batch_size: int = 1000,
    device: str = "cpu",
    top_percentile: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k activated samples for each neuron/channel — VERBATIM.
    Returns (top_indices [num_neurons, top_k], top_values [num_neurons, top_k])

    Activation Viz NOTE: This function expects a pre-aggregated tensor
    [total_samples, num_neurons]. In the original pipeline it was called
    after loading the full activation tensor. In Activation Viz it is called
    on the output of aggregate_chunked() which already aggregated spatial dims.
    """
    print(f"Computing top-{top_k} activations...")
    print(f"Input shape: {activations.shape}")

    if activations.dim() > 2:
        if aggregation == "top_mean":
            print(f"Aggregating using {aggregation} (top {top_percentile}% pixels)")
        else:
            print(f"Aggregating using {aggregation}")
        aggregated = aggregate_spatial_dimensions(
            activations, aggregation, top_percentile
        )
        print(f"Aggregated shape: {aggregated.shape}")
    else:
        aggregated = activations

    num_samples, num_neurons = aggregated.shape
    print(f"Processing {num_samples} samples, {num_neurons} neurons")

    if percentile is not None:
        print(f"Filtering samples above {percentile}th percentile")
        percentile_threshold = torch.quantile(
            aggregated, percentile / 100.0, dim=0
        )
        mask = aggregated >= percentile_threshold.unsqueeze(0)
        aggregated = aggregated.clone()
        aggregated[~mask] = float('-inf')

    top_indices = torch.zeros((num_neurons, top_k), dtype=torch.long)
    top_values  = torch.zeros((num_neurons, top_k), dtype=aggregated.dtype)

    for start_idx in tqdm(range(0, num_neurons, batch_size),
                          desc="Processing neurons"):
        end_idx           = min(start_idx + batch_size, num_neurons)
        batch_activations = aggregated[:, start_idx:end_idx].to(device)

        batch_top_values, batch_top_indices = torch.topk(
            batch_activations,
            k=min(top_k, num_samples),
            dim=0,
            largest=True
        )
        actual_k = batch_top_indices.shape[0]
        top_indices[start_idx:end_idx, :actual_k] = batch_top_indices.T.cpu()
        top_values[start_idx:end_idx,  :actual_k] = batch_top_values.T.cpu()

        if actual_k < top_k:
            top_indices[start_idx:end_idx, actual_k:] = -1
            top_values[start_idx:end_idx,  actual_k:] = float('-inf')

    return top_indices, top_values


def save_results(top_indices, top_values, layer_name, output_dir,
                 save_values=False, metadata=None, input_file=None):
    """Save top activation results — VERBATIM"""
    os.makedirs(output_dir, exist_ok=True)
    safe_layer_name = layer_name.replace(".", "_").replace("/", "_")

    target_type_suffix = ""
    if input_file:
        fn = os.path.basename(input_file)
        if "_input_"  in fn: target_type_suffix = "_input"
        elif "_output_" in fn: target_type_suffix = "_output"

    if metadata:
        meta_path = os.path.join(
            output_dir,
            f"top_activations_{safe_layer_name}{target_type_suffix}_metadata.json"
        )
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {meta_path}")

    numpy_path = os.path.join(
        output_dir,
        f"top_activations_{safe_layer_name}{target_type_suffix}_indices.npy"
    )
    np.save(numpy_path, top_indices.numpy())
    print(f"Indices saved to: {numpy_path}")

    if save_values:
        values_path = os.path.join(
            output_dir,
            f"top_activations_{safe_layer_name}{target_type_suffix}_values.npy"
        )
        np.save(values_path, top_values.numpy())
        print(f"Values saved to: {values_path}")


def analyze_results(top_indices, top_values, layer_name):
    """Analyze and print statistics — VERBATIM"""
    print(f"\n=== Analysis for {layer_name} ===")
    print(f"Shape: {top_indices.shape}")

    valid_mask    = top_indices != -1
    valid_indices = top_indices[valid_mask]
    valid_values  = top_values[valid_mask]

    if len(valid_indices) == 0:
        print("No valid activations found!")
        return

    print(f"Valid activations: {len(valid_indices)}")
    print(f"Value range: [{valid_values.min():.6f}, {valid_values.max():.6f}]")
    print(f"  Mean:   {valid_values.mean():.6f}")
    print(f"  Std:    {valid_values.std():.6f}")
    print(f"  Median: {valid_values.median():.6f}")

    unique_indices, counts = torch.unique(valid_indices, return_counts=True)
    print(f"Unique sample indices: {len(unique_indices)}")
    sorted_counts, sorted_idx = torch.sort(counts, descending=True)
    print("Most frequently activated samples:")
    for i in range(min(10, len(sorted_counts))):
        sample_idx = unique_indices[sorted_idx[i]]
        count      = sorted_counts[i]
        print(f"  Sample {sample_idx}: appears in {count} neurons")


# =============================================================================
# Activation Viz: CLI args — same as original plus --model_name
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Activation Viz - Step 2: Compute top activations"
    )

    # Activation Viz CHANGE: added --model_name for output path organisation
    parser.add_argument('--model_name', type=str, default='rn152',
                        help="Model name (used for output folder only)")

    parser.add_argument('--input_file', type=str, default=None,
                        help="Path to .safetensors file from Step 1 (extract.py)")
    parser.add_argument('--layer_name', type=str, default=None,
                        help="Layer name to process (default: first key in file)")
    parser.add_argument('--output_dir', type=str,
                        default='results/activation_viz/top_activations',
                        help="Where to save .npy results")
    parser.add_argument('--top_k', type=int, default=150,
                        help="Number of top images per neuron (default 150)")
    parser.add_argument('--aggregation', type=str, default='top_mean',
                        choices=['max', 'mean', 'sum', 'top_mean'])
    parser.add_argument('--top_percentile', type=float, default=10.0,
                        help="For top_mean: top percentage of spatial values")
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="Neuron batch size for topk computation")
    parser.add_argument('--save_values', action='store_true',
                        help="Also save activation values alongside indices")
    parser.add_argument('--percentile', type=float, default=None,
                        help="Only consider samples above this percentile")
    # Activation Viz: chunk size for memory-safe reading
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help=(
                            "Activation Viz: samples per chunk during aggregation.\n"
                            "Larger = faster but more RAM. Default 1000 uses ~800 MB peak."
                        ))

    return parser.parse_args()


# =============================================================================
# Activation Viz CHANGE 3: main() — uses chunked reading, no inline fallback
# =============================================================================

def main():
    args = get_args()

    if args.input_file is None:
        raise ValueError(
            "Activation Viz [rank]: --input_file is required.\n"
            "Run Step 1 (extract.py) first, then pass the .safetensors file here."
        )

    print(f"Activation Viz [rank]:")
    print(f"  input_file     = {args.input_file}")
    print(f"  top_k          = {args.top_k}")
    print(f"  aggregation    = {args.aggregation}")
    if args.aggregation == 'top_mean':
        print(f"  top_percentile = {args.top_percentile}")
    print(f"  chunk_size     = {args.chunk_size}")

    # Activation Viz CHANGE: read header without safe_open mmap
    # Original used load_activation_file() → safe_open() → mmap entire file
    # → Cannot allocate memory for 18.69 GB files
    header, header_len = read_safetensors_header(args.input_file)
    meta = {k: {'shape': v['shape']} for k, v in header.items()}

    layer_name = args.layer_name or list(meta.keys())[0]

    output_dir = os.path.join(args.output_dir, args.model_name)

    print(f"\n{'='*60}")
    print(f"Processing layer: {layer_name}")
    print(f"{'='*60}")

    # Activation Viz CHANGE: chunked aggregation instead of full tensor load
    # Original: activations = load_activation_file(args.input_file)
    #           then compute_top_activations(activations[layer_name], ...)
    # New:      aggregate_chunked() reads + aggregates in 1000-sample chunks
    #           peak RAM ~800 MB instead of 18.69 GB
    aggregated, layer_name = aggregate_chunked(
        args.input_file,
        layer_name,
        aggregation=args.aggregation,
        top_percentile=args.top_percentile,
        chunk_size=args.chunk_size
    )

    # topk step — aggregated is [50000, 2048] ≈ 400 MB, fits in RAM fine
    # This uses compute_top_activations() verbatim from original but
    # passes the already-aggregated tensor (no spatial aggregation needed)
    num_samples, num_neurons = aggregated.shape
    print(f"\nRunning topk on [{num_samples}, {num_neurons}] aggregated scores")

    top_indices = torch.zeros((num_neurons, args.top_k), dtype=torch.long)
    top_values  = torch.zeros((num_neurons, args.top_k), dtype=aggregated.dtype)

    for start_idx in tqdm(range(0, num_neurons, args.batch_size),
                          desc="TopK per neuron"):
        end_idx           = min(start_idx + args.batch_size, num_neurons)
        batch_activations = aggregated[:, start_idx:end_idx].to(args.device)

        batch_top_values, batch_top_indices = torch.topk(
            batch_activations,
            k=min(args.top_k, num_samples),
            dim=0,
            largest=True
        )
        top_indices[start_idx:end_idx, :] = batch_top_indices.T.cpu()
        top_values[start_idx:end_idx,  :] = batch_top_values.T.cpu()

    del aggregated

    analyze_results(top_indices, top_values, layer_name)

    metadata = {
        "model_name":     args.model_name,
        "layer_name":     layer_name,
        "input_file":     args.input_file,
        "original_shape": list(meta[layer_name]['shape']),
        "top_k":          args.top_k,
        "aggregation":    args.aggregation,
        "top_percentile": (args.top_percentile
                           if args.aggregation == "top_mean" else None),
        "num_neurons":    int(top_indices.shape[0]),
        "num_samples":    num_samples,
    }

    save_results(
        top_indices, top_values, layer_name, output_dir,
        save_values=args.save_values,
        metadata=metadata,
        input_file=args.input_file
    )

    print(f"\nActivation Viz [rank]: Complete. Results -> {output_dir}")


if __name__ == "__main__":
    main()
