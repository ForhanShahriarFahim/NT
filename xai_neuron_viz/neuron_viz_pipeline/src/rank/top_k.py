# =============================================================================
# FILE: neuron_viz_pipeline/src/rank/top_k.py
#
# Purpose:
#   Stage 2 — find the top-k most-activating samples per neuron/channel.
#
# Workflow:
#   1. Open the Stage 1 .safetensors file as a lazy numpy memmap
#      (via src.utils.io_safetensors.open_safetensors_memmap — no mmap OOM)
#   2. Stream through the memmap in chunks, applying spatial/sequence
#      aggregation per chunk → one score per (sample, neuron)
#      The aggregated tensor fits comfortably in RAM:
#        rn152: 50 000 × 2048 × 4 bytes ≈ 400 MB
#        vit:   50 000 × 768  × 4 bytes ≈ 150 MB
#   3. Run torch.topk along the sample dimension → top-k per neuron
#   4. Save (indices, values, metadata) to disk
#
# Source attribution:
#   - aggregate_vit_sequence        ← VERBATIM from preprocessing/
#                                     compute_top_activations.py (reference file)
#   - aggregate_conv_spatial        ← VERBATIM (same source)
#   - compute_top_activations       ← VERBATIM (same source); already handles
#                                     pre-aggregated 2D input by skipping its
#                                     own aggregation step
#   - save_results                  ← ADAPTED: takes explicit paths from
#                                     src/utils/paths.py instead of constructing
#                                     its own. Filename patterns remain
#                                     byte-identical to the reference.
#   - analyze_results               ← VERBATIM (optional diagnostic printout)
#   - aggregate_chunked             ← NEW wrapper that chunks through the memmap.
#                                     Based on your RAM-safe pattern from
#                                     pipeline_a/rank.py.
#
# Why we don't just call the reference compute_top_activations directly:
#   The reference loads the ENTIRE tensor via safetensors.safe_open.get_tensor(),
#   which mmaps the whole 20 GB file at once → triggers "unable to mmap" on
#   RAM-limited machines. Our aggregate_chunked reads chunks lazily via
#   numpy.memmap (the helper from src/utils/io_safetensors.py).
# =============================================================================

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-chunk aggregation helpers — VERBATIM from compute_top_activations.py
# ---------------------------------------------------------------------------

def aggregate_vit_sequence(
    tensor: torch.Tensor,
    aggregation: str,
    top_percentile: float = 10.0,
) -> torch.Tensor:
    """
    Aggregate the sequence dimension of ViT activations.

    Args:
        tensor:          shape (batch, seq_len, hidden_dim)
        aggregation:     'max' | 'mean' | 'sum' | 'top_mean'
        top_percentile:  only used when aggregation='top_mean'

    Returns:
        (batch, hidden_dim)

    VERBATIM from preprocessing/compute_top_activations.py:aggregate_vit_sequence
    """
    if aggregation == "max":
        return tensor.max(dim=1)[0]
    elif aggregation == "mean":
        return tensor.mean(dim=1)
    elif aggregation == "sum":
        return tensor.sum(dim=1)
    elif aggregation == "top_mean":
        _batch, seq_len, _hidden = tensor.shape
        # For each sample and each hidden dim, take mean of top-percentile
        # sequence positions. Example: seq_len=197, top_percentile=10
        #   → k = max(1, int(197 * 10 / 100)) = 19 positions
        k = max(1, int(seq_len * top_percentile / 100.0))
        top_values, _ = torch.topk(tensor, k=k, dim=1, largest=True)
        return top_values.mean(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def aggregate_conv_spatial(
    tensor: torch.Tensor,
    aggregation: str,
    top_percentile: float = 10.0,
) -> torch.Tensor:
    """
    Aggregate the spatial dimension(s) of conv activations.

    Args:
        tensor:          shape (batch, channels, H, W[, ...])
        aggregation:     'max' | 'mean' | 'sum' | 'top_mean'
        top_percentile:  only used when aggregation='top_mean'

    Returns:
        (batch, channels)

    VERBATIM from preprocessing/compute_top_activations.py:aggregate_conv_spatial
    """
    if aggregation == "max":
        # Reduce max over each trailing dim, inside-out
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.max(dim=dim)[0]
    elif aggregation == "mean":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.mean(dim=dim)
    elif aggregation == "sum":
        for dim in range(tensor.dim() - 1, 1, -1):
            tensor = tensor.sum(dim=dim)
    elif aggregation == "top_mean":
        # Flatten spatial dims → compute top-percentile over flat spatial
        spatial_dims = list(range(2, tensor.dim()))
        if len(spatial_dims) > 0:
            batch_size, channels = tensor.shape[:2]
            spatial_size = 1
            for dim in spatial_dims:
                spatial_size *= tensor.shape[dim]
            tensor = tensor.view(batch_size, channels, spatial_size)

            # Example for rn152 layer4.2.conv3: spatial = 7*7 = 49
            #   top_percentile=10 → k = max(1, int(49 * 10 / 100)) = 4 positions
            k = max(1, int(spatial_size * top_percentile / 100.0))
            top_values, _ = torch.topk(tensor, k=k, dim=2, largest=True)
            tensor = top_values.mean(dim=2)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return tensor


# ---------------------------------------------------------------------------
# RAM-safe chunked aggregation — NEW (shaped like pipeline_a/rank.py)
# ---------------------------------------------------------------------------

def aggregate_chunked(
    memmap: np.memmap,
    layer_type: str,
    aggregation: str = "top_mean",
    top_percentile: float = 10.0,
    chunk_size: int = 1000,
) -> torch.Tensor:
    """
    Read a large activation tensor as a numpy memmap and apply per-chunk
    spatial/sequence aggregation, accumulating into a single (N, F) tensor.

    Why this exists:
        The reference compute_top_activations.py loads the whole tensor via
        safetensors.safe_open(...).get_tensor(key), which calls mmap() on the
        entire file and triggers "unable to mmap 20 GB" on RAM-limited systems.
        Here we never materialize more than `chunk_size` samples at a time.

    RAM usage:
        Peak = chunk_size × feature_size × 4 bytes
             = 1000 × 2048 × 49 × 4 ≈ 400 MB (rn152 layer4.2.conv3)
             = 1000 × 197  × 768 × 4 ≈ 600 MB (vit blocks.11.mlp.fc2)

    Args:
        memmap:         numpy.memmap from open_safetensors_memmap()
                        Shape depends on the original tensor:
                          conv  : (N, C, H, W[, ...])
                          linear: (N, seq_len, hidden_dim)
        layer_type:     'conv' or 'linear' (from cfg.model.layer_type)
        aggregation:    'max' | 'mean' | 'sum' | 'top_mean'
        top_percentile: only used for 'top_mean'
        chunk_size:     samples per chunk (from cfg.rank.chunk_size)

    Returns:
        torch.Tensor of shape (num_samples, num_neurons)
    """
    n_samples = memmap.shape[0]
    if n_samples == 0:
        raise ValueError(f"memmap has 0 samples — empty activations file?")

    # Validate layer_type and choose the per-chunk aggregator once
    if layer_type == "conv":
        agg_fn = aggregate_conv_spatial
    elif layer_type == "linear":
        agg_fn = aggregate_vit_sequence
    else:
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. Expected 'conv' or 'linear'."
        )

    # Peek at the first chunk to discover the output feature size
    # (we need to know the output shape BEFORE allocating the result tensor)
    first_chunk_np = np.asarray(memmap[0:min(chunk_size, n_samples)]).copy()
    first_chunk = torch.from_numpy(first_chunk_np).float()
    first_agg = agg_fn(first_chunk, aggregation, top_percentile)
    # first_agg shape: (chunk_size, num_neurons)
    num_neurons = first_agg.shape[1]

    print(f"  [aggregate_chunked] input  shape = {tuple(memmap.shape)}")
    print(f"  [aggregate_chunked] layer_type   = {layer_type}  aggregation = {aggregation}")
    print(f"  [aggregate_chunked] output shape = ({n_samples}, {num_neurons})")
    print(f"  [aggregate_chunked] chunk_size   = {chunk_size}  "
          f"(~{chunk_size * num_neurons * 4 / 1024**2:.0f} MB/chunk in output)")

    # Pre-allocate the output tensor (small — fits in RAM)
    out = torch.empty((n_samples, num_neurons), dtype=torch.float32)

    # Store the first chunk result and continue from there
    end_first = first_chunk.shape[0]
    out[0:end_first] = first_agg.float()
    del first_chunk, first_chunk_np, first_agg

    # Process remaining chunks
    for start in tqdm(
        range(end_first, n_samples, chunk_size),
        desc="Aggregating activations",
    ):
        end = min(start + chunk_size, n_samples)
        # .copy() forces the memmap pages for this chunk to be read into
        # RAM so the memmap pages can be released when this chunk goes out
        # of scope (important for keeping peak memory bounded).
        chunk_np = np.asarray(memmap[start:end]).copy()
        chunk = torch.from_numpy(chunk_np).float()
        agg = agg_fn(chunk, aggregation, top_percentile)
        out[start:end] = agg.float()
        del chunk, chunk_np, agg

    return out


# ---------------------------------------------------------------------------
# Top-k computation — VERBATIM from compute_top_activations.py
# ---------------------------------------------------------------------------

def compute_top_activations(
    activations: torch.Tensor,
    top_k: int,
    aggregation: str = "max",
    percentile: Optional[float] = None,
    batch_size: int = 1000,
    device: str = "cpu",
    top_percentile: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k activated samples for each neuron/channel.

    In our pipeline we ALWAYS pass a pre-aggregated 2D tensor (from
    aggregate_chunked), so the internal `aggregate_spatial_dimensions` branch
    is skipped — the activations.dim() == 2 check falls through directly
    to the topk logic. But we keep the signature + logic byte-identical to
    the reference so future callers can pass 3D/4D tensors too.

    Args:
        activations:    (num_samples, num_neurons)  in our usage
                         — reference also accepts (num_samples, num_neurons, ...)
        top_k:          how many samples to keep per neuron (e.g. 150)
        aggregation:    kept for signature compatibility; ignored when dim()==2
        percentile:     optional: only consider samples above Nth percentile
        batch_size:     neurons processed per GPU batch (not sample batch)
        device:         'cpu' or 'cuda' for the topk computation
        top_percentile: kept for signature compatibility

    Returns:
        top_indices:  (num_neurons, top_k)  — sample indices, -1 where padded
        top_values:   (num_neurons, top_k)  — activation values, -inf padded

    VERBATIM from preprocessing/compute_top_activations.py:compute_top_activations
    """
    print(f"Computing top-{top_k} activations...")
    print(f"Input shape: {activations.shape}")

    # Aggregate spatial dimensions (skipped if already 2D — our case)
    if activations.dim() > 2:
        if aggregation == "top_mean":
            print(f"Aggregating spatial dimensions using {aggregation} "
                  f"(top {top_percentile}% pixels)")
        else:
            print(f"Aggregating spatial dimensions using {aggregation}")
        from .top_k import aggregate_conv_spatial as _agg  # self-reference fallback
        aggregated = _agg(activations, aggregation, top_percentile)
        print(f"Aggregated shape: {aggregated.shape}")
    else:
        aggregated = activations

    num_samples, num_neurons = aggregated.shape
    print(f"Processing {num_samples} samples, {num_neurons} neurons")

    # Optional percentile filtering
    if percentile is not None:
        print(f"Filtering samples above {percentile}th percentile")
        percentile_threshold = torch.quantile(aggregated, percentile / 100.0, dim=0)
        mask = aggregated >= percentile_threshold.unsqueeze(0)
        aggregated = aggregated.clone()
        aggregated[~mask] = float("-inf")

    # Initialize result tensors
    top_indices = torch.zeros((num_neurons, top_k), dtype=torch.long)
    top_values = torch.zeros((num_neurons, top_k), dtype=aggregated.dtype)

    # Process neurons in batches (memory-aware)
    for start_idx in tqdm(range(0, num_neurons, batch_size), desc="Processing neurons"):
        end_idx = min(start_idx + batch_size, num_neurons)
        batch_activations = aggregated[:, start_idx:end_idx].to(device)

        batch_top_values, batch_top_indices = torch.topk(
            batch_activations,
            k=min(top_k, num_samples),
            dim=0,
            largest=True,
        )

        actual_k = batch_top_indices.shape[0]
        top_indices[start_idx:end_idx, :actual_k] = batch_top_indices.T.cpu()
        top_values[start_idx:end_idx, :actual_k] = batch_top_values.T.cpu()

        # Pad with -1 if fewer samples than top_k
        if actual_k < top_k:
            top_indices[start_idx:end_idx, actual_k:] = -1
            top_values[start_idx:end_idx, actual_k:] = float("-inf")

    return top_indices, top_values


# ---------------------------------------------------------------------------
# Result saving — ADAPTED to take explicit paths from paths.py
# ---------------------------------------------------------------------------

def save_results(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    indices_path: str,
    values_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    metadata: Optional[Dict] = None,
    save_values: bool = True,
) -> None:
    """
    Save top-k results to disk.

    Adapted from preprocessing/compute_top_activations.py:save_results.
    The original version constructed filenames internally from layer_name;
    we take explicit paths from src/utils/paths.py (top_indices_file,
    top_values_file, top_k_metadata_file) to keep filename logic centralized
    and byte-identical to pipeline_a/rank.py output.

    Args:
        top_indices:    (num_neurons, top_k) — from compute_top_activations
        top_values:     (num_neurons, top_k)
        indices_path:   full path to the indices .npy file (required)
        values_path:    full path to the values .npy file (if save_values)
        metadata_path:  full path to the metadata .json file (if metadata)
        metadata:       dict to save alongside the .npy files
        save_values:    if True and values_path given, save top_values too
    """
    os.makedirs(os.path.dirname(indices_path), exist_ok=True)

    if metadata is not None and metadata_path is not None:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  metadata → {metadata_path}")

    np.save(indices_path, top_indices.numpy())
    print(f"  indices  → {indices_path}")

    if save_values and values_path is not None:
        np.save(values_path, top_values.numpy())
        print(f"  values   → {values_path}")


# ---------------------------------------------------------------------------
# Optional diagnostic — VERBATIM from compute_top_activations.py
# ---------------------------------------------------------------------------

def analyze_results(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Print statistics about the top-k results.

    VERBATIM from preprocessing/compute_top_activations.py:analyze_results.
    Purely diagnostic — safe to skip in production runs.
    """
    print(f"\n=== Analysis for {layer_name} ===")
    print(f"Shape: {top_indices.shape}")

    valid_mask = top_indices != -1
    valid_indices = top_indices[valid_mask]
    valid_values = top_values[valid_mask]

    if len(valid_indices) == 0:
        print("No valid activations found!")
        return

    print(f"Valid activations: {len(valid_indices)}")
    print(f"Value range: [{valid_values.min():.6f}, {valid_values.max():.6f}]")
    print(f"Value statistics:")
    print(f"  Mean:   {valid_values.mean():.6f}")
    print(f"  Std:    {valid_values.std():.6f}")
    print(f"  Median: {valid_values.median():.6f}")

    unique_indices, counts = torch.unique(valid_indices, return_counts=True)
    print(f"Unique sample indices: {len(unique_indices)}")
    print(f"Most frequently activated samples (top 10):")
    sorted_counts, sorted_idx = torch.sort(counts, descending=True)
    for i in range(min(10, len(sorted_counts))):
        sample_idx = unique_indices[sorted_idx[i]]
        count = sorted_counts[i]
        print(f"  Sample {sample_idx.item():>6d}: appears in {count.item()} neurons")