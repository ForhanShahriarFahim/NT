# =============================================================================
# FILE: neuron_viz_pipeline/src/utils/io_safetensors.py
#
# Purpose:
#   Three RAM-safe helpers for reading and writing large .safetensors files.
#   Used by Stage 1 (extract), Stage 2 (rank), and Stage 3 (xai_maps).
#
#   The official `safetensors` library uses mmap() which tries to map the
#   entire file into memory on open. For our 18-28 GB activation files on
#   systems with limited RAM (like your machine), that triggers:
#     RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory
#   These helpers avoid that by reading only the JSON header, then using
#   numpy.memmap with explicit byte offsets — numpy memmap is lazy (only
#   reads pages actually accessed).
#
# Three helpers:
#   1. read_safetensors_header(path)
#      Reads the JSON header only (tiny). Returns (header_dict, header_len).
#
#   2. save_safetensors_streaming(layer_name, memmap, path, chunk_size)
#      Writes a large numpy array to .safetensors format in chunks so peak
#      RAM stays at ~chunk_size samples, not the whole array.
#
#   3. open_safetensors_memmap(path, layer_name)
#      Returns (numpy.memmap, resolved_layer_name). Memmap is lazy —
#      memmap[i] only reads sample i from disk.
#
# Source:
#   Extracted verbatim from your pipeline_a code:
#     - read_safetensors_header   ← pipeline_a/rank.py
#     - save_safetensors_streaming ← pipeline_a/extract.py (_save_safetensors_streaming)
#     - open_safetensors_memmap   ← pipeline_a/crop.py (load_original_activations)
#   Centralizing them here avoids duplication across three stages.
#
# Safetensors binary format (for reference):
#   [8 bytes: little-endian uint64 = header JSON length]
#   [header_length bytes: UTF-8 JSON string {layer_name: {dtype, shape, offsets}}]
#   [raw tensor bytes...]
# =============================================================================

import json
import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


# Mapping from safetensors dtype strings to numpy dtypes.
# BF16 (bfloat16) is not natively supported by numpy → we read as float32.
# In practice activation tensors are almost always F32, so this is fine.
_DTYPE_MAP = {
    "F32":  np.float32,
    "F16":  np.float16,
    "BF16": np.float32,
    "I64":  np.int64,
    "I32":  np.int32,
    "I16":  np.int16,
    "U8":   np.uint8,
}

# Reverse mapping for writes (numpy dtype → safetensors string)
_DTYPE_REVERSE = {
    "float32":  "F32",
    "float16":  "F16",
    "bfloat16": "BF16",
    "int64":    "I64",
    "int32":    "I32",
    "int16":    "I16",
    "uint8":    "U8",
}


# ---------------------------------------------------------------------------
# Helper 1 — Read JSON header only, no mmap
# Source: pipeline_a/rank.py → read_safetensors_header
# ---------------------------------------------------------------------------

def read_safetensors_header(file_path: str) -> Tuple[dict, int]:
    """
    Read only the JSON header of a safetensors file without mmapping the data.

    Why this exists:
        safetensors.safe_open() calls mmap() on the entire file immediately
        on open. For an 18 GB file on a system with limited RAM, that raises:
            RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory
        We read only the first 8 + header_len bytes — the tensor data is
        never touched — so no mmap is needed at all.

    Args:
        file_path: path to a .safetensors file

    Returns:
        (header_dict, header_len)
        header_dict — parsed JSON; keys are tensor names (usually layer names),
                      values are dicts with 'dtype', 'shape', 'data_offsets'.
                      The internal '__metadata__' key is stripped.
        header_len  — length of the JSON header in bytes (needed to compute
                      the byte offset where tensor data starts).

    Example header_dict:
        {
            "layer4.2.conv3": {
                "dtype": "F32",
                "shape": [50000, 2048, 7, 7],
                "data_offsets": [0, 20070400000]
            }
        }
    """
    file_path = str(file_path)
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len).decode("utf-8").strip()
        header = json.loads(header_json)
    # Strip internal metadata key if present
    header.pop("__metadata__", None)
    return header, header_len


# ---------------------------------------------------------------------------
# Helper 2 — Stream-write a large array to safetensors
# Source: pipeline_a/extract.py → _save_safetensors_streaming
# ---------------------------------------------------------------------------

def save_safetensors_streaming(
    layer_name: str,
    arr: np.ndarray,
    save_path: str,
    chunk_size: int = 1000,
) -> None:
    """
    Write a large numpy array to .safetensors format in chunks.

    Why this exists:
        safetensors.torch.save_file() calls tensor.tobytes() internally,
        which copies the entire tensor into RAM at once. For a
        (50000, 2048, 7, 7) float32 tensor (~20 GB), that triggers MemoryError.

    How it avoids the OOM:
        Writes the safetensors binary format manually:
            [8 bytes uint64: header_len]
            [JSON header bytes]
            [raw bytes streamed in chunks]
        Only chunk_size samples (~1000) are in RAM at any given moment.
        Peak RAM = chunk_size × feature_size × dtype_bytes
                 = 1000 × 2048 × 49 × 4 ≈ 400 MB for rn152

    Args:
        layer_name : the key under which the tensor will be saved (e.g.
                     "layer4.2.conv3"). Matches what safe_open(...).get_tensor()
                     would return.
        arr        : numpy array or numpy memmap. If memmap, chunks are read
                     lazily from disk — peak RAM stays at one chunk.
        save_path  : destination .safetensors path
        chunk_size : samples per write chunk (default 1000). Don't raise
                     above 2000 unless you know you have RAM headroom.

    Raises:
        RuntimeError if the final file size doesn't match expected
                     (indicates a partial write).
    """
    dtype_str = _DTYPE_REVERSE.get(str(arr.dtype))
    if dtype_str is None:
        raise ValueError(
            f"Unsupported dtype {arr.dtype}. Supported: {list(_DTYPE_REVERSE.keys())}"
        )

    total_bytes = int(arr.nbytes)
    shape       = list(arr.shape)

    # Build JSON header — data_offsets are byte ranges in the data section
    header_dict = {
        layer_name: {
            "dtype":        dtype_str,
            "shape":        shape,
            "data_offsets": [0, total_bytes],
        }
    }
    header_json  = json.dumps(header_dict, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    # Safetensors requires the header block to be padded to an 8-byte
    # boundary with spaces (0x20). This matches the reference writer.
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    n_samples = shape[0] if shape else 0

    print(f"  [save_safetensors_streaming] {n_samples} samples "
          f"in chunks of {chunk_size}")
    print(f"  [save_safetensors_streaming] → {save_path}")

    with open(save_path, "wb") as f:
        # 8-byte little-endian uint64 = length of header
        f.write(struct.pack("<Q", len(header_bytes)))
        # JSON header
        f.write(header_bytes)
        # Stream data chunks — never more than chunk_size samples in RAM
        written = 0
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = np.asarray(arr[start:end])  # force disk read for memmap
            f.write(chunk.tobytes())
            del chunk
            written += (end - start)
            if written % 10000 == 0 or end == n_samples:
                print(f"    streamed {written}/{n_samples} samples")

    # Verify file size matches expectation
    actual_size   = os.path.getsize(save_path)
    expected_size = 8 + len(header_bytes) + total_bytes
    if actual_size != expected_size:
        raise RuntimeError(
            f"File size mismatch for {save_path}:\n"
            f"  expected {expected_size} bytes\n"
            f"  actual   {actual_size} bytes\n"
            f"Partial write — file is corrupt."
        )
    gb = actual_size / (1024 ** 3)
    print(f"  [save_safetensors_streaming] verified {gb:.2f} GB")


# ---------------------------------------------------------------------------
# Helper 3 — Open safetensors as lazy numpy memmap
# Source: pipeline_a/crop.py → load_original_activations
# ---------------------------------------------------------------------------

def open_safetensors_memmap(
    file_path: str,
    layer_name: str = None,
) -> Tuple[np.memmap, str]:
    """
    Open a .safetensors file as a lazy numpy memmap.

    Why this exists:
        Same mmap problem as read_safetensors_header — safetensors.safe_open()
        mmaps the entire file. This helper reads the JSON header manually,
        then creates a numpy.memmap at the tensor's byte offset. numpy memmap
        is lazy — only pages actually accessed (via indexing like memmap[i])
        are read from disk.

    Memory behavior:
        - open_safetensors_memmap returns immediately with ~0 RAM cost
        - memmap[i] reads sample i from disk into ~(sample_size) RAM
        - memmap[100:200] reads 100 samples into RAM
        - Never loads the whole file

    Args:
        file_path  : path to the .safetensors file
        layer_name : which tensor to open. If None, uses the first tensor
                     in the file (typical case for single-layer files).

    Returns:
        (mm, resolved_layer_name)
        mm                   — numpy.memmap with the tensor's full shape
        resolved_layer_name  — the actual layer name used (useful when
                                layer_name was None and we auto-selected)

    Example:
        mm, name = open_safetensors_memmap(
            "activations_layer4_2_conv3_output_raw.safetensors",
            "layer4.2.conv3"
        )
        print(mm.shape)   # (50000, 2048, 7, 7) — no data read yet
        sample = mm[42]    # NOW data for sample 42 is read from disk
    """
    file_path = str(file_path)
    header, header_len = read_safetensors_header(file_path)

    available_keys = list(header.keys())
    if not available_keys:
        raise ValueError(f"Safetensors file has no tensors: {file_path}")

    if layer_name and layer_name in available_keys:
        selected = layer_name
    else:
        selected = available_keys[0]
        if layer_name:
            print(f"  [open_safetensors_memmap] requested layer "
                  f"'{layer_name}' not found — using first available: "
                  f"'{selected}'")
            print(f"  [open_safetensors_memmap] Available layers: {available_keys}")

    info         = header[selected]
    shape        = info["shape"]
    dtype_str    = info["dtype"]
    data_offsets = info["data_offsets"]

    np_dtype = _DTYPE_MAP.get(dtype_str, np.float32)

    # Byte offset where tensor data starts in the file:
    #   [8 bytes header_len][header_len bytes JSON][tensor_start...]
    # The tensor's data_offsets[0] is the offset WITHIN the data section.
    tensor_start_byte = 8 + header_len + data_offsets[0]

    mm = np.memmap(
        file_path,
        dtype=np_dtype,
        mode="r",
        offset=tensor_start_byte,
        shape=tuple(shape),
    )

    return mm, selected


# ---------------------------------------------------------------------------
# Smoke test — requires an existing .safetensors file. Harmless if absent.
#   python -m neuron_viz_pipeline.src.utils.io_safetensors <path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.io_safetensors <path_to_safetensors>")
        print("       (provide path to an existing .safetensors file)")
        sys.exit(0)

    path = sys.argv[1]
    if not Path(path).is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"=== Header of {path} ===")
    header, header_len = read_safetensors_header(path)
    print(f"  header_len = {header_len} bytes")
    for key, info in header.items():
        print(f"  {key}: shape={info['shape']} dtype={info['dtype']}")

    print(f"\n=== Memmap test ===")
    mm, name = open_safetensors_memmap(path)
    print(f"  opened layer '{name}' — shape {mm.shape}, dtype {mm.dtype}")
    sample = np.asarray(mm[0])
    print(f"  sample 0 shape: {sample.shape}, "
          f"range [{sample.min():.4f}, {sample.max():.4f}]")
    print("\nSmoke test passed.")