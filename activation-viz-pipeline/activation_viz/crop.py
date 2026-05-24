# =============================================================================
# Activation Viz: activation_viz/crop.py
#
# ORIGIN: Adapted from preprocessing/crop_activation_regions.py
#
# CHANGES vs original crop_activation_regions.py:
# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 [CRITICAL — memory fix]:
#   load_original_activations() — original used safe_open() which mmaps the
#   entire file. For 18.69 GB files raises:
#     RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory
#   Fix: replaced with numpy memmap that reads lazily from disk.
#   Returns np.memmap instead of torch.Tensor — only the samples actually
#   accessed are read from disk, not the entire file.
#
# CHANGE 2 [CRITICAL — numpy/torch compatibility]:
#   get_activation_map() — original assumed activations was always a torch
#   tensor. With memmap fix, activations is now numpy array per sample.
#   Fix: added isinstance check + torch.from_numpy() conversion.
#
# CHANGE 3 [imports]:
#   Removed: from dsets import get_dataset
#            from models import get_fn_model_loader
#            from utils.helper import load_config
#            from timm.data import resolve_data_config / create_transform
#   Added:   from data.data_proces import MakeDataset  (CoE dataset loader)
#
# CHANGE 4 [dataset loading]:
#   Added _make_dataset() helper — replaces config-based get_dataset() calls.
#   Used in get_cropped_images_from_activations() and main().
#
# CHANGE 5 [CLI args]:
#   Removed: --config_file
#   Added:   --model_name, --data_path, --dataset
#
# WHAT IS VERBATIM from original:
#   resize_activation_map()        — unchanged
#   get_crop_bbox()                — ALL THREE methods unchanged
#   crop_and_resize_image()        — unchanged
#   create_alpha_mask_crop()       — unchanged
#   create_activation_overlay()    — unchanged
#   save_image_with_info()         — unchanged
#   get_cropped_images_from_activations() — unchanged except _make_dataset()
# =============================================================================

import argparse
import os
import math
import struct
import json
import types
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm
from safetensors import safe_open
from torchvision.transforms.functional import gaussian_blur

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Activation Viz CHANGE 3: CoE dataset loader
from data.data_proces import MakeDataset
import torchvision.transforms as T


# =============================================================================
# Activation Viz CHANGE 4: dataset helper
# Replaces: get_dataset(dataset_name)(data_path=..., preprocessing=False)
# =============================================================================

def _make_dataset(model_name, data_path, dataset='imagenet-val',
                  preprocessing=False):
    """
    Activation Viz: Create MakeDataset using CoE's loader.
    preprocessing=False → raw pixels (no normalisation) for saving as images
    preprocessing=True  → normalised tensors for model inference
    """
    a = types.SimpleNamespace()
    a.model_name    = model_name
    a.resume        = None
    a.dataset       = dataset
    a.dataset_name  = dataset.split('-')[0]
    a.dataset_split = dataset.split('-')[1]
    a.data_dir      = os.path.join(data_path, a.dataset_name)

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        *([] if not preprocessing else [
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    ])
    return MakeDataset(a, transform=transform, dataset_split=a.dataset_split)


# =============================================================================
# VERBATIM from crop_activation_regions.py — load_data unchanged
# =============================================================================

def load_data(indices_file: str,
              values_file: Optional[str] = None) -> Tuple[np.ndarray,
                                                           Optional[np.ndarray]]:
    """Load top activation indices and values — VERBATIM"""
    print(f"Loading indices from: {indices_file}")
    indices = np.load(indices_file)
    print(f"Indices shape: {indices.shape}")
    values = None
    if values_file and os.path.exists(values_file):
        print(f"Loading values from: {values_file}")
        values = np.load(values_file)
        print(f"Values shape: {values.shape}")
    return indices, values


# =============================================================================
# Activation Viz CHANGE 1: load_original_activations — replaces safe_open
# =============================================================================

def load_original_activations(activation_file: str,
                               layer_name: Optional[str] = None):
    """
    Activation Viz FIX: Load activation file as lazy numpy memmap.

    ORIGINAL CODE:
        with safe_open(activation_file, framework="pt", device="cpu") as f:
            activations = f.get_tensor(selected_layer)
        return activations, selected_layer

    WHY CHANGED:
        safe_open() mmaps the ENTIRE file immediately on open.
        For 18.69 GB files on limited RAM this raises:
          RuntimeError: unable to mmap 20070400104 bytes: Cannot allocate memory

    FIX:
        1. Read JSON header manually (tiny, no mmap needed)
        2. Create numpy memmap starting at tensor data offset
        numpy memmap is lazy — only reads pages when actually accessed.
        memmap[sample_idx] reads just one sample from disk, not the whole file.

    RETURN:
        (np.memmap, layer_name) instead of (torch.Tensor, layer_name)
        Callers must handle numpy input — see get_activation_map() fix.
    """
    print(f"Loading original activations from: {activation_file}")

    # ── read JSON header only (no mmap of tensor data) ─────────────────
    with open(activation_file, 'rb') as f:
        header_len_bytes = f.read(8)
        header_len       = struct.unpack('<Q', header_len_bytes)[0]
        header_json      = f.read(header_len).decode('utf-8').strip()
        header           = json.loads(header_json)

    header.pop('__metadata__', None)
    available_keys = list(header.keys())
    print(f"Available layers: {available_keys}")

    if layer_name and layer_name in available_keys:
        selected_layer = layer_name
    else:
        selected_layer = available_keys[0]
        print(f"Using layer: {selected_layer}")

    info         = header[selected_layer]
    shape        = info['shape']
    dtype_str    = info['dtype']
    data_offsets = info['data_offsets']

    dtype_map = {
        'F32': np.float32, 'F16': np.float16,
        'BF16': np.float32, 'I64': np.int64, 'I32': np.int32,
    }
    np_dtype = dtype_map.get(dtype_str, np.float32)

    # ── byte offset where tensor data starts ───────────────────────────
    tensor_start_byte = 8 + header_len + data_offsets[0]

    # ── lazy memmap — only pages actually accessed are read from disk ───
    mm = np.memmap(
        activation_file,
        dtype=np_dtype,
        mode='r',
        offset=tensor_start_byte,
        shape=tuple(shape)
    )

    print(f"Activation shape: {shape} (lazy memmap — reads on demand)")
    return mm, selected_layer


# =============================================================================
# Activation Viz CHANGE 2: get_activation_map — handles numpy memmap input
# =============================================================================

def get_activation_map(activations, sample_idx: int,
                        aggregation: str = "mean") -> Optional[np.ndarray]:
    """
    Extract and process activation map for a specific sample.

    Activation Viz CHANGE from original:
        Original assumed activations was always a torch.Tensor.
        With the memmap fix, activations is np.memmap — indexing returns
        a numpy array, not a tensor. torch methods (.mean(), .max(), etc.)
        do not work on numpy arrays.

    Fix: added isinstance check. If numpy, convert to torch first.
         np.array(memmap[idx]) forces the disk read for just that sample.
    """
    if sample_idx >= activations.shape[0]:
        raise ValueError(
            f"Sample index {sample_idx} out of range "
            f"(max: {activations.shape[0]-1})"
        )

    # Activation Viz CHANGE: memmap → numpy → torch
    raw = activations[sample_idx]
    if isinstance(raw, np.ndarray):
        # np.array() forces actual disk read of this one sample only
        sample_activation = torch.from_numpy(np.array(raw)).float()
    else:
        sample_activation = raw  # already a torch tensor

    if sample_activation.dim() == 1:
        return None
    elif sample_activation.dim() == 2:
        # ViT: [seq_len, hidden_dim] → reshape to spatial grid
        seq_len, hidden_dim = sample_activation.shape
        num_patches = seq_len - 1
        patch_size  = int(math.sqrt(num_patches))
        if patch_size * patch_size == num_patches:
            print(f"Reshaping ViT: {seq_len} tokens → {patch_size}x{patch_size}")
            spatial_tokens = sample_activation[1:]   # remove class token
            spatial_grid   = spatial_tokens.reshape(patch_size, patch_size, hidden_dim)
            sample_activation = spatial_grid.permute(2, 0, 1)
        else:
            print(f"Warning: Cannot reshape sequence length {seq_len}")
            return None

    if aggregation == "mean":
        activation_map = sample_activation.mean(dim=0)
    elif aggregation == "max":
        activation_map = sample_activation.max(dim=0)[0]
    elif aggregation == "sum":
        activation_map = sample_activation.sum(dim=0)
    elif aggregation == "raw":
        activation_map = sample_activation
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return activation_map.numpy()


# =============================================================================
# VERBATIM from crop_activation_regions.py — no changes to any of these
# =============================================================================

def resize_activation_map(activation_map: np.ndarray,
                           target_size: Tuple[int, int]) -> np.ndarray:
    """Resize activation map to image size via bilinear interpolation — VERBATIM"""
    if activation_map is None:
        return None
    activation_tensor = (torch.from_numpy(activation_map)
                         .unsqueeze(0).unsqueeze(0).float())
    resized = F.interpolate(
        activation_tensor, size=target_size,
        mode='bilinear', align_corners=False
    )
    return resized.squeeze().numpy()


def get_crop_bbox(activation_map: np.ndarray, method: str,
                   threshold_percentile: float = 90.0,
                   padding: int = 20) -> Tuple[int, int, int, int]:
    """
    Get bounding box for cropping based on activation map — VERBATIM.

    THREE METHODS:
      threshold: find pixels above percentile → bbox around active region
      bbox:      weighted centroid → square crop centered there
      center:    fixed center crop (ignores activation map)

    This is the zoom-first part of the renderer: it crops into the active
    region before any mask is applied, which produces tighter visualizations
    than masking the full frame.
    """
    H, W = activation_map.shape

    if method == "threshold":
        threshold = np.percentile(activation_map, threshold_percentile)
        mask      = activation_map >= threshold
        coords    = np.where(mask)
        if len(coords[0]) == 0:
            center_h, center_w = H // 2, W // 2
            crop_size = min(H, W) // 2
            y1 = max(0, center_h - crop_size)
            y2 = min(H, center_h + crop_size)
            x1 = max(0, center_w - crop_size)
            x2 = min(W, center_w + crop_size)
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            y1 = max(0, y_min - padding)
            y2 = min(H, y_max + padding)
            x1 = max(0, x_min - padding)
            x2 = min(W, x_max + padding)

    elif method == "bbox":
        y_coords, x_coords = np.indices(activation_map.shape)
        total_activation   = activation_map.sum()
        if total_activation > 0:
            center_y = (y_coords * activation_map).sum() / total_activation
            center_x = (x_coords * activation_map).sum() / total_activation
        else:
            center_y, center_x = H // 2, W // 2
        crop_size = min(H, W) // 2
        y1 = max(0, int(center_y - crop_size))
        y2 = min(H, int(center_y + crop_size))
        x1 = max(0, int(center_x - crop_size))
        x2 = min(W, int(center_x + crop_size))

    elif method == "center":
        center_h, center_w = H // 2, W // 2
        crop_size = min(H, W) // 2
        y1 = max(0, center_h - crop_size)
        y2 = min(H, center_h + crop_size)
        x1 = max(0, center_w - crop_size)
        x2 = min(W, center_w + crop_size)
    else:
        raise ValueError(f"Unknown crop method: {method}")

    return x1, y1, x2, y2


def crop_and_resize_image(image: Image.Image,
                           bbox: Tuple[int, int, int, int],
                           target_size: int) -> Image.Image:
    """Crop and resize image based on bounding box — VERBATIM"""
    x1, y1, x2, y2 = bbox
    cropped = image.crop((x1, y1, x2, y2))
    return cropped.resize((target_size, target_size), Image.LANCZOS)


def create_alpha_mask_crop(image: Image.Image,
                            activation_map: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            target_size: int,
                            threshold_percentile: float = 50.0) -> Image.Image:
    """
    Create cropped image with alpha mask based on activation map — VERBATIM.

    Steps (all from original):
      1. Crop image and activation map to bbox (zoom into active region first)
      2. Resize both to target_size
      3. Apply 51x51 Gaussian blur (smooths ViT patch boundaries)
      4. Percentile threshold → binary mask
      5. PIL composite: activated pixels visible, rest black
    """
    x1, y1, x2, y2 = bbox

    cropped_image      = image.crop((x1, y1, x2, y2))
    cropped_activation = activation_map[y1:y2, x1:x2]

    resized_image = cropped_image.resize((target_size, target_size), Image.LANCZOS)

    activation_tensor = (torch.from_numpy(cropped_activation)
                         .unsqueeze(0).unsqueeze(0).float())
    resized_activation = F.interpolate(
        activation_tensor, size=(target_size, target_size),
        mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # 51x51 Gaussian blur — intentionally large to smooth patch boundaries
    blurred = gaussian_blur(
        torch.from_numpy(resized_activation).unsqueeze(0).unsqueeze(0),
        kernel_size=[51, 51]
    )
    resized_activation = blurred.squeeze().numpy()

    activation_threshold = np.percentile(resized_activation, threshold_percentile)
    alpha_mask = (resized_activation >= activation_threshold).astype(np.uint8) * 255

    alpha_mask_pil   = Image.fromarray(alpha_mask, mode='L')
    resized_image    = resized_image.convert('RGBA')
    black_background = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 255))
    masked_image     = Image.composite(resized_image, black_background, alpha_mask_pil)
    return masked_image


def create_activation_overlay(image: Image.Image,
                               activation_map: np.ndarray,
                               alpha: float = 0.4) -> Image.Image:
    """Create JET colormap overlay of activation on original image — VERBATIM"""
    activation_norm = (
        (activation_map - activation_map.min()) /
        (activation_map.max() - activation_map.min() + 1e-8)
    )
    activation_colored = cv2.applyColorMap(
        (activation_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    activation_colored = cv2.cvtColor(activation_colored, cv2.COLOR_BGR2RGB)
    activation_img = Image.fromarray(activation_colored)
    activation_img = activation_img.resize(image.size, Image.LANCZOS)
    return Image.blend(image, activation_img, alpha)


def save_image_with_info(image: Image.Image, save_path: str, info: Dict):
    """Save image with companion JSON metadata — VERBATIM"""
    image.save(save_path)
    info_path = (save_path.replace('.jpg', '_info.json')
                          .replace('.png', '_info.json'))
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


# =============================================================================
# get_cropped_images_from_activations — VERBATIM except _make_dataset()
# Activation Viz CHANGE 4: dataset loading via _make_dataset()
# =============================================================================

def get_cropped_images_from_activations(
    sample_indices: List[int],
    neuron_idx: int,
    activations,               # np.memmap or torch.Tensor
    model_name: str,
    data_path: str,
    dataset: str = 'imagenet-val',
    layer_name: str = "unknown",
    crop_method: str = "threshold",
    threshold_percentile: float = 90.0,
    crop_size: int = 224,
    padding: int = 20,
    alpha_mask: bool = True,
    mask_threshold: float = 50.0
) -> List[Tuple[Image.Image, Dict]]:
    """
    FASTEST entry point: uses pre-computed activations, no model inference.
    Activation Viz CHANGE: uses _make_dataset() instead of get_dataset().
    """
    print(f"Activation Viz [crop]: {len(sample_indices)} samples, "
          f"neuron={neuron_idx}, method={crop_method}, alpha_mask={alpha_mask}")

    # Activation Viz CHANGE: CoE-style dataset loading
    dataset_original = _make_dataset(model_name, data_path, dataset,
                                      preprocessing=False)

    valid_sample_indices = [
        idx for idx in sample_indices
        if 0 <= idx < len(dataset_original) and idx < activations.shape[0]
    ]
    if len(valid_sample_indices) != len(sample_indices):
        print(f"  Warning: {len(sample_indices) - len(valid_sample_indices)} "
              f"invalid indices filtered out")

    # detect architecture from activation shape
    sample_raw = activations[0]
    if isinstance(sample_raw, np.ndarray):
        sample_act = torch.from_numpy(np.array(sample_raw)).float()
    else:
        sample_act = sample_raw

    has_spatial = True
    if sample_act.dim() == 1:
        has_spatial = False
    elif sample_act.dim() == 2:
        seq_len, hidden_dim = sample_act.shape
        num_patches = seq_len - 1
        patch_size  = int(math.sqrt(num_patches))
        if patch_size * patch_size != num_patches:
            has_spatial = False

    if has_spatial and neuron_idx >= activations.shape[1]:
        raise ValueError(
            f"neuron_idx {neuron_idx} >= num_neurons {activations.shape[1]}"
        )

    results = []

    for sample_idx in valid_sample_indices:
        try:
            image, label = dataset_original[sample_idx]

            if not isinstance(image, Image.Image):
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                if isinstance(image, np.ndarray):
                    if image.ndim == 3 and image.shape[0] == 3:
                        image = image.transpose(1, 2, 0)
                    if image.max() <= 1.0:
                        image = (image * 255).clip(0, 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                    image = Image.fromarray(image)

            if not has_spatial:
                W, H = image.size
                csz  = min(W, H) // 2
                cx, cy = W // 2, H // 2
                bbox = (max(0, cx - csz), max(0, cy - csz),
                        min(W, cx + csz), min(H, cy + csz))
                cropped_image = crop_and_resize_image(image, bbox, crop_size)
            else:
                activation_map = get_activation_map(
                    activations, sample_idx, aggregation="raw"
                )
                if activation_map is None:
                    W, H = image.size
                    csz  = min(W, H) // 2
                    cx, cy = W // 2, H // 2
                    bbox = (max(0, cx - csz), max(0, cy - csz),
                            min(W, cx + csz), min(H, cy + csz))
                    cropped_image = crop_and_resize_image(image, bbox, crop_size)
                else:
                    neuron_activation_map = activation_map[neuron_idx]
                    resized_activation = resize_activation_map(
                        neuron_activation_map, image.size[::-1]
                    )
                    bbox = get_crop_bbox(
                        resized_activation, crop_method,
                        threshold_percentile, padding
                    )
                    if alpha_mask:
                        cropped_image = create_alpha_mask_crop(
                            image, resized_activation, bbox,
                            crop_size, mask_threshold
                        )
                    else:
                        cropped_image = crop_and_resize_image(
                            image, bbox, crop_size
                        )

            metadata = {
                "neuron_idx":  int(neuron_idx),
                "sample_idx":  int(sample_idx),
                "label":       (int(label) if isinstance(label, (int, np.integer))
                                else str(label)),
                "layer_name":  layer_name,
                "crop_bbox":   [int(x) for x in bbox],
                "crop_method": crop_method,
                "alpha_mask":  alpha_mask,
            }
            results.append((cropped_image, metadata))

        except Exception as e:
            print(f"  Error processing sample {sample_idx}: {e}")
            continue

    print(f"  Successfully processed {len(results)}/{len(valid_sample_indices)}")
    return results


# =============================================================================
# Activation Viz: CLI args and main()
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Activation Viz - Step 3: Crop and mask images"
    )

    # Activation Viz CHANGE 5: direct args instead of --config_file
    parser.add_argument('--model_name', type=str, default='rn152')
    parser.add_argument('--data_path',  type=str, default='./dataset')
    parser.add_argument('--dataset',    type=str, default='imagenet-val')

    parser.add_argument('--indices_file',    type=str, required=True)
    parser.add_argument('--values_file',     type=str, default=None)
    parser.add_argument('--activation_file', type=str, required=True)
    parser.add_argument('--output_dir',      type=str,
                        default='results/activation_viz/cropped_regions')
    parser.add_argument('--layer_name',      type=str, default=None)
    parser.add_argument('--neuron_indices',  type=int, nargs='+', default=None)
    parser.add_argument('--all_neurons',     action='store_true')
    parser.add_argument('--top_k_samples',   type=int, default=150)
    parser.add_argument('--crop_method',     type=str, default='threshold',
                        choices=['threshold', 'bbox', 'center'])
    parser.add_argument('--threshold_percentile', type=float, default=90.0)
    parser.add_argument('--crop_size',       type=int,   default=224)
    parser.add_argument('--padding',         type=int,   default=20)
    parser.add_argument('--save_overlay',    action='store_true')
    parser.add_argument('--alpha_mask',      action='store_true')
    parser.add_argument('--mask_threshold',  type=float, default=50.0)

    return parser.parse_args()


def main():
    args = get_args()

    print(f"Activation Viz [crop]:")
    print(f"  model       = {args.model_name}")
    print(f"  crop_method = {args.crop_method}")
    print(f"  alpha_mask  = {args.alpha_mask}")
    print(f"  top_k       = {args.top_k_samples}")

    indices, values = load_data(args.indices_file, args.values_file)

    # Activation Viz CHANGE 1: load as lazy memmap (no safe_open mmap)
    activations, layer_name = load_original_activations(
        args.activation_file, args.layer_name
    )

    # Activation Viz CHANGE 4: MakeDataset instead of get_dataset()
    dataset = _make_dataset(
        args.model_name, args.data_path, args.dataset, preprocessing=False
    )
    print(f"  Dataset loaded: {len(dataset)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_neurons:
        neuron_indices_to_process = list(range(indices.shape[0]))
    elif args.neuron_indices is not None:
        neuron_indices_to_process = args.neuron_indices
    else:
        neuron_indices_to_process = [0, 1, 2, 3, 4]

    print(f"  Processing neurons: {neuron_indices_to_process}")

    for neuron_idx in tqdm(neuron_indices_to_process, desc="Processing neurons"):
        if neuron_idx >= indices.shape[0]:
            print(f"  Skipping neuron {neuron_idx} (out of range)")
            continue

        neuron_dir = os.path.join(args.output_dir, f"neuron_{neuron_idx:03d}")
        os.makedirs(neuron_dir, exist_ok=True)

        neuron_sample_indices = indices[neuron_idx, :args.top_k_samples]
        neuron_values_row = (values[neuron_idx, :args.top_k_samples]
                             if values is not None else None)

        for rank, sample_idx in enumerate(neuron_sample_indices):
            if sample_idx == -1:
                continue
            try:
                image, label = dataset[sample_idx]

                if not isinstance(image, Image.Image):
                    if isinstance(image, torch.Tensor):
                        image = image.detach().cpu().numpy()
                    if isinstance(image, np.ndarray):
                        if image.ndim == 3 and image.shape[0] == 3:
                            image = image.transpose(1, 2, 0)
                        if image.max() <= 1.0:
                            image = (image * 255).clip(0, 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                        image = Image.fromarray(image)

                # get_activation_map handles memmap → torch conversion
                activation_map = get_activation_map(
                    activations, sample_idx, aggregation="raw"
                )

                if activation_map is None:
                    W, H = image.size
                    csz  = min(W, H) // 2
                    cx, cy = W // 2, H // 2
                    bbox = (max(0, cx - csz), max(0, cy - csz),
                            min(W, cx + csz), min(H, cy + csz))
                    cropped_image        = crop_and_resize_image(image, bbox,
                                                                  args.crop_size)
                    resized_activation   = None
                    cropped_without_mask = None
                else:
                    neuron_activation_map = activation_map[neuron_idx]
                    resized_activation = resize_activation_map(
                        neuron_activation_map, image.size[::-1]
                    )
                    # get_crop_bbox zooms into active region first
                    bbox = get_crop_bbox(
                        resized_activation, args.crop_method,
                        args.threshold_percentile, args.padding
                    )
                    if args.alpha_mask:
                        cropped_image = create_alpha_mask_crop(
                            image, resized_activation, bbox,
                            args.crop_size, args.mask_threshold
                        )
                    else:
                        cropped_image = crop_and_resize_image(
                            image, bbox, args.crop_size
                        )
                    cropped_without_mask = crop_and_resize_image(
                        image, bbox, args.crop_size
                    )

                info = {
                    "neuron_idx":       int(neuron_idx),
                    "sample_idx":       int(sample_idx),
                    "rank":             int(rank),
                    "activation_value": (float(neuron_values_row[rank])
                                         if neuron_values_row is not None else None),
                    "label":            (int(label)
                                         if isinstance(label, (int, np.integer))
                                         else str(label)),
                    "layer_name":       layer_name,
                    "crop_bbox":        [int(x) for x in bbox],
                    "crop_method":      ("center_crop"
                                         if resized_activation is None
                                         else args.crop_method),
                    "alpha_mask":       args.alpha_mask,
                    "mask_threshold":   (args.mask_threshold
                                         if args.alpha_mask else None),
                }

                ext = ".png" if args.alpha_mask else ".jpg"
                crop_path = os.path.join(
                    neuron_dir,
                    f"rank_{rank:04d}_sample_{sample_idx}_crop{ext}"
                )
                save_image_with_info(cropped_image, crop_path, info)

                if cropped_without_mask is not None:
                    cropped_without_mask.save(os.path.join(
                        neuron_dir,
                        f"rank_{rank:04d}_sample_{sample_idx}_crop_no_mask.jpg"
                    ))

                if args.save_overlay and resized_activation is not None:
                    overlay = create_activation_overlay(image, resized_activation)
                    overlay.save(os.path.join(
                        neuron_dir,
                        f"rank_{rank:04d}_sample_{sample_idx}_overlay.jpg"
                    ))

            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                import traceback; traceback.print_exc()
                continue

    print(f"\nActivation Viz [crop]: Complete. Results -> {args.output_dir}")


if __name__ == "__main__":
    main()
