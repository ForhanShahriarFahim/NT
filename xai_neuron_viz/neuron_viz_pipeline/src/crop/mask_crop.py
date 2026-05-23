# =============================================================================
# FILE: neuron_viz_pipeline/src/crop/mask_crop.py
#
# Purpose:
#   Cropping and alpha-masking functions for Stage 4.
#
#   These functions are a VERBATIM port (minus 3 debug prints) of lines
#   207-380 from preprocessing/crop_activation_regions.py. They take a
#   generic 2D saliency map [H, W] float and produce cropped/masked PIL
#   images — they don't care whether the saliency came from raw activations
#   (original pipeline) or IxG (our new pipeline).
#
# What was ported verbatim:
#   - resize_activation_map      (lines 207-223)
#   - get_crop_bbox              (lines 226-280)
#   - crop_and_resize_image      (lines 283-294)
#   - create_alpha_mask_crop     (lines 297-348) — MINUS 3 debug prints
#   - create_activation_overlay  (lines 351-368)
#   - save_image_with_info       (lines 371-379)
#
# What was removed (debug prints inside create_alpha_mask_crop):
#   - line 300: print("create_alpha_mask_crop called with activation_map shape...")
#   - lines 329-332: print("Alpha mask: X/Y pixels kept ...")
#   - line 346: print("Alpha mask crop completed - image mode: ...")
#   These fired ~3 times per image × 50-150 images per run = 150-450 lines of
#   routine log spam per neuron. Removing them has ZERO effect on the saved
#   PNG/JPG/JSON files — the saliency math, bbox computation, alpha mask,
#   and image compositing are all unchanged.
#
# Other than those 3 removed prints, every line — variable names, comments,
# function signatures, magic numbers (kernel_size=[51, 51], percentile 90,
# etc.) — matches the reference exactly.
#
# Input contract:
#   activation_map (despite its name) is any [H, W] float32 numpy array where
#   higher values mean "this pixel matters more." For us, this is IxG saliency.
#   For the original pipeline_a, it was raw activation magnitude.
# =============================================================================

import json
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur


# ---------------------------------------------------------------------------
# resize_activation_map
# Source: crop_activation_regions.py lines 207-223 (verbatim)
# ---------------------------------------------------------------------------

def resize_activation_map(activation_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize activation map to match image size"""
    if activation_map is None:
        return None

    # Convert to tensor for interpolation
    activation_tensor = torch.from_numpy(activation_map).unsqueeze(0).unsqueeze(0).float()

    # Resize using bilinear interpolation
    resized = F.interpolate(
        activation_tensor,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )

    return resized.squeeze().numpy()


# ---------------------------------------------------------------------------
# get_crop_bbox
# Source: crop_activation_regions.py lines 226-280 (verbatim)
# ---------------------------------------------------------------------------

def get_crop_bbox(activation_map: np.ndarray, method: str, threshold_percentile: float = 90.0,
                  padding: int = 20) -> Tuple[int, int, int, int]:
    """Get bounding box for cropping based on activation map"""
    H, W = activation_map.shape

    if method == "threshold":
        # Threshold-based cropping
        threshold = np.percentile(activation_map, threshold_percentile)
        mask = activation_map >= threshold

        # Find bounding box of activated region
        coords = np.where(mask)
        if len(coords[0]) == 0:
            # Fallback to center crop if no activations above threshold
            center_h, center_w = H // 2, W // 2
            crop_size = min(H, W) // 2
            y1, y2 = max(0, center_h - crop_size), min(H, center_h + crop_size)
            x1, x2 = max(0, center_w - crop_size), min(W, center_w + crop_size)
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Add padding
            y1 = max(0, y_min - padding)
            y2 = min(H, y_max + padding)
            x1 = max(0, x_min - padding)
            x2 = min(W, x_max + padding)

    elif method == "bbox":
        # Use entire activation map to find bounding box
        # Find center of mass
        y_coords, x_coords = np.indices(activation_map.shape)
        total_activation = activation_map.sum()

        if total_activation > 0:
            center_y = (y_coords * activation_map).sum() / total_activation
            center_x = (x_coords * activation_map).sum() / total_activation
        else:
            center_y, center_x = H // 2, W // 2

        # Create bounding box around center
        crop_size = min(H, W) // 2
        y1 = max(0, int(center_y - crop_size))
        y2 = min(H, int(center_y + crop_size))
        x1 = max(0, int(center_x - crop_size))
        x2 = min(W, int(center_x + crop_size))

    elif method == "center":
        # Simple center crop
        center_h, center_w = H // 2, W // 2
        crop_size = min(H, W) // 2
        y1, y2 = max(0, center_h - crop_size), min(H, center_h + crop_size)
        x1, x2 = max(0, center_w - crop_size), min(W, center_w + crop_size)

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# crop_and_resize_image
# Source: crop_activation_regions.py lines 283-294 (verbatim)
# ---------------------------------------------------------------------------

def crop_and_resize_image(image: Image.Image, bbox: Tuple[int, int, int, int],
                         target_size: int) -> Image.Image:
    """Crop and resize image based on bounding box"""
    x1, y1, x2, y2 = bbox

    # Crop image
    cropped = image.crop((x1, y1, x2, y2))

    # Resize to target size
    resized = cropped.resize((target_size, target_size), Image.LANCZOS)

    return resized


# ---------------------------------------------------------------------------
# create_alpha_mask_crop
# Source: crop_activation_regions.py lines 297-348
# CHANGES: 3 debug print() calls removed (see file header for list).
# All other lines are identical to the reference.
# ---------------------------------------------------------------------------

def create_alpha_mask_crop(image: Image.Image, activation_map: np.ndarray, bbox: Tuple[int, int, int, int],
                          target_size: int, threshold_percentile: float = 50.0) -> Image.Image:
    """Create cropped image with alpha mask based on activation map"""
    # [REMOVED] print(f"create_alpha_mask_crop called with activation_map shape: {activation_map.shape}")
    x1, y1, x2, y2 = bbox

    # Crop image
    cropped_image = image.crop((x1, y1, x2, y2))

    # Crop activation map
    cropped_activation = activation_map[y1:y2, x1:x2]

    # Resize both image and activation map
    resized_image = cropped_image.resize((target_size, target_size), Image.LANCZOS)

    # Resize activation map
    activation_tensor = torch.from_numpy(cropped_activation).unsqueeze(0).unsqueeze(0).float()
    resized_activation = F.interpolate(
        activation_tensor,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Add Gaussian blur (kernel_size must be odd and > 0)
    blurred_activation = gaussian_blur(torch.from_numpy(resized_activation).unsqueeze(0).unsqueeze(0), kernel_size=[51, 51])
    resized_activation = blurred_activation.squeeze().numpy()

    # Create alpha mask based on activation threshold
    activation_threshold = np.percentile(resized_activation, threshold_percentile)
    alpha_mask = (resized_activation >= activation_threshold).astype(np.uint8) * 255

    # [REMOVED debug info print:]
    # pixels_kept = np.sum(alpha_mask > 0)
    # total_pixels = alpha_mask.size
    # print(f"Alpha mask: {pixels_kept}/{total_pixels} pixels kept ({pixels_kept/total_pixels*100:.1f}%) with threshold {threshold_percentile}%")

    # Convert to PIL Image
    alpha_mask_pil = Image.fromarray(alpha_mask, mode='L')

    # Convert image to RGBA if not already
    if resized_image.mode != 'RGBA':
        resized_image = resized_image.convert('RGBA')

    # Create black background
    black_background = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 255))

    # Apply alpha mask - only keep activated regions
    masked_image = Image.composite(resized_image, black_background, alpha_mask_pil)
    # [REMOVED] print(f"Alpha mask crop completed - image mode: {masked_image.mode}")

    return masked_image


# ---------------------------------------------------------------------------
# create_activation_overlay
# Source: crop_activation_regions.py lines 351-368 (verbatim)
# ---------------------------------------------------------------------------

def create_activation_overlay(image: Image.Image, activation_map: np.ndarray,
                            alpha: float = 0.4) -> Image.Image:
    """Create overlay of activation map on original image"""
    # Normalize activation map
    activation_norm = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)

    # Convert to heatmap
    activation_colored = cv2.applyColorMap((activation_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    activation_colored = cv2.cvtColor(activation_colored, cv2.COLOR_BGR2RGB)
    activation_img = Image.fromarray(activation_colored)

    # Resize to match image
    activation_img = activation_img.resize(image.size, Image.LANCZOS)

    # Create overlay
    overlay = Image.blend(image, activation_img, alpha)

    return overlay


# ---------------------------------------------------------------------------
# save_image_with_info
# Source: crop_activation_regions.py lines 371-379 (verbatim)
# ---------------------------------------------------------------------------

def save_image_with_info(image: Image.Image, save_path: str, info: Dict):
    """Save image with metadata"""
    # Save image
    image.save(save_path)

    # Save metadata
    info_path = save_path.replace('.jpg', '_info.json').replace('.png', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)