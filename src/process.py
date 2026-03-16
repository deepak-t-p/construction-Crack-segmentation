"""
Post-processing utilities for predicted segmentation masks.

Applies morphological operations (opening / closing) and small-component
removal to clean up raw CLIPSeg logit outputs.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import PostprocessConfig


def postprocess_mask(
    raw_logits: torch.Tensor | np.ndarray,
    config: PostprocessConfig | None = None,
    threshold: float = 0.5,
    morph_kernel_size: int = 5,
    min_component_area: int = 100,
    use_opening: bool = True,
    use_closing: bool = True,
) -> np.ndarray:
    """Convert raw logits to a clean binary mask.

    Pipeline:
        1. Sigmoid activation
        2. Threshold to binary
        3. Morphological opening (remove small noise)
        4. Morphological closing (fill small holes)
        5. Connected-component filtering (remove blobs < min_area)

    Args:
        raw_logits: Model output logits, shape ``(H, W)`` (single image).
        config: Optional PostprocessConfig. If provided, overrides
            individual keyword arguments.
        threshold: Binarisation threshold.
        morph_kernel_size: Size of the structuring element.
        min_component_area: Minimum area (pixels) to keep a component.
        use_opening: Whether to apply morphological opening.
        use_closing: Whether to apply morphological closing.

    Returns:
        Cleaned binary mask as a uint8 numpy array, values {0, 255}.
    """
    # Use config values if provided
    if config is not None:
        threshold = config.threshold
        morph_kernel_size = config.morph_kernel_size
        min_component_area = config.min_component_area
        use_opening = config.use_opening
        use_closing = config.use_closing

    # Convert to numpy if necessary
    if isinstance(raw_logits, torch.Tensor):
        logits_np = raw_logits.detach().cpu().float().numpy()
    else:
        logits_np = raw_logits.astype(np.float32)

    # Sigmoid → threshold
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    mask = (probs >= threshold).astype(np.uint8) * 255

    # Morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )

    if use_opening:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if use_closing:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small connected components
    mask = _remove_small_components(mask, min_component_area)

    return mask


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` pixels.

    Args:
        mask: Binary mask (uint8, values {0, 255}).
        min_area: Minimum pixel count to retain a component.

    Returns:
        Filtered binary mask.
    """
    if min_area <= 0:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for i in range(1, n_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def clean_boundaries(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth ragged mask boundaries using Gaussian blur + re-threshold.

    Args:
        mask: Binary mask (uint8, values {0, 255}).
        kernel_size: Gaussian kernel size (must be odd).

    Returns:
        Boundary-smoothed binary mask.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return smoothed


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a coloured mask on top of an image.

    Args:
        image: RGB image, shape ``(H, W, 3)``, uint8.
        mask: Binary mask, shape ``(H, W)``, uint8 {0, 255}.
        color: RGB colour for the overlay.
        alpha: Opacity of the overlay.

    Returns:
        Blended image with mask overlay.
    """
    overlay = image.copy()
    coloured = np.zeros_like(image)
    coloured[:] = color

    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha,
        coloured[mask_bool], alpha,
        0,
    )

    return overlay
