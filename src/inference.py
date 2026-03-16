"""
Single-image inference for the Drywall QA Prompted Segmentation project.

Given an image and a text prompt (e.g. "wall crack" or "taping area"),
runs CLIPSeg forward, post-processes the output, and saves the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from .config import DrywallQAConfig, load_config
from .evaluate import load_trained_model
from .model import build_model, load_processor
from .process import clean_boundaries, create_overlay, postprocess_mask


@torch.no_grad()
def run_inference(
    image_path: str | Path,
    text_prompt: str,
    model: CLIPSegForImageSegmentation,
    processor: CLIPSegProcessor,
    config: DrywallQAConfig,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run prompted segmentation on a single image.

    Args:
        image_path: Path to the input image.
        text_prompt: Natural language description of what to segment
            (e.g. "wall crack", "drywall seam", "taping area").
        model: Trained CLIPSeg model.
        processor: CLIPSeg processor.
        config: Project configuration.
        device: Torch device override.

    Returns:
        Tuple of:
            - ``image``: Original image as RGB numpy array (H, W, 3).
            - ``raw_mask``: Raw probability mask (H, W), float32 [0, 1].
            - ``processed_mask``: Cleaned binary mask (H, W), uint8 {0, 255}.
            - ``overlay``: Image with mask overlay (H, W, 3), uint8.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(0).cpu()  # (H, W)

    # Convert to numpy
    image_np = np.array(image)

    # Raw probability map
    raw_probs = torch.sigmoid(logits).numpy()

    # Resize to original image dimensions
    raw_probs_resized = cv2.resize(
        raw_probs,
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Resize logits for post-processing
    logits_resized = cv2.resize(
        logits.numpy(),
        (original_size[0], original_size[1]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Post-process
    processed_mask = postprocess_mask(logits_resized, config.postprocess)
    processed_mask = clean_boundaries(processed_mask, kernel_size=3)

    # Create overlay
    overlay = create_overlay(image_np, processed_mask, color=(0, 255, 0), alpha=0.4)

    return image_np, raw_probs_resized, processed_mask, overlay


def save_results(
    image: np.ndarray,
    raw_mask: np.ndarray,
    processed_mask: np.ndarray,
    overlay: np.ndarray,
    output_dir: str | Path,
    prefix: str = "result",
):
    """Save all inference outputs to disk.

    Args:
        image: Original image.
        raw_mask: Raw probability mask.
        processed_mask: Cleaned binary mask.
        overlay: Overlay image.
        output_dir: Directory to save results.
        prefix: Filename prefix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    cv2.imwrite(
        str(output_dir / f"{prefix}_input.png"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )

    # Save raw probability heatmap
    raw_vis = (raw_mask * 255).astype(np.uint8)
    raw_coloured = cv2.applyColorMap(raw_vis, cv2.COLORMAP_HOT)
    cv2.imwrite(str(output_dir / f"{prefix}_raw_heatmap.png"), raw_coloured)

    # Save processed mask
    cv2.imwrite(str(output_dir / f"{prefix}_mask.png"), processed_mask)

    # Save overlay
    cv2.imwrite(
        str(output_dir / f"{prefix}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    print(f"[inference] Results saved to {output_dir}/")
    print(f"  - {prefix}_input.png")
    print(f"  - {prefix}_raw_heatmap.png")
    print(f"  - {prefix}_mask.png")
    print(f"  - {prefix}_overlay.png")


def infer_from_checkpoint(
    image_path: str,
    text_prompt: str,
    checkpoint_path: str,
    config_path: str | None = None,
    output_dir: str | None = None,
):
    """Full inference pipeline from a saved checkpoint.

    Convenience function that loads config, model, and processor,
    then runs inference and saves results.

    Args:
        image_path: Path to the input image.
        text_prompt: Natural language prompt.
        checkpoint_path: Path to the model checkpoint.
        config_path: Optional path to YAML config.
        output_dir: Optional output directory override.
    """
    config = load_config(config_path)
    processor = load_processor(config)
    model = load_trained_model(checkpoint_path, config)

    if output_dir is None:
        output_dir = str(Path(config.project_root) / config.output.dir / "inference")

    image, raw_mask, processed_mask, overlay = run_inference(
        image_path, text_prompt, model, processor, config,
    )

    # Create a descriptive prefix
    prompt_slug = text_prompt.replace(" ", "_").lower()[:30]
    img_name = Path(image_path).stem
    prefix = f"{img_name}_{prompt_slug}"

    save_results(image, raw_mask, processed_mask, overlay, output_dir, prefix)
