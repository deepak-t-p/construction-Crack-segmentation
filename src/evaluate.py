"""
Evaluation pipeline for the Drywall QA Prompted Segmentation project.

Runs the full validation set through the model, computes per-category
metrics, and generates publication-quality visualizations:
  - Loss curves (train vs. validation)
  - Per-category bar charts (mIoU, Dice)
  - Side-by-side grids: Input | Ground Truth | Raw Prediction | Processed Prediction
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation

from .config import DrywallQAConfig, load_config
from .metrics import MetricTracker
from .model import build_model, load_processor
from .process import create_overlay, postprocess_mask


@torch.no_grad()
def evaluate_model(
    model: CLIPSegForImageSegmentation,
    dataloader,
    config: DrywallQAConfig,
    device: torch.device | None = None,
) -> dict:
    """Run full evaluation on a dataloader.

    Args:
        model: Trained CLIPSeg model.
        dataloader: Validation DataLoader.
        config: Project configuration.
        device: Torch device.

    Returns:
        Dict with overall and per-category metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    tracker = MetricTracker(threshold=config.postprocess.threshold)
    total_loss = 0.0
    n_batches = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["mask"].to(device)
        categories = batch.get("category", None)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        if logits.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(
                masks.unsqueeze(1), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1)

        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1

        tracker.update(logits, masks, categories)

    results = tracker.compute()
    results["val_loss"] = total_loss / max(n_batches, 1)
    return results


def load_trained_model(
    checkpoint_path: str | Path,
    config: DrywallQAConfig,
) -> CLIPSegForImageSegmentation:
    """Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        config: Project configuration.

    Returns:
        The loaded CLIPSeg model with trained weights.
    """
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[evaluate] Loaded checkpoint from epoch {ckpt['epoch']}, "
        f"step {ckpt['global_step']}, best mIoU={ckpt['best_miou']:.4f}"
    )
    return model


# ────────────────────────────────────────────────────────────────
#  Visualization
# ────────────────────────────────────────────────────────────────

def plot_loss_curves(log_path: str | Path, output_dir: str | Path):
    """Plot training and validation loss curves from the CSV log.

    Args:
        log_path: Path to the ``training_log.csv``.
        output_dir: Directory to save the plot.
    """
    log_path = Path(log_path)
    output_dir = Path(output_dir)

    steps, train_losses, val_losses = [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_losses.append(float(row["val_loss"]))
            else:
                val_losses.append(None)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(steps, train_losses, label="Train Loss", color="#2196F3", linewidth=2)

    # Filter out None values for val_losses
    val_steps = [s for s, v in zip(steps, val_losses) if v is not None]
    val_vals = [v for v in val_losses if v is not None]
    if val_vals:
        ax.plot(val_steps, val_vals, label="Val Loss", color="#FF5722", linewidth=2)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss (BCEWithLogits)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = output_dir / "loss_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Loss curves saved to {save_path}")


def plot_category_metrics(results: dict, output_dir: str | Path):
    """Plot per-category bar charts for mIoU and Dice.

    Args:
        results: Evaluation results dict from :func:`evaluate_model`.
        output_dir: Directory to save the plots.
    """
    output_dir = Path(output_dir)
    per_cat = results.get("per_category", {})
    if not per_cat:
        print("[visualize] No per-category results to plot.")
        return

    categories = list(per_cat.keys())
    iou_values = [per_cat[c]["mIoU"] for c in categories]
    dice_values = [per_cat[c]["mDice"] for c in categories]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours = ["#4CAF50", "#FF9800", "#2196F3", "#E91E63"]

    # mIoU bar chart
    bars1 = axes[0].bar(
        categories, iou_values,
        color=colours[:len(categories)], edgecolor="white", linewidth=1.2,
    )
    axes[0].set_ylabel("mIoU", fontsize=12)
    axes[0].set_title("Mean IoU by Category", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars1, iou_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold",
        )

    # Dice bar chart
    bars2 = axes[1].bar(
        categories, dice_values,
        color=colours[:len(categories)], edgecolor="white", linewidth=1.2,
    )
    axes[1].set_ylabel("Dice Coefficient", fontsize=12)
    axes[1].set_title("Mean Dice by Category", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars2, dice_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold",
        )

    fig.tight_layout()
    save_path = output_dir / "category_metrics.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Category metrics saved to {save_path}")


@torch.no_grad()
def generate_prediction_grid(
    model: CLIPSegForImageSegmentation,
    dataloader,
    config: DrywallQAConfig,
    n_samples: int = 8,
    output_dir: str | Path | None = None,
    device: torch.device | None = None,
):
    """Generate side-by-side visualizations: Input | GT | Raw Pred | Processed Pred.

    Args:
        model: Trained model.
        dataloader: Validation DataLoader.
        config: Project config.
        n_samples: Number of samples to visualize.
        output_dir: Save directory.
        device: Torch device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if output_dir is None:
        output_dir = Path(config.project_root) / config.output.dir
    output_dir = Path(output_dir)

    model.eval()
    model.to(device)

    collected = 0
    images, gts, raw_preds, proc_preds, prompts_list = [], [], [], [], []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["mask"]
        batch_prompts = batch.get("prompt", [""] * len(masks))

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        logits = outputs.logits.cpu()

        for i in range(len(masks)):
            if collected >= n_samples:
                break

            img = pixel_values[i].cpu().permute(1, 2, 0).numpy()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

            gt = masks[i].numpy()
            gt_vis = (gt * 255).astype(np.uint8)

            raw = torch.sigmoid(logits[i]).numpy()
            raw_vis = (raw * 255).astype(np.uint8)

            # Resize raw prediction to match image size
            if raw_vis.shape != gt_vis.shape:
                import cv2
                raw_vis = cv2.resize(raw_vis, (gt_vis.shape[1], gt_vis.shape[0]))
                raw_for_proc = cv2.resize(
                    logits[i].numpy(), (gt_vis.shape[1], gt_vis.shape[0])
                )
            else:
                raw_for_proc = logits[i].numpy()

            proc = postprocess_mask(raw_for_proc, config.postprocess)

            images.append(img)
            gts.append(gt_vis)
            raw_preds.append(raw_vis)
            proc_preds.append(proc)
            prompts_list.append(batch_prompts[i] if isinstance(batch_prompts, list) else batch_prompts)
            collected += 1

        if collected >= n_samples:
            break

    if not images:
        print("[visualize] No samples to visualize.")
        return

    # Build grid
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "Raw Prediction", "Processed Prediction"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_ylabel(f'"{prompts_list[i]}"', fontsize=10, rotation=0, labelpad=80)
        axes[i, 1].imshow(gts[i], cmap="gray", vmin=0, vmax=255)
        axes[i, 2].imshow(raw_preds[i], cmap="hot", vmin=0, vmax=255)
        axes[i, 3].imshow(proc_preds[i], cmap="gray", vmin=0, vmax=255)

        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle(
        "Drywall QA — Segmentation Predictions",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    save_path = output_dir / "prediction_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Prediction grid saved to {save_path}")


def generate_visualizations(
    results: dict,
    config: DrywallQAConfig,
):
    """Generate all evaluation visualizations.

    Args:
        results: Evaluation results dict.
        config: Project configuration.
    """
    output_dir = Path(config.project_root) / config.output.dir
    log_path = output_dir / config.output.log_file

    # Loss curves
    if log_path.exists():
        plot_loss_curves(log_path, output_dir)

    # Per-category metrics
    plot_category_metrics(results, output_dir)

    print(f"\n[evaluate] All visualizations saved to: {output_dir}")
