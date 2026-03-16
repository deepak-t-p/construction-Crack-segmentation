"""
Training loop for the Drywall QA Prompted Segmentation project.

Manages the full training pipeline: forward/backward passes, gradient
accumulation, periodic validation, checkpoint saving, and loss logging.
Uses a combined Dice + BCE loss for improved segmentation accuracy,
especially on thin structures like cracks. Mixed-precision training
is enabled for faster training on modern GPUs.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from .config import DrywallQAConfig, load_config
from .dataset import get_dataloaders
from .metrics import MetricTracker
from .model import build_model, get_optimizer, get_scheduler, load_processor


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation.

    Dice loss directly optimises overlap (IoU-like), which helps with
    class-imbalanced masks (thin cracks, small joints). Combining with
    BCE preserves stable gradients early in training.

    Args:
        bce_weight: Weight for BCE component (default 0.5).
        dice_weight: Weight for Dice component (default 0.5).
        smooth: Smoothing factor to avoid division by zero.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE component
        bce_loss = self.bce(logits, targets)

        # Dice component
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class Trainer:
    """Encapsulates the full CLIPSeg training pipeline.

    Args:
        config: Project configuration.
    """

    def __init__(self, config: DrywallQAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        print(f"[trainer] Using device: {self.device}")
        if self.use_amp:
            print("[trainer] Mixed-precision training (AMP) enabled")

        # Seed
        torch.manual_seed(config.training.seed)

        # Model & processor
        self.processor = load_processor(config)
        self.model = build_model(config).to(self.device)

        # Data
        self.train_loader, self.val_loader = get_dataloaders(config, self.processor)

        # Optimizer & scheduler
        self.optimizer = get_optimizer(self.model, config)
        total_steps = (
            len(self.train_loader) * config.training.epochs
            // config.training.grad_accumulation_steps
        )
        self.scheduler = get_scheduler(self.optimizer, config, total_steps)

        # Loss — combined Dice + BCE for better segmentation accuracy
        self.criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
        print("[trainer] Using combined Dice + BCE loss")

        # Mixed-precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Tracking
        self.best_miou = 0.0
        self.global_step = 0
        self.output_dir = Path(config.project_root) / config.output.dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / config.output.log_file

        # Initialise CSV log
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step", "train_loss", "val_loss",
                "val_mIoU", "val_mDice", "val_mAccuracy", "lr_encoder", "lr_decoder",
            ])

    def train(self):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"  Starting training for {self.config.training.epochs} epochs")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"  Val samples:   {len(self.val_loader.dataset)}")
        print(f"  Batch size:    {self.config.training.batch_size}")
        print(f"  Device:        {self.device}")
        print(f"  Loss:          Dice + BCE (0.5 / 0.5)")
        print(f"  AMP:           {'enabled' if self.use_amp else 'disabled'}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.training.epochs + 1):
            self._train_one_epoch(epoch)

        print(f"\n[trainer] Training complete. Best mIoU: {self.best_miou:.4f}")
        print(f"[trainer] Outputs saved to: {self.output_dir}")

    def _train_one_epoch(self, epoch: int):
        """Train for a single epoch."""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.training.epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            loss = self._train_step(batch)
            epoch_loss += loss
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{epoch_loss / n_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Periodic validation
            if (
                self.val_loader is not None
                and self.global_step > 0
                and self.global_step % self.config.training.val_interval_steps == 0
            ):
                val_results = self._validate(epoch)
                self._log_step(epoch, epoch_loss / n_batches, val_results)
                self.model.train()

        # End-of-epoch validation
        if self.val_loader is not None:
            val_results = self._validate(epoch)
            self._log_step(epoch, epoch_loss / n_batches, val_results)
        else:
            self._log_step(epoch, epoch_loss / n_batches, None)

    def _train_step(self, batch: dict) -> float:
        """Execute a single training step with mixed-precision.

        Returns the scalar loss value.
        """
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        masks = batch["mask"].to(self.device)

        # Forward with AMP autocast
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

            logits = outputs.logits  # (B, H, W)

            # Resize mask to match logits if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(
                    masks.unsqueeze(1),
                    size=logits.shape[-2:],
                    mode="nearest",
                ).squeeze(1)

            loss = self.criterion(logits, masks)
            loss = loss / self.config.training.grad_accumulation_steps

        # Backward with scaler
        self.scaler.scale(loss).backward()

        # Optimizer step (with gradient accumulation)
        self.global_step += 1
        if self.global_step % self.config.training.grad_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * self.config.training.grad_accumulation_steps

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """Run validation and return metrics."""
        self.model.eval()
        tracker = MetricTracker(threshold=self.config.postprocess.threshold)
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="  Validating", leave=False):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            masks = batch["mask"].to(self.device)
            categories = batch.get("category", None)

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(
                    masks.unsqueeze(1),
                    size=logits.shape[-2:],
                    mode="nearest",
                ).squeeze(1)

            loss = self.criterion(logits, masks)
            total_loss += loss.item()
            n_batches += 1

            tracker.update(logits, masks, categories)

        results = tracker.compute()
        results["val_loss"] = total_loss / max(n_batches, 1)

        print(
            f"  [val] step={self.global_step} | "
            f"loss={results['val_loss']:.4f} | "
            f"mIoU={results['mIoU']:.4f} | "
            f"mDice={results['mDice']:.4f} | "
            f"mAcc={results['mAccuracy']:.4f}"
        )

        # Per-category breakdown
        for cat, cat_metrics in results.get("per_category", {}).items():
            print(
                f"    [{cat}] mIoU={cat_metrics['mIoU']:.4f} | "
                f"mDice={cat_metrics['mDice']:.4f} | "
                f"n={cat_metrics['n_samples']}"
            )

        # Save best checkpoint
        if results["mIoU"] > self.best_miou:
            self.best_miou = results["mIoU"]
            self._save_checkpoint(epoch, is_best=True)

        return results

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt_path = self.output_dir / (
            "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_miou": self.best_miou,
            "config": {
                "model_name": self.config.model.name,
                "image_size": self.config.model.image_size,
            },
        }, ckpt_path)
        tag = " (best)" if is_best else ""
        print(f"  [checkpoint] Saved{tag}: {ckpt_path}")

    def _log_step(self, epoch: int, train_loss: float, val_results: Optional[dict]):
        """Append a row to the CSV training log."""
        lrs = self.scheduler.get_last_lr()
        lr_enc = lrs[0] if len(lrs) > 0 else 0
        lr_dec = lrs[1] if len(lrs) > 1 else lrs[0]

        row = [
            epoch,
            self.global_step,
            f"{train_loss:.6f}",
            f"{val_results['val_loss']:.6f}" if val_results else "",
            f"{val_results['mIoU']:.6f}" if val_results else "",
            f"{val_results['mDice']:.6f}" if val_results else "",
            f"{val_results['mAccuracy']:.6f}" if val_results else "",
            f"{lr_enc:.2e}",
            f"{lr_dec:.2e}",
        ]

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def train(config: Optional[DrywallQAConfig] = None, config_path: Optional[str] = None):
    """Top-level training entry point.

    Args:
        config: A pre-constructed config. If not provided, loads from
            ``config_path``.
        config_path: Path to YAML config file.
    """
    if config is None:
        config = load_config(config_path)

    trainer = Trainer(config)
    trainer.train()
    return trainer
