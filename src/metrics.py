"""
Segmentation metrics for the Drywall QA project.

Provides mIoU, Dice coefficient, and Pixel Accuracy — the three
evaluation criteria specified in the problem statement.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute Intersection over Union for a batch.

    Args:
        pred: Raw logits or probabilities, shape ``(B, H, W)``.
        target: Binary ground-truth masks, shape ``(B, H, W)``.
        threshold: Threshold to binarise predictions.

    Returns:
        IoU per sample, shape ``(B,)``.
    """
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    target_bin = (target >= 0.5).float()

    intersection = (pred_bin * target_bin).sum(dim=(-2, -1))
    union = pred_bin.sum(dim=(-2, -1)) + target_bin.sum(dim=(-2, -1)) - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute Dice coefficient (F1) for a batch.

    Args:
        pred: Raw logits or probabilities, shape ``(B, H, W)``.
        target: Binary ground-truth masks, shape ``(B, H, W)``.
        threshold: Threshold to binarise predictions.

    Returns:
        Dice score per sample, shape ``(B,)``.
    """
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    target_bin = (target >= 0.5).float()

    intersection = (pred_bin * target_bin).sum(dim=(-2, -1))
    denominator = pred_bin.sum(dim=(-2, -1)) + target_bin.sum(dim=(-2, -1))

    dice = (2.0 * intersection + 1e-7) / (denominator + 1e-7)
    return dice


def compute_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute pixel-level accuracy for a batch.

    Args:
        pred: Raw logits or probabilities, shape ``(B, H, W)``.
        target: Binary ground-truth masks, shape ``(B, H, W)``.
        threshold: Threshold to binarise predictions.

    Returns:
        Accuracy per sample, shape ``(B,)``.
    """
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    target_bin = (target >= 0.5).float()

    correct = (pred_bin == target_bin).float().sum(dim=(-2, -1))
    total = torch.tensor(target.shape[-2] * target.shape[-1], device=target.device).float()

    return correct / total


class MetricTracker:
    """Accumulates per-batch metrics and computes epoch-level means.

    Usage::

        tracker = MetricTracker()
        for batch in loader:
            tracker.update(logits, masks, categories)
        results = tracker.compute()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._iou: List[float] = []
        self._dice: List[float] = []
        self._acc: List[float] = []
        self._per_category: Dict[str, Dict[str, List[float]]] = {}

    def reset(self):
        """Clear all accumulated metrics."""
        self._iou.clear()
        self._dice.clear()
        self._acc.clear()
        self._per_category.clear()

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        categories: Optional[List[str]] = None,
    ):
        """Add a batch of predictions and targets.

        Args:
            pred: Logits, shape ``(B, H, W)``.
            target: Ground-truth masks, shape ``(B, H, W)``.
            categories: Optional list of category names per sample.
        """
        iou = compute_iou(pred, target, self.threshold)
        dice = compute_dice(pred, target, self.threshold)
        acc = compute_pixel_accuracy(pred, target, self.threshold)

        self._iou.extend(iou.cpu().tolist())
        self._dice.extend(dice.cpu().tolist())
        self._acc.extend(acc.cpu().tolist())

        if categories is not None:
            for i, cat in enumerate(categories):
                if cat not in self._per_category:
                    self._per_category[cat] = {"iou": [], "dice": [], "acc": []}
                self._per_category[cat]["iou"].append(iou[i].item())
                self._per_category[cat]["dice"].append(dice[i].item())
                self._per_category[cat]["acc"].append(acc[i].item())

    def compute(self) -> dict:
        """Compute mean metrics across all accumulated samples.

        Returns:
            Dict with ``mIoU``, ``mDice``, ``mAccuracy``, and
            per-category breakdowns.
        """
        def _mean(lst):
            return sum(lst) / max(len(lst), 1)

        results = {
            "mIoU": _mean(self._iou),
            "mDice": _mean(self._dice),
            "mAccuracy": _mean(self._acc),
            "n_samples": len(self._iou),
            "per_category": {},
        }

        for cat, metrics in self._per_category.items():
            results["per_category"][cat] = {
                "mIoU": _mean(metrics["iou"]),
                "mDice": _mean(metrics["dice"]),
                "mAccuracy": _mean(metrics["acc"]),
                "n_samples": len(metrics["iou"]),
            }

        return results
