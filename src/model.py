"""
Model utilities for the Drywall QA Prompted Segmentation project.

Handles CLIPSeg model loading, optional encoder freezing, and
optimiser / scheduler construction with differential learning rates.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from .config import DrywallQAConfig


def build_model(config: DrywallQAConfig) -> CLIPSegForImageSegmentation:
    """Download / load the CLIPSeg checkpoint and optionally freeze the encoder.

    Args:
        config: Project configuration.

    Returns:
        A :class:`CLIPSegForImageSegmentation` model ready for training.
    """
    model = CLIPSegForImageSegmentation.from_pretrained(config.model.name)

    if config.model.freeze_encoder:
        # Freeze CLIP backbone entirely
        for param in model.clip.parameters():
            param.requires_grad = False
        print("[model] CLIP encoder frozen — only decoder will be trained.")
    else:
        print("[model] Full model (encoder + decoder) will be fine-tuned with differential LR.")

    return model


def load_processor(config: DrywallQAConfig) -> CLIPSegProcessor:
    """Load the CLIPSeg processor (tokenizer + image preprocessor).

    Args:
        config: Project configuration.

    Returns:
        A :class:`CLIPSegProcessor` instance.
    """
    return CLIPSegProcessor.from_pretrained(config.model.name)


def get_optimizer(
    model: CLIPSegForImageSegmentation,
    config: DrywallQAConfig,
) -> AdamW:
    """Construct an AdamW optimiser with differential learning rates.

    The CLIP encoder parameters receive a much lower learning rate than
    the segmentation decoder, preventing catastrophic forgetting of the
    pre-trained representations while still allowing fine-tuning.

    Args:
        model: The CLIPSeg model.
        config: Project configuration.

    Returns:
        An :class:`AdamW` optimiser.
    """
    # Separate encoder and decoder parameters
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("clip."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = [
        {
            "params": encoder_params,
            "lr": config.training.encoder_lr,
            "name": "encoder",
        },
        {
            "params": decoder_params,
            "lr": config.training.decoder_lr,
            "name": "decoder",
        },
    ]

    optimizer = AdamW(param_groups, weight_decay=config.training.weight_decay)

    n_enc = sum(p.numel() for p in encoder_params)
    n_dec = sum(p.numel() for p in decoder_params)
    print(
        f"[optimizer] Encoder params: {n_enc:,} (lr={config.training.encoder_lr}), "
        f"Decoder params: {n_dec:,} (lr={config.training.decoder_lr})"
    )

    return optimizer


def get_scheduler(
    optimizer: AdamW,
    config: DrywallQAConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a learning-rate scheduler.

    Uses cosine annealing with optional linear warm-up.

    Args:
        optimizer: The optimiser.
        config: Project configuration.
        total_steps: Total training steps across all epochs.

    Returns:
        A learning-rate scheduler.
    """
    warmup = config.training.warmup_steps

    if config.training.scheduler == "cosine":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + __import__("math").cos(3.14159265 * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    return scheduler
