"""
Centralized configuration for the Drywall QA Prompted Segmentation project.

Loads hyperparameters from a YAML file and exposes them as a typed dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class ModelConfig:
    name: str = "CIDAS/clipseg-rd64-refined"
    image_size: int = 352
    freeze_encoder: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    encoder_lr: float = 1e-6
    decoder_lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_steps: int = 100
    val_interval_steps: int = 170
    grad_accumulation_steps: int = 1
    seed: int = 42


@dataclass
class DatasetConfig:
    cracks_dir: str = "data/cracks"
    joints_dir: str = "data/drywall-joints"
    train_split: float = 0.9
    prompts: Dict[str, List[str]] = field(default_factory=lambda: {
        "crack": [
            "crack", "wall crack", "structural crack",
            "surface crack", "crack on wall",
        ],
        "joint": [
            "drywall joint", "drywall seam", "taping area",
            "wall joint", "drywall tape joint",
        ],
    })


@dataclass
class PostprocessConfig:
    threshold: float = 0.5
    morph_kernel_size: int = 5
    min_component_area: int = 100
    use_closing: bool = True
    use_opening: bool = True


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_best_only: bool = True
    log_file: str = "training_log.csv"


@dataclass
class DrywallQAConfig:
    """Top-level configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Resolved absolute project root (set at load time)
    project_root: str = ""


def _merge_dict_into_dataclass(dc, data: dict):
    """Recursively overwrite dataclass fields from a dict."""
    for key, val in data.items():
        if hasattr(dc, key):
            attr = getattr(dc, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(val, dict):
                _merge_dict_into_dataclass(attr, val)
            else:
                setattr(dc, key, val)


def load_config(config_path: str | Path | None = None) -> DrywallQAConfig:
    """Load config from a YAML file, falling back to defaults.

    Args:
        config_path: Path to a YAML config file. If *None*, uses
            ``configs/default.yaml`` relative to the project root.

    Returns:
        A fully-populated :class:`DrywallQAConfig` instance.
    """
    cfg = DrywallQAConfig()

    # Determine project root (directory containing pyproject.toml)
    if config_path is not None:
        project_root = Path(config_path).resolve().parent.parent
    else:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "default.yaml"

    cfg.project_root = str(project_root)

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _merge_dict_into_dataclass(cfg, raw)

    # Ensure output directory exists
    out_dir = Path(cfg.project_root) / cfg.output.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    return cfg
