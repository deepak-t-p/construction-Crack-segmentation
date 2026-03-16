"""
Dataset module for the Drywall QA Prompted Segmentation project.

Handles loading of crack and drywall-joint images with their VOC annotations,
converting polygon and bounding-box annotations into binary masks, and pairing
each sample with a randomly selected text prompt from its category.

Datasets (download in Pascal-VOC format from Roboflow):
  - Cracks:         https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36
  - Drywall Joints: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect
"""

from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import CLIPSegProcessor

from .config import DrywallQAConfig


# ────────────────────────────────────────────────────────────────
#  VOC Annotation Parsing
# ────────────────────────────────────────────────────────────────

def _parse_voc_xml(xml_path: str | Path) -> dict:
    """Parse a Pascal-VOC XML annotation file.

    Returns dict with keys:
        - ``size``: (width, height)
        - ``objects``: list of dicts with ``name``, ``bbox``, ``polygon``
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    width = int(size_el.find("width").text)
    height = int(size_el.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text

        # Try polygon first
        polygon = None
        poly_el = obj.find("polygon")
        if poly_el is not None:
            points = []
            for pt in poly_el:
                if pt.tag.startswith("x"):
                    idx = pt.tag[1:]  # e.g. "1" from "x1"
                    x = float(pt.text)
                    y_el = poly_el.find(f"y{idx}")
                    if y_el is not None:
                        y = float(y_el.text)
                        points.append((x, y))
            if points:
                polygon = points

        # Bounding box
        bbox = None
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            bbox = (
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            )

        objects.append({"name": name, "bbox": bbox, "polygon": polygon})

    return {"size": (width, height), "objects": objects}


def _annotation_to_mask(
    annotation: dict,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Convert VOC annotation objects into a single binary mask.

    Uses polygon annotations if available; otherwise falls back to bboxes.

    Args:
        annotation: Output of :func:`_parse_voc_xml`.
        target_size: (width, height) of the output mask.

    Returns:
        Binary mask as a uint8 numpy array of shape (H, W).
    """
    w, h = annotation["size"]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for obj in annotation["objects"]:
        if obj["polygon"] is not None:
            pts = [(int(x), int(y)) for x, y in obj["polygon"]]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        elif obj["bbox"] is not None:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            draw.rectangle([x1, y1, x2, y2], fill=255)

    # Resize to target
    mask = mask.resize(target_size, Image.NEAREST)
    return np.array(mask, dtype=np.uint8)


# ────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ────────────────────────────────────────────────────────────────

class DrywallQADataset(Dataset):
    """Dataset for Drywall QA segmentation.

    Each sample returns an image, a binary mask, and a text prompt.

    Expects a directory laid out as::

        root/
        ├── train/        (or root itself if no split dirs)
        │   ├── image1.jpg
        │   ├── image1.xml
        │   ├── image2.jpg
        │   └── image2.xml
        └── valid/
            └── ...

    Args:
        root_dir: Path to the dataset root (e.g. ``data/cracks``).
        category: ``"crack"`` or ``"joint"``.
        prompts: List of text prompts for this category.
        processor: HuggingFace CLIPSegProcessor.
        image_size: Target image dimension.
        split: ``"train"`` or ``"valid"`` — selects subdirectory if it exists.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root_dir: str | Path,
        category: str,
        prompts: List[str],
        processor: CLIPSegProcessor,
        image_size: int = 352,
        split: str = "train",
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.prompts = prompts
        self.processor = processor
        self.image_size = image_size
        self.split = split

        # Locate image/annotation pairs
        self.samples: List[Tuple[Path, Path]] = []
        self._discover_samples()

    def _discover_samples(self):
        """Find all (image, xml) pairs in the dataset directory."""
        # Check for split subdirectory
        split_dir = self.root_dir / self.split
        if split_dir.is_dir():
            search_dir = split_dir
        else:
            search_dir = self.root_dir

        if not search_dir.exists():
            print(f"[WARNING] Dataset directory not found: {search_dir}")
            return

        for img_file in sorted(search_dir.iterdir()):
            if img_file.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            xml_file = img_file.with_suffix(".xml")
            if xml_file.exists():
                self.samples.append((img_file, xml_file))

        print(f"  [{self.category}/{self.split}] Found {len(self.samples)} samples in {search_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, xml_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Parse annotation → binary mask
        annotation = _parse_voc_xml(xml_path)
        mask = _annotation_to_mask(annotation, (self.image_size, self.image_size))
        mask_tensor = torch.from_numpy(mask).float() / 255.0  # 0.0 or 1.0

        # Select a random text prompt
        prompt = random.choice(self.prompts)

        # Process image + text via CLIPSeg processor
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

        # Squeeze batch dim (processor returns [1, ...])
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "mask": mask_tensor,
            "category": self.category,
            "prompt": prompt,
        }


# ────────────────────────────────────────────────────────────────
#  DataLoader Factory
# ────────────────────────────────────────────────────────────────

def get_dataloaders(
    config: DrywallQAConfig,
    processor: CLIPSegProcessor,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Combines the crack and drywall-joint datasets into a single
    :class:`ConcatDataset`, then splits into train/val.

    Args:
        config: Project configuration.
        processor: CLIPSeg processor for image+text encoding.

    Returns:
        ``(train_loader, val_loader)``
    """
    root = Path(config.project_root)
    prompts = config.dataset.prompts
    img_size = config.model.image_size

    datasets_train: List[Dataset] = []
    datasets_val: List[Dataset] = []

    # ── Crack Dataset ──
    cracks_dir = root / config.dataset.cracks_dir
    if cracks_dir.exists():
        ds_train = DrywallQADataset(
            cracks_dir, "crack", prompts.get("crack", ["crack"]),
            processor, img_size, split="train",
        )
        ds_val = DrywallQADataset(
            cracks_dir, "crack", prompts.get("crack", ["crack"]),
            processor, img_size, split="valid",
        )
        if len(ds_train) > 0:
            datasets_train.append(ds_train)
        if len(ds_val) > 0:
            datasets_val.append(ds_val)
    else:
        print(f"[WARNING] Cracks dataset not found at {cracks_dir}")

    # ── Drywall Joint Dataset ──
    joints_dir = root / config.dataset.joints_dir
    if joints_dir.exists():
        ds_train = DrywallQADataset(
            joints_dir, "joint", prompts.get("joint", ["drywall joint"]),
            processor, img_size, split="train",
        )
        ds_val = DrywallQADataset(
            joints_dir, "joint", prompts.get("joint", ["drywall joint"]),
            processor, img_size, split="valid",
        )
        if len(ds_train) > 0:
            datasets_train.append(ds_train)
        if len(ds_val) > 0:
            datasets_val.append(ds_val)
    else:
        print(f"[WARNING] Drywall-joint dataset not found at {joints_dir}")

    if not datasets_train:
        raise FileNotFoundError(
            "No dataset found! Download and place them in:\n"
            f"  Cracks:  {root / config.dataset.cracks_dir}\n"
            f"  Joints:  {root / config.dataset.joints_dir}\n"
            "Download from:\n"
            "  https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36\n"
            "  https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect"
        )

    # If no validation split directory exists, auto-split from train
    if not datasets_val and datasets_train:
        combined = ConcatDataset(datasets_train)
        n_total = len(combined)
        n_train = int(n_total * config.dataset.train_split)
        n_val = n_total - n_train
        train_set, val_set = random_split(
            combined, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.training.seed),
        )
    else:
        train_set = ConcatDataset(datasets_train)
        val_set = ConcatDataset(datasets_val) if datasets_val else None

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
        )

    return train_loader, val_loader
