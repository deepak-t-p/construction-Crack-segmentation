# 🏗️ Prompted Segmentation for Drywall QA

**Automated defect detection in drywall construction using text-prompted segmentation.**

This project fine-tunes [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) to detect and segment **cracks** and **drywall joints** from construction images using natural language prompts — no retraining needed to inspect different defect types.

---

## 🎯 Problem Statement

Manual quality assurance in drywall construction is time-consuming and error-prone. This system automates defect detection by allowing users to describe what to look for in plain English:

- `"wall crack"` → segments cracks on walls
- `"taping area"` → highlights drywall seams requiring tape
- `"structural crack"` → finds deep structural damage

## 🏛️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Input Image │────►│   CLIPSeg     │◄────│ Text Prompt  │
│  (352×352)   │     │  (Encoder +   │     │ "wall crack" │
└──────────────┘     │   Decoder)    │     └──────────────┘
                     └──────┬────────┘
                            │ raw logits
                     ┌──────▼────────┐
                     │ Post-Process  │
                     │ (threshold +  │
                     │  morphology)  │
                     └──────┬────────┘
                            │
                     ┌──────▼────────┐
                     │ Binary Mask   │
                     │ + Overlay     │
                     └───────────────┘
```

**Model:** CLIPSeg (`CIDAS/clipseg-rd64-refined`) — 150M params, combines CLIP text/image embeddings with a segmentation decoder.

## 📁 Project Structure

```
origin/
├── configs/
│   └── default.yaml          # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration loader
│   ├── dataset.py            # VOC XML parsing + DataLoader
│   ├── model.py              # CLIPSeg loading + differential LR
│   ├── train.py              # Training loop
│   ├── metrics.py            # mIoU, Dice, Pixel Accuracy
│   ├── process.py            # Morphological post-processing
│   ├── evaluate.py           # Evaluation + visualisations
│   └── inference.py          # Single-image inference
├── data/
│   ├── cracks/               # Crack dataset (VOC format)
│   └── drywall-joints/       # Joint dataset (VOC format)
├── outputs/                  # Checkpoints, logs, plots
├── train_model.py            # CLI: training
├── run_inference.py          # CLI: inference
├── pyproject.toml            # Dependencies
└── README.md
```

## 🚀 Setup

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Download datasets

Download both datasets in **Pascal-VOC format** from Roboflow:

| Dataset | URL | Place in |
|---------|-----|----------|
| Cracks | [cracks-3ii36](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) | `data/cracks/` |
| Drywall Joints | [drywall-join-detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | `data/drywall-joints/` |

Each dataset folder should contain `train/` and `valid/` subdirectories with `.jpg` + `.xml` file pairs.

### 3. Train

```bash
# Default config (10 epochs, batch 16)
python train_model.py

# Custom settings
python train_model.py --epochs 5 --batch-size 8

# Freeze encoder (faster, less GPU memory)
python train_model.py --freeze-encoder --batch-size 32
```

### 4. Inference

```bash
# With trained checkpoint
python run_inference.py \
    --image path/to/wall_photo.jpg \
    --prompt "wall crack" \
    --checkpoint outputs/best_model.pt

# Without checkpoint (uses pre-trained CLIPSeg)
python run_inference.py \
    --image path/to/wall_photo.jpg \
    --prompt "taping area"
```

## ⚙️ Key Design Decisions

| Feature | Details |
|---------|---------|
| **Differential LR** | Encoder: `1e-6`, Decoder: `1e-4` — prevents forgetting CLIP features |
| **Loss function** | `BCEWithLogitsLoss` — binary cross-entropy for mask prediction |
| **Post-processing** | Morphological open/close → small-component removal |
| **Text augmentation** | 7+ prompts per category, randomly sampled during training |
| **Checkpointing** | Best model saved by validation mIoU |

## 📊 Evaluation Metrics

- **mIoU** (Mean Intersection over Union)
- **mDice** (Dice Coefficient / F1 Score)
- **Pixel Accuracy**

After training, visualisations are saved in `outputs/`:
- `loss_curves.png` — train vs. validation loss
- `category_metrics.png` — per-category mIoU and Dice
- `prediction_grid.png` — side-by-side: Input | GT | Raw | Processed

## 🔧 Configuration

All hyperparameters live in `configs/default.yaml`. Key options:

```yaml
training:
  epochs: 10
  batch_size: 16          # reduce for low-VRAM GPUs
  encoder_lr: 1.0e-6
  decoder_lr: 1.0e-4

postprocess:
  threshold: 0.5
  morph_kernel_size: 5
  min_component_area: 100
```

demo :  https://construction-wall-inspection.streamlit.app/


## 📜 License

This project is for educational and research purposes.
