# Improve Model Accuracy — Best Model

## Current Baseline
- **Best mIoU: 0.489** | Dice: 0.635 | Accuracy: 0.939
- Crack mIoU: 0.449 | Joint mIoU: 0.528
- 10 epochs, BCE loss, no augmentation, no class balancing

## Key Bottlenecks Identified

1. **No data augmentation** — model sees exact same images every epoch, easy to overfit
2. **Massive class imbalance** — 5164 crack vs 820 joint samples (6:1), joint category undertrained
3. **LR decays too fast** — cosine to zero by epoch 10, model stops learning by epoch 7
4. **BCE-only loss** — doesn't directly optimize IoU (already adding Dice+BCE)

## Proposed Changes

---

### 1. Data Augmentation — [dataset.py](file:///d:/Deepak/projects/origin/src/dataset.py)

Add training-time augmentations that are applied **before** CLIPSeg processing:
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Color jitter (brightness, contrast, saturation)
- Random rotation (±15°)

Augmentations apply to **both image and mask** simultaneously for spatial transforms.

### 2. Class-Balanced Sampling — [dataset.py](file:///d:/Deepak/projects/origin/src/dataset.py)

Use `WeightedRandomSampler` to oversample the minority class (joints) so each batch has ~equal representation.

### 3. Training Config — [default.yaml](file:///d:/Deepak/projects/origin/configs/default.yaml)

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| epochs | 10 | 25 | More time to converge |
| encoder_lr | 1e-6 | 5e-6 | Slightly more encoder adaptation |
| decoder_lr | 1e-4 | 3e-4 | More aggressive decoder training |
| warmup_steps | 100 | 200 | Smoother start with higher LR |
| val_interval | 170 | 250 | Less frequent val with more epochs |
| grad_accumulation | 1 | 2 | Effective batch 32 for stability |

### 4. Dice+BCE Loss — Already implemented in [train.py](file:///d:/Deepak/projects/origin/src/train.py)

## Verification

After retraining, compare new vs old:
- mIoU (target: >0.55)
- Per-category breakdown (crack and joint)
