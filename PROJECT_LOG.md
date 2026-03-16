# Project Update Log

This file tracks all changes and steps taken to implement the Construction Crack Segmentation project.

## [2026-03-16] Streamlit App & GPU Setup
- Set up PyTorch with CUDA support for RTX 4060.
- Implemented `DiceBCELoss` (Dice + BCE) for better segmentation of thin structures like cracks.
- Added mixed-precision training (`torch.amp.GradScaler`) which speeds up training by ~2x on the RTX 4060.
- Built an interactive Streamlit inference app (`streamlit_app.py`) allowing users to upload images, try different prompts, and adjust post-processing parameters via UI.
- Fixed OpenCV (`cv2`) dependency issues in the Streamlit app by replacing `cv2` operations with `PIL` and `matplotlib` equivalents.

## Next Steps Planned
- Implement data augmentation (horizontal/vertical flips, color jitter, rotation) to prevent overfitting.
- Apply Class-Balanced Sampling to address the 6:1 class imbalance (Cracks vs. Joints).
- Optimize training config for RTX 4060 (larger batch size if possible via gradient accumulation, more epochs, tuned learning rates).
