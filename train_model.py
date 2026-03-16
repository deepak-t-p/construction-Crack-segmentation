"""
CLI entry point for training the Drywall QA segmentation model.

Usage:
    python train_model.py --config configs/default.yaml
    python train_model.py --epochs 5 --batch-size 8
"""

import argparse
import sys

from src.config import load_config
from src.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train CLIPSeg for Drywall QA Prompted Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py
  python train_model.py --config configs/default.yaml
  python train_model.py --epochs 5 --batch-size 8 --encoder-lr 1e-6
  python train_model.py --freeze-encoder
        """,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--encoder-lr", type=float, default=None, help="Override encoder LR")
    parser.add_argument("--decoder-lr", type=float, default=None, help="Override decoder LR")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--freeze-encoder", action="store_true",
        help="Freeze CLIP encoder (train decoder only)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.encoder_lr is not None:
        config.training.encoder_lr = args.encoder_lr
    if args.decoder_lr is not None:
        config.training.decoder_lr = args.decoder_lr
    if args.output_dir is not None:
        config.output.dir = args.output_dir
    if args.freeze_encoder:
        config.model.freeze_encoder = True
    if args.seed is not None:
        config.training.seed = args.seed

    # Print config summary
    print("=" * 60)
    print("  Drywall QA — Prompted Segmentation Training")
    print("=" * 60)
    print(f"  Model:       {config.model.name}")
    print(f"  Epochs:      {config.training.epochs}")
    print(f"  Batch size:  {config.training.batch_size}")
    print(f"  Encoder LR:  {config.training.encoder_lr}")
    print(f"  Decoder LR:  {config.training.decoder_lr}")
    print(f"  Image size:  {config.model.image_size}")
    print(f"  Output dir:  {config.output.dir}")
    print("=" * 60)

    # Train
    try:
        train(config=config)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        print(
            "\nPlease download the datasets and place them in the data/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
