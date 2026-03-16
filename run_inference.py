"""
CLI entry point for single-image prompted segmentation inference.

Usage:
    python run_inference.py --image photo.jpg --prompt "wall crack"
    python run_inference.py --image photo.jpg --prompt "taping area" --checkpoint outputs/best_model.pt
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.evaluate import load_trained_model
from src.inference import run_inference, save_results
from src.model import build_model, load_processor


def main():
    parser = argparse.ArgumentParser(
        description="Run prompted segmentation inference on a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --image wall_photo.jpg --prompt "wall crack"
  python run_inference.py --image site.png --prompt "taping area" --checkpoint outputs/best_model.pt
  python run_inference.py --image img.jpg --prompt "drywall seam" --output results/
        """,
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help='Text prompt describing what to segment (e.g. "wall crack", "taping area")',
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint (.pt). If not provided, uses the pre-trained CLIPSeg model.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results (default: outputs/inference/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override binarisation threshold (default: from config)",
    )

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = load_config(args.config)
    if args.threshold is not None:
        config.postprocess.threshold = args.threshold

    # Load model
    processor = load_processor(config)
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        model = load_trained_model(ckpt_path, config)
    else:
        print("[INFO] No checkpoint provided — using pre-trained CLIPSeg model.")
        model = build_model(config)

    # Run inference
    print(f'\n[inference] Image: {image_path}')
    print(f'[inference] Prompt: "{args.prompt}"')

    image, raw_mask, processed_mask, overlay = run_inference(
        str(image_path), args.prompt, model, processor, config,
    )

    # Save results
    output_dir = args.output or str(Path(config.project_root) / config.output.dir / "inference")
    prompt_slug = args.prompt.replace(" ", "_").lower()[:30]
    prefix = f"{image_path.stem}_{prompt_slug}"

    save_results(image, raw_mask, processed_mask, overlay, output_dir, prefix)

    print(f"\n[Done] Check results in: {output_dir}/")


if __name__ == "__main__":
    main()
