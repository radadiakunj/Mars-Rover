"""
Train YOLOv11 rock detector on Mars terrain dataset.

Produces:
  models/mars_rock_detector.pt   (copy of best weights)
  runs/detect/mars_rock_train/weights/best.pt

Usage:
  python scripts/train.py
  python scripts/train.py --epochs 50 --model yolo11n.pt --batch 8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "mars_rocks" / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO rock detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO weights (yolo11n.pt or yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (use 4 for yolo11m on 8GB GPU)")
    parser.add_argument("--device", default="", help="0 for GPU, cpu for CPU (auto-detect if empty)")
    parser.add_argument("--name", default="mars_rock_train", help="Run name under runs/detect/")
    parser.add_argument(
        "--export",
        default="mars_rock_detector.pt",
        help="Exported filename under models/ (e.g. mars_rock_detector_m.pt)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Fine-tune from --model weights instead of training from scratch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {args.data}\n"
            "Run first: python scripts/download_dataset.py"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and str(device) != "cpu":
        print(f"Using GPU: {torch.cuda.get_device_name(int(device))}")
    else:
        print("Using CPU (install CUDA torch: scripts/install_cuda_torch.ps1)")

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"Training on: {args.data}")
    train_kwargs = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        exist_ok=True,
        patience=25,
        save=True,
        plots=True,
        verbose=True,
        # Strong color/shape augmentations — rocks vary in hue and form.
        hsv_h=0.03,
        hsv_s=0.75,
        hsv_v=0.55,
        degrees=12.0,
        translate=0.12,
        scale=0.55,
        mixup=0.12,
        copy_paste=0.1,
    )
    if args.resume:
        train_kwargs["resume"] = True
    results = model.train(**train_kwargs)

    best_weights = PROJECT_ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    export_path = MODELS_DIR / args.export

    if best_weights.exists():
        shutil.copy2(best_weights, export_path)
        print(f"\nTraining complete.")
        print(f"  Best weights: {best_weights}")
        print(f"  Exported to:  {export_path}")
    else:
        print(f"\nTraining finished but best.pt not found at {best_weights}")
        print(f"Results: {results}")


if __name__ == "__main__":
    main()
