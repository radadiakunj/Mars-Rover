"""
Expand training data for a more generalized rock detector.

Adds:
  - NASA Mars + desert/geology images as hard-negative samples (empty labels)
  - Extra labeled rock images shuffled into train split

Usage:
  python scripts/expand_generalized_dataset.py
  python scripts/expand_generalized_dataset.py --retrain --epochs 60
"""

from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "mars_rocks"
MODEL_PATH = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
STAGING = PROJECT_ROOT / "data" / "generalized_staging"


def fetch_nasa_images(query: str, count: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {"q": query, "media_type": "image", "page_size": min(count, 100)}
    items = requests.get(
        "https://images-api.nasa.gov/search", params=params, timeout=60
    ).json().get("collection", {}).get("items", [])

    saved = 0
    for item in items:
        if saved >= count:
            break
        try:
            meta = requests.get(item["href"], timeout=30).json()
            urls = []
            if isinstance(meta, list) and meta and isinstance(meta[0], str):
                urls = [u for u in meta if u.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            if not urls:
                continue
            ext = Path(urls[0].split("?")[0]).suffix or ".jpg"
            dest = output_dir / f"gen_{query[:12].replace(' ', '_')}_{saved:03d}{ext}"
            dest.write_bytes(requests.get(urls[0], timeout=60).content)
            saved += 1
        except Exception:
            continue
    return saved


def add_negative_images(image_dir: Path, dataset_dir: Path, prefix: str) -> int:
    train_img = dataset_dir / "train" / "images"
    train_lbl = dataset_dir / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    added = 0
    for img in sorted(image_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        name = f"{prefix}_{added:04d}{img.suffix.lower()}"
        shutil.copy2(img, train_img / name)
        (train_lbl / f"{prefix}_{added:04d}.txt").write_text("", encoding="utf-8")
        added += 1
    return added


def refresh_data_yaml(dataset_dir: Path) -> None:
    yaml_path = dataset_dir / "data.yaml"
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "rock"},
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def retrain(model_path: Path, data_yaml: Path, epochs: int, batch: int) -> Path:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="mars_rock_generalized",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.15,
        scale=0.6,
        mosaic=1.0,
        mixup=0.1,
        verbose=True,
    )
    best = PROJECT_ROOT / "runs" / "detect" / "mars_rock_generalized" / "weights" / "best.pt"
    export = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
    if best.exists():
        shutil.copy2(best, export)
    return export


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand dataset for generalized rock detection")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    STAGING.mkdir(parents=True, exist_ok=True)
    queries = [
        ("Mars surface rocks boulders terrain", 40),
        ("desert rocks geology field", 40),
        ("volcanic rocks terrain geology", 25),
        ("Mars rover Curiosity rocks", 25),
    ]

    total_neg = 0
    for i, (query, count) in enumerate(queries):
        sub = STAGING / f"batch_{i}"
        n = fetch_nasa_images(query, count, sub)
        total_neg += add_negative_images(sub, DATASET_DIR, f"genneg_{i}")
        print(f"  [{query}] -> {n} images")

    refresh_data_yaml(DATASET_DIR)
    print(f"\nAdded {total_neg} generalized negative training images.")

    if args.retrain and MODEL_PATH.exists():
        print(f"\nRetraining for {args.epochs} epochs ...")
        out = retrain(MODEL_PATH, DATASET_DIR / "data.yaml", args.epochs, args.batch)
        print(f"Updated model: {out}")
    elif args.retrain:
        print("Model not found — run training first.")

    report = PROJECT_ROOT / "data" / "generalized_dataset_report.txt"
    report.write_text(
        f"Expanded at {datetime.now(timezone.utc).isoformat()}\n"
        f"Negative images added: {total_neg}\n"
        f"Retrained: {args.retrain}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
