"""
Improve rock-only detection by cleaning labels and adding negative training frames.

Steps:
  1. Remove oversized / loose training boxes (often sand/terrain mislabeled as rock)
  2. Extract likely false-positive video frames and add as NEGATIVE samples (empty labels)
  3. Optionally fine-tune the model on the improved dataset

Usage:
  python scripts/improve_dataset.py
  python scripts/improve_dataset.py --retrain --epochs 40
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "mars_rocks"
DEFAULT_VIDEO = PROJECT_ROOT / "Videos" / "archive" / "06_terrain_negative_test.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
REPORT_PATH = PROJECT_ROOT / "data" / "dataset_improvement_report.json"

# Annotation bbox area above this fraction is treated as terrain-like (removed).
MAX_LABEL_AREA_RATIO = 0.18


def parse_annotation_bbox(parts: list[str], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Return pixel bbox from YOLO bbox or segmentation polygon line."""
    if len(parts) < 5:
        return None

    if len(parts) == 5:
        _, xc, yc, bw, bh = map(float, parts)
        x1 = int((xc - bw / 2) * img_w)
        y1 = int((yc - bh / 2) * img_h)
        x2 = int((xc + bw / 2) * img_w)
        y2 = int((yc + bh / 2) * img_h)
        return x1, y1, x2, y2

    # Segmentation polygon: class + normalized x,y pairs
    coords = list(map(float, parts[1:]))
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def clean_oversized_labels(dataset_dir: Path, max_area_ratio: float) -> dict:
    """Drop training annotations whose bounding rect covers too much of the image."""
    stats = {"files_checked": 0, "annotations_removed": 0, "files_emptied": 0}

    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            kept: list[str] = []
            stats["files_checked"] += 1

            for line in lines:
                parts = line.split()
                bbox = parse_annotation_bbox(parts, w, h)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                area_ratio = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
                if area_ratio <= max_area_ratio:
                    kept.append(line)
                else:
                    stats["annotations_removed"] += 1

            if not kept and lines:
                stats["files_emptied"] += 1
            lbl_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    return stats


def frame_has_suspicious_detections(boxes, img_w: int, img_h: int) -> bool:
    """Heuristic: frame likely contains false positives worth adding as negative."""
    if boxes is None or len(boxes) == 0:
        return False

    low_conf = 0
    large_boxes = 0
    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(img_w * img_h, 1)
        if conf < 0.55:
            low_conf += 1
        if area_ratio > 0.10 and conf < 0.70:
            large_boxes += 1

    if len(boxes) >= 10:
        return True
    if low_conf >= 3:
        return True
    if large_boxes >= 2:
        return True
    return False


def add_negative_frames_from_video(
    video: Path,
    model_path: Path,
    dataset_dir: Path,
    max_negatives: int,
    sample_every: int,
    scan_conf: float,
) -> dict:
    """Save suspicious frames as negative training images (empty label files)."""
    train_img = dataset_dir / "train" / "images"
    train_lbl = dataset_dir / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    saved = 0
    frame_idx = 0
    added_files: list[str] = []

    while saved < max_negatives:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        results = model.predict(frame, conf=scan_conf, device=device, verbose=False)
        boxes = results[0].boxes

        if frame_has_suspicious_detections(boxes, w, h):
            name = f"neg_{video.stem}_f{frame_idx:05d}"
            img_out = train_img / f"{name}.jpg"
            lbl_out = train_lbl / f"{name}.txt"
            cv2.imwrite(str(img_out), frame)
            lbl_out.write_text("", encoding="utf-8")
            added_files.append(img_out.name)
            saved += 1

        frame_idx += 1

    cap.release()
    return {"negative_frames_added": saved, "files": added_files}


def fine_tune(model_path: Path, data_yaml: Path, epochs: int, batch: int) -> str:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="mars_rock_finetune",
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )
    best = PROJECT_ROOT / "runs" / "detect" / "mars_rock_finetune" / "weights" / "best.pt"
    export = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
    if best.exists():
        shutil.copy2(best, export)
    return str(export)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean labels and add negative frames")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-negatives", type=int, default=35)
    parser.add_argument("--sample-every", type=int, default=12, help="Sample every N frames")
    parser.add_argument("--scan-conf", type=float, default=0.30, help="Low conf to find false positives")
    parser.add_argument("--retrain", action="store_true", help="Fine-tune after dataset improvements")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError("Dataset missing. Run: python scripts/download_dataset.py")

    print("Step 1/3: Cleaning oversized training labels ...")
    clean_stats = clean_oversized_labels(DATASET_DIR, MAX_LABEL_AREA_RATIO)
    print(f"  Removed {clean_stats['annotations_removed']} loose/terrain-like boxes")

    neg_stats = {"negative_frames_added": 0, "files": []}
    if args.video.exists() and args.model.exists():
        print("Step 2/3: Adding negative frames from suspicious detections ...")
        neg_stats = add_negative_frames_from_video(
            args.video,
            args.model,
            DATASET_DIR,
            args.max_negatives,
            args.sample_every,
            args.scan_conf,
        )
        print(f"  Added {neg_stats['negative_frames_added']} negative training frames")
    else:
        print("Step 2/3: Skipped negative extraction (video or model missing)")

    export_path = str(args.model)
    if args.retrain:
        print("Step 3/3: Fine-tuning model on improved dataset ...")
        data_yaml = DATASET_DIR / "data.yaml"
        export_path = fine_tune(args.model, data_yaml, args.epochs, args.batch)
        print(f"  Updated model: {export_path}")
    else:
        print("Step 3/3: Skipped retrain (pass --retrain to fine-tune)")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clean_labels": clean_stats,
        "negative_frames": neg_stats,
        "retrained": args.retrain,
        "model": export_path,
        "tips": [
            "Negative frames have empty .txt label files — model learns 'no rock here'.",
            "Review data/mars_rocks/train/images/neg_* and delete any that contain real rocks.",
            "Re-run: python scripts/detect_rocks.py --video Videos/samples/01_mars_rover_rocks.mp4",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    print("Next: python scripts/detect_rocks.py")


if __name__ == "__main__":
    main()
