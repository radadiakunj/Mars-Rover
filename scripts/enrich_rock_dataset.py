"""
Add diverse rock color/shape samples from videos and NASA images, then retrain.

Mines training data focusing on rock appearance (not sand):
  - Negative frames from terrain-only sample videos (empty labels)
  - Positive labels from model + color/shape validation
  - Extra positives from appearance-based segmentation (gray/dark rocks)

Usage:
  python scripts/enrich_rock_dataset.py
  python scripts/enrich_rock_dataset.py --retrain --epochs 80
  python scripts/enrich_rock_dataset.py --retrain --base-model runs/detect/mars_rock_train/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests
import torch
import yaml
from ultralytics import YOLO

from detection_filters import is_likely_rock_box
from rock_appearance import find_rock_boxes_by_appearance, to_yolo_line

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "mars_rocks"
SAMPLES_DIR = PROJECT_ROOT / "Videos" / "eval"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
STAGING = PROJECT_ROOT / "data" / "rock_diversity_staging"
REPORT_PATH = PROJECT_ROOT / "data" / "rock_enrichment_report.json"

NEGATIVE_VIDEO_STEMS: set[str] = set()  # negatives sourced from Videos/archive/ if needed
ROCK_VIDEO_STEMS = {
    "01_mars_rover_rocks",
    "05_rock_labeled_variety",
}

NASA_ROCK_QUERIES = [
    ("Mars rock boulder close-up", 30),
    ("Mars surface gray rock pebble", 25),
    ("volcanic rock desert geology", 25),
    ("Mars Curiosity rock sample", 20),
]


def fetch_diverse_images(query: str, count: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = requests.get(
        "https://images-api.nasa.gov/search",
        params={"q": query, "media_type": "image", "page_size": min(count, 100)},
        timeout=60,
    ).json().get("collection", {}).get("items", [])

    saved = 0
    for item in items:
        if saved >= count:
            break
        try:
            meta = requests.get(item["href"], timeout=30).json()
            urls: list[str] = []
            if isinstance(meta, list) and meta and isinstance(meta[0], str):
                urls = [u for u in meta if u.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            if not urls:
                continue
            ext = Path(urls[0].split("?")[0]).suffix or ".jpg"
            dest = output_dir / f"div_{query[:10].replace(' ', '_')}_{saved:03d}{ext}"
            dest.write_bytes(requests.get(urls[0], timeout=60).content)
            saved += 1
        except Exception:
            continue
    return saved


def add_negative_frame(frame, stem: str, frame_idx: int, train_img: Path, train_lbl: Path) -> str | None:
    name = f"neg_{stem}_f{frame_idx:05d}"
    if (train_img / f"{name}.jpg").exists():
        return None
    cv2.imwrite(str(train_img / f"{name}.jpg"), frame)
    (train_lbl / f"{name}.txt").write_text("", encoding="utf-8")
    return name


def add_positive_frame(
    frame,
    boxes: list[tuple[int, int, int, int]],
    stem: str,
    frame_idx: int,
    train_img: Path,
    train_lbl: Path,
) -> str | None:
    if not boxes:
        return None
    name = f"pos_{stem}_f{frame_idx:05d}"
    if (train_img / f"{name}.jpg").exists():
        return None
    h, w = frame.shape[:2]
    cv2.imwrite(str(train_img / f"{name}.jpg"), frame)
    lines = [to_yolo_line(x1, y1, x2, y2, w, h) for x1, y1, x2, y2 in boxes]
    (train_lbl / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name


def mine_from_video(
    video: Path,
    model: YOLO,
    *,
    is_negative: bool,
    sample_every: int,
    max_frames: int,
    scan_conf: float,
    train_img: Path,
    train_lbl: Path,
    device: str | int,
) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"video": video.name, "error": "cannot open"}

    stem = video.stem
    saved = 0
    frame_idx = 0
    stats = {"video": video.name, "negatives": 0, "positives": 0, "boxes": 0}

    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        if is_negative:
            if add_negative_frame(frame, stem, frame_idx, train_img, train_lbl):
                stats["negatives"] += 1
                saved += 1
        else:
            h, w = frame.shape[:2]
            boxes: list[tuple[int, int, int, int]] = []

            # Model proposals filtered by geometry + color/shape.
            results = model.predict(frame, conf=scan_conf, device=device, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    c = float(box.conf[0])
                    if is_likely_rock_box(
                        x1, y1, x2, y2, c, w, h, min_conf=scan_conf, img=frame, min_appearance=0.34
                    ):
                        boxes.append((x1, y1, x2, y2))

            # Appearance-only boxes for color/shape diversity (gray, dark, volcanic).
            for box in find_rock_boxes_by_appearance(frame, min_score=0.40):
                if box not in boxes:
                    boxes.append(box)

            if boxes:
                add_positive_frame(frame, boxes[:12], stem, frame_idx, train_img, train_lbl)
                stats["positives"] += 1
                stats["boxes"] += len(boxes[:12])
                saved += 1

        frame_idx += 1

    cap.release()
    return stats


def mine_from_staged_images(image_dir: Path, train_img: Path, train_lbl: Path, prefix: str) -> dict:
    added = 0
    boxes_total = 0
    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        boxes = find_rock_boxes_by_appearance(frame, min_score=0.36)
        if not boxes:
            continue
        name = f"{prefix}_{added:04d}"
        shutil.copy2(img_path, train_img / f"{name}{img_path.suffix.lower()}")
        lines = [to_yolo_line(x1, y1, x2, y2, w, h) for x1, y1, x2, y2 in boxes[:15]]
        (train_lbl / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        added += 1
        boxes_total += len(lines)
    return {"images": added, "boxes": boxes_total}


def refresh_data_yaml(dataset_dir: Path) -> None:
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "rock"},
    }
    (dataset_dir / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def retrain(base_model: Path, data_yaml: Path, epochs: int, batch: int, name: str) -> Path:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(base_model))
    model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=name,
        exist_ok=True,
        patience=25,
        save=True,
        plots=True,
        # Color/shape diversity augmentations for varied rock appearance.
        hsv_h=0.03,
        hsv_s=0.75,
        hsv_v=0.55,
        degrees=12.0,
        translate=0.12,
        scale=0.55,
        shear=2.0,
        mosaic=1.0,
        mixup=0.12,
        copy_paste=0.1,
        fliplr=0.5,
        verbose=True,
    )
    best = PROJECT_ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    export = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
    if best.exists():
        shutil.copy2(best, export)
    return export


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich dataset with rock color/shape diversity")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample-every", type=int, default=18)
    parser.add_argument("--max-per-video", type=int, default=25)
    parser.add_argument("--scan-conf", type=float, default=0.52)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Fine-tune from this .pt (default: --model)",
    )
    parser.add_argument("--run-name", default="mars_rock_shape_color")
    parser.add_argument(
        "--skip-mining",
        action="store_true",
        help="Only retrain — skip video/NASA mining (use after first enrich run)",
    )
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        raise FileNotFoundError("Dataset missing. Run: python scripts/download_dataset.py")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    train_img = DATASET_DIR / "train" / "images"
    train_lbl = DATASET_DIR / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))

    video_stats: list[dict] = []
    nasa_stats: list[dict] = []

    if args.skip_mining:
        print("Skipping dataset mining (--skip-mining).")
    else:
        print("Step 1/3: Mining frames from sample videos ...")
        for video in sorted(SAMPLES_DIR.glob("*.mp4")):
            is_neg = video.stem in NEGATIVE_VIDEO_STEMS
            print(f"  {video.name} ({'negative' if is_neg else 'positive'})")
            stats = mine_from_video(
                video,
                model,
                is_negative=is_neg,
                sample_every=args.sample_every,
                max_frames=args.max_per_video,
                scan_conf=args.scan_conf,
                train_img=train_img,
                train_lbl=train_lbl,
                device=device,
            )
            video_stats.append(stats)
            print(f"    -> {stats}")

        print("\nStep 2/3: Fetching diverse rock images (NASA) ...")
        STAGING.mkdir(parents=True, exist_ok=True)
        for i, (query, count) in enumerate(NASA_ROCK_QUERIES):
            sub = STAGING / f"batch_{i}"
            n = fetch_diverse_images(query, count, sub)
            mined = mine_from_staged_images(sub, train_img, train_lbl, f"div_{i}")
            nasa_stats.append({"query": query, "downloaded": n, **mined})
            print(f"  [{query}] downloaded={n}, labeled={mined}")

        refresh_data_yaml(DATASET_DIR)

    train_count = len(list(train_img.glob("*")))
    export_path = str(args.model)
    if args.retrain:
        base = args.base_model or args.model
        print(f"\nStep 3/3: Retraining from {base} for {args.epochs} epochs ...")
        export_path = str(retrain(base, DATASET_DIR / "data.yaml", args.epochs, args.batch, args.run_name))
        print(f"  Updated: {export_path}")
    else:
        print("\nStep 3/3: Skipped retrain (pass --retrain to fine-tune)")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_images_total": train_count,
        "video_mining": video_stats,
        "nasa_diversity": nasa_stats,
        "retrained": args.retrain,
        "model": export_path,
        "next": "python scripts/detect_rocks.py --video Videos/samples/06_terrain_negative_test.mp4 --model models/mars_rock_detector.pt",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Train images now: {train_count}")


if __name__ == "__main__":
    main()
