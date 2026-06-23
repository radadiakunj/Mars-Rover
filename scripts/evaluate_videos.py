"""
Evaluate the trained rock detector on the curated eval set (Videos/eval/).

Only videos with clearly visible rocks are included.
Excluded clips live in Videos/archive/ (see manifest.json).

Usage:
  python scripts/evaluate_videos.py
  python scripts/evaluate_videos.py --model models/mars_rock_detector_m.pt
  python scripts/evaluate_videos.py --conf 0.40 --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_config import MODEL_NANO, OUTPUT_DIR, eval_video_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = OUTPUT_DIR / "evaluation"


def resolve_device(requested: str) -> str | int:
    if requested:
        return requested
    return 0 if torch.cuda.is_available() else "cpu"


def evaluate_video(
    model: YOLO,
    source: Path,
    output_video: Path,
    conf: float,
    device: str | int,
    save_miss_frames: bool,
    miss_dir: Path,
) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    stats = {
        "video": source.name,
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "frames_with_detections": 0,
        "frames_without_detections": 0,
        "total_boxes": 0,
        "confidences": [],
        "max_boxes_in_frame": 0,
        "miss_frame_samples": [],
    }

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf, device=device, verbose=False)
        boxes = results[0].boxes
        n = 0 if boxes is None else len(boxes)

        if n == 0:
            stats["frames_without_detections"] += 1
            if save_miss_frames and len(stats["miss_frame_samples"]) < 8:
                miss_path = miss_dir / f"{source.stem}_no_detect_f{frame_idx:05d}.jpg"
                cv2.imwrite(str(miss_path), frame)
                stats["miss_frame_samples"].append(miss_path.name)
        else:
            stats["frames_with_detections"] += 1
            stats["total_boxes"] += n
            stats["max_boxes_in_frame"] = max(stats["max_boxes_in_frame"], n)
            if boxes is not None and boxes.conf is not None:
                stats["confidences"].extend(boxes.conf.cpu().tolist())

        annotated = results[0].plot()
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    confs = stats["confidences"]
    stats["avg_confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
    stats["min_confidence"] = round(min(confs), 3) if confs else 0.0
    stats["max_confidence"] = round(max(confs), 3) if confs else 0.0
    stats["detection_rate_pct"] = round(
        100 * stats["frames_with_detections"] / max(frame_idx, 1), 1
    )
    stats["avg_boxes_per_detected_frame"] = round(
        stats["total_boxes"] / max(stats["frames_with_detections"], 1), 2
    )
    del stats["confidences"]
    stats["output_video"] = str(output_video.relative_to(PROJECT_ROOT))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rock detector on curated eval videos")
    parser.add_argument("--model", type=Path, default=MODEL_NANO)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--device", default="", help="0 for GPU, cpu for CPU, auto if empty")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        raise SystemExit(1)

    videos = eval_video_paths()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    miss_dir = OUTPUT_ROOT / "miss_frames"
    miss_dir.mkdir(exist_ok=True)

    print(f"Eval set : {len(videos)} videos (Videos/eval/)")
    print(f"Model    : {args.model}")
    print(f"Device   : {device}")
    print(f"Conf     : {args.conf}")

    model = YOLO(str(args.model))
    video_results = []

    for video in videos:
        out_path = OUTPUT_ROOT / f"{video.stem}_eval_detected.mp4"
        print(f"\nEvaluating: {video.name} ...")
        result = evaluate_video(model, video, out_path, args.conf, device, True, miss_dir)
        video_results.append(result)
        print(
            f"  Detection rate: {result['detection_rate_pct']}% | "
            f"Avg conf: {result['avg_confidence']} | "
            f"Output: {out_path.name}"
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set": "Videos/eval/",
        "model": str(args.model.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "confidence_threshold": args.conf,
        "videos": video_results,
    }
    report_path = OUTPUT_ROOT / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    print("Archived (non-eval) videos: Videos/archive/manifest.json")


if __name__ == "__main__":
    main()
