"""
Compare YOLO11n (nano) vs YOLO11m (medium) on the curated eval video set.

Uses the same rock filters as detect_rocks.py for a fair apples-to-apples comparison.

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --conf 0.40
  python scripts/compare_models.py --models models/mars_rock_detector.pt models/mars_rock_detector_m.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_config import EVAL_DIR, MODEL_MEDIUM, MODEL_NANO, OUTPUT_DIR, eval_video_paths
from video_detection import process_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = OUTPUT_DIR / "evaluation" / "model_comparison.json"


def run_model_on_eval(
    model_path: Path,
    videos: list[Path],
    *,
    conf: float,
    max_area: float,
    out_subdir: str,
) -> list[dict]:
    if not model_path.exists():
        return [{"model": model_path.name, "error": "file not found"}]

    model = YOLO(str(model_path))
    out_dir = OUTPUT_DIR / "evaluation" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"Model: {model_path.name}")
    print(f"{'=' * 60}")

    for video in videos:
        stats = process_video(
            model,
            video,
            conf=conf,
            max_area=max_area,
            save=True,
            show=False,
            output_dir=out_dir,
            project_root=PROJECT_ROOT,
        )
        row = {
            "video": video.name,
            "frames": stats.frames,
            "rocks_detected": stats.boxes_kept,
            "raw_boxes": stats.boxes_raw,
            "filtered_pct": round(stats.filtered_pct, 1),
            "avg_conf": stats.avg_conf,
            "output": stats.output,
            "errors": stats.errors,
        }
        results.append(row)
        print(
            f"  {video.name}: {stats.boxes_kept} rocks / {stats.frames} frames "
            f"(avg conf {stats.avg_conf}, filtered {stats.filtered_pct:.1f}%)"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare nano vs medium YOLO on eval videos")
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=[MODEL_NANO, MODEL_MEDIUM],
        help="Model .pt files to compare",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--max-area", type=float, default=0.12)
    args = parser.parse_args()

    videos = eval_video_paths()
    print(f"Eval set ({len(videos)} videos) from {EVAL_DIR}")

    comparison: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(EVAL_DIR.relative_to(PROJECT_ROOT)),
        "videos": [v.name for v in videos],
        "conf": args.conf,
        "models": [],
    }

    for model_path in args.models:
        model_path = model_path.resolve()
        tag = model_path.stem.replace("mars_rock_detector", "yolo").replace("_", "-")
        if tag == "yolo":
            tag = "yolo-nano"
        rows = run_model_on_eval(
            model_path,
            videos,
            conf=args.conf,
            max_area=args.max_area,
            out_subdir=tag,
        )
        total_rocks = sum(r.get("rocks_detected", 0) for r in rows if "rocks_detected" in r)
        total_frames = sum(r.get("frames", 0) for r in rows if "frames" in r)
        comparison["models"].append(
            {
                "path": str(model_path.relative_to(PROJECT_ROOT)) if model_path.exists() else str(model_path),
                "label": model_path.stem,
                "total_rocks_detected": total_rocks,
                "total_frames": total_frames,
                "per_video": rows,
            }
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nComparison report: {REPORT_PATH}")

    valid = [m for m in comparison["models"] if "error" not in str(m.get("per_video", ""))]
    if len(valid) >= 2:
        n_rocks = valid[0]["total_rocks_detected"]
        m_rocks = valid[1]["total_rocks_detected"]
        print(f"\nSummary — rocks detected (all eval videos):")
        print(f"  {valid[0]['label']}: {n_rocks}")
        print(f"  {valid[1]['label']}: {m_rocks}")
        if m_rocks > n_rocks:
            print("  -> Medium model detected more rocks on the eval set.")
        elif m_rocks < n_rocks:
            print("  -> Nano model detected more rocks (check for false positives in output videos).")
        else:
            print("  -> Equal detection counts — compare output videos visually.")


if __name__ == "__main__":
    main()
