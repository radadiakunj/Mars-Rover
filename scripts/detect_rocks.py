"""
Rock detection on a video file — pick input video and trained .pt model.

Interactive (default):
  python scripts/detect_rocks.py

Command line:
  python scripts/detect_rocks.py --video Videos/eval/05_rock_labeled_variety.mp4 --model models/mars_rock_detector.pt
  python scripts/detect_rocks.py --video my_clip.mp4 --model runs/detect/mars_rock_train/weights/best.pt --show

Output is saved to Videos/output/<video_name>_detected.mp4
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
OUTPUT_DIR = PROJECT_ROOT / "Videos" / "output"
VIDEO_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
    ("All files", "*.*"),
]
MODEL_TYPES = [("YOLO weights", "*.pt"), ("All files", "*.*")]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_detection import process_video  # noqa: E402


def pick_file(title: str, filetypes: list[tuple[str, str]], initial_dir: Path) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial_dir),
        filetypes=filetypes,
    )
    root.destroy()
    return Path(path) if path else None


def interactive_select() -> tuple[Path, Path]:
    print("Select input video, then your trained .pt model.\n")

    video = pick_file("Select input video", VIDEO_TYPES, PROJECT_ROOT / "Videos" / "eval")
    if video is None:
        raise SystemExit("No video selected.")

    model = pick_file("Select trained YOLO model (.pt)", MODEL_TYPES, PROJECT_ROOT / "models")
    if model is None:
        raise SystemExit("No model selected.")

    return video, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect rocks in a video using your trained YOLO model"
    )
    parser.add_argument("--video", type=Path, default=None, help="Input video path")
    parser.add_argument("--model", type=Path, default=None, help="Trained .pt model path")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--max-area", type=float, default=0.12, help="Max box area ratio filter")
    parser.add_argument("--show", action="store_true", help="Show live preview (press q to quit)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.video is None or args.model is None:
        video, model_path = interactive_select()
    else:
        video, model_path = args.video, args.model

    if not video.exists():
        messagebox.showerror("Error", f"Video not found:\n{video}")
        raise SystemExit(1)
    if not model_path.exists():
        messagebox.showerror("Error", f"Model not found:\n{model_path}")
        raise SystemExit(1)

    print(f"Video  : {video}")
    print(f"Model  : {model_path}")
    print(f"Conf   : {args.conf}")
    print(f"Output : {args.output_dir}\n")

    model = YOLO(str(model_path))
    stats = process_video(
        model,
        video,
        conf=args.conf,
        max_area=args.max_area,
        save=True,
        show=args.show,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
    )

    print(f"Frames processed : {stats.frames}")
    print(f"Rocks detected   : {stats.boxes_kept}")
    print(f"Filtered (FP)    : {stats.filtered_pct:.1f}%")
    print(f"Avg confidence   : {stats.avg_conf}")
    if stats.output:
        out_full = PROJECT_ROOT / stats.output
        print(f"\nSaved: {out_full}")
    if stats.errors:
        print(f"Errors: {', '.join(stats.errors)}")


if __name__ == "__main__":
    main()
