"""Paths and lists for the curated rock-detection evaluation set."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "Videos" / "eval"
ARCHIVE_DIR = PROJECT_ROOT / "Videos" / "archive"
OUTPUT_DIR = PROJECT_ROOT / "Videos" / "output"

# Official benchmark clips — clearly visible rocks + Pi close-up simulation.
EVAL_VIDEO_NAMES = (
    "01_mars_rover_rocks.mp4",
    "05_rock_labeled_variety.mp4",
    "08_pi_closeup_rock_test.mp4",
)

MODEL_NANO = PROJECT_ROOT / "models" / "mars_rock_detector.pt"
MODEL_MEDIUM = PROJECT_ROOT / "models" / "mars_rock_detector_m.pt"


def eval_video_paths() -> list[Path]:
    paths = [EVAL_DIR / name for name in EVAL_VIDEO_NAMES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Evaluation videos missing. Run: python scripts/setup_eval_videos.py\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    return paths
