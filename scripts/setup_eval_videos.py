"""
Organize sample videos into eval (clear rocks) vs archive (excluded benchmarks).

  Videos/eval/     — 01, 05  (official test set)
  Videos/archive/  — 02, 03, 06, 07  (not used for standard evaluation)

Usage:
  python scripts/setup_eval_videos.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from eval_config import ARCHIVE_DIR, EVAL_DIR, EVAL_VIDEO_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SAMPLES = PROJECT_ROOT / "Videos" / "samples"

EVAL_FILES = set(EVAL_VIDEO_NAMES)
ARCHIVE_FILES = {
    "02_mars_terrain_slideshow.mp4",
    "03_nasa_mars_surface.mp4",
    "06_terrain_negative_test.mp4",
    "07_rover_pan_simulation.mp4",
}


def move_video(src: Path, dest_dir: Path) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() == src.resolve():
        return
    if dest.exists():
        return
    shutil.move(str(src), str(dest))
    print(f"  {src.name} -> {dest_dir.name}/")


def main() -> None:
    print("Setting up evaluation video folders ...\n")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Gather from legacy samples/ and any stray Videos/ root files.
    search_dirs = [LEGACY_SAMPLES, PROJECT_ROOT / "Videos"]
    seen: set[str] = set()

    for name in EVAL_FILES | ARCHIVE_FILES:
        for folder in search_dirs:
            src = folder / name
            if not src.exists() or name in seen:
                continue
            if name in EVAL_FILES:
                move_video(src, EVAL_DIR)
            else:
                move_video(src, ARCHIVE_DIR)
            seen.add(name)

    eval_count = len(list(EVAL_DIR.glob("*.mp4")))
    arch_count = len(list(ARCHIVE_DIR.glob("*.mp4")))
    print(f"\nEval videos   : {eval_count} in {EVAL_DIR}")
    print(f"Archived      : {arch_count} in {ARCHIVE_DIR}")
    print("\nStandard evaluation uses Videos/eval/ only.")
    print("Run: python scripts/compare_models.py")


if __name__ == "__main__":
    main()
