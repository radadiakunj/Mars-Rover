"""
Build Videos/eval/08_pi_closeup_rock_test.mp4 — simulates Pi camera with a rock in front.

Sequence (640x480, like IMX219 preview):
  1. Empty terrain (no rocks) — model should stay quiet
  2. Close-up labeled rock images — model should draw boxes
  3. Brief empty again — sanity check

Usage:
  python scripts/build_pi_test_video.py
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMG = PROJECT_ROOT / "data" / "mars_rocks" / "train" / "images"
TRAIN_LBL = PROJECT_ROOT / "data" / "mars_rocks" / "train" / "labels"
OUT = PROJECT_ROOT / "Videos" / "eval" / "08_pi_closeup_rock_test.mp4"

WIDTH, HEIGHT, FPS = 640, 480, 20
HOLD_SEC = 1.5
EMPTY_SEC = 1.0


def has_large_rock_label(lbl_path: Path) -> bool:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return False
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            bw, bh = float(parts[3]), float(parts[4])
            if bw * bh >= 0.02:
                return True
    return False


def is_negative_image(lbl_path: Path) -> bool:
    return lbl_path.exists() and lbl_path.stat().st_size == 0


def fit_frame(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((HEIGHT, WIDTH, 3), (40, 35, 30), dtype=np.uint8)
    y0 = (HEIGHT - nh) // 2
    x0 = (WIDTH - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def collect_images() -> tuple[list[Path], list[Path]]:
    positives: list[Path] = []
    negatives: list[Path] = []
    for img in TRAIN_IMG.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = TRAIN_LBL / f"{img.stem}.txt"
        if is_negative_image(lbl):
            negatives.append(img)
        elif has_large_rock_label(lbl):
            positives.append(img)
    return positives, negatives


def write_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    for f in frames:
        writer.write(f)
    writer.release()


def main() -> None:
    positives, negatives = collect_images()
    if len(positives) < 6:
        raise RuntimeError("Not enough labeled rock images in data/mars_rocks/train/")

    random.seed(42)
    rock_imgs = random.sample(positives, min(12, len(positives)))
    empty_imgs = random.sample(negatives, min(4, len(negatives))) if negatives else []

    frames: list[np.ndarray] = []
    n_empty = int(EMPTY_SEC * FPS)
    n_hold = int(HOLD_SEC * FPS)

    for _ in range(n_empty):
        if empty_imgs:
            img = cv2.imread(str(random.choice(empty_imgs)))
            if img is not None:
                frames.append(fit_frame(img))
                continue
        frames.append(np.full((HEIGHT, WIDTH, 3), (55, 50, 45), dtype=np.uint8))

    for img_path in rock_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        frame = fit_frame(img)
        for _ in range(n_hold):
            frames.append(frame)

    for _ in range(n_empty):
        frames.append(np.full((HEIGHT, WIDTH, 3), (55, 50, 45), dtype=np.uint8))

    write_video(frames, OUT)
    duration = len(frames) / FPS
    print(f"Created: {OUT}")
    print(f"  {len(frames)} frames, {duration:.1f}s @ {FPS}fps, {WIDTH}x{HEIGHT}")
    print(f"  Rock scenes: {len(rock_imgs)} | Empty lead-in/out: {EMPTY_SEC}s each")


if __name__ == "__main__":
    main()
