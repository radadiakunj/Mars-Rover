"""
Download diverse rock-related sample videos into Videos/samples/.

Drop your own .mp4 files into Videos/samples/ anytime — no code changes needed.

Usage:
  python scripts/download_videos.py
"""

from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "Videos" / "samples"
LEGACY_DIR = PROJECT_ROOT / "Videos"
GITHUB_ZIP = (
    "https://github.com/Gson-glitch/Mars-Rock-Detection-with-ROS2-and-YOLOv11/archive/refs/heads/main.zip"
)
MARS_VIDEO_PATH = (
    "Mars-Rock-Detection-with-ROS2-and-YOLOv11-main/dataset/mars-video.mp4"
)

MANIFEST = [
    {
        "file": "01_mars_rover_rocks.mp4",
        "description": "Mars rover rock field — primary reference clip",
        "source": "GitHub Mars-Rock-Detection dataset",
    },
    {
        "file": "02_mars_terrain_slideshow.mp4",
        "description": "Labeled Mars terrain images — varied angles and lighting",
        "source": "Training set slideshow",
    },
    {
        "file": "03_nasa_mars_surface.mp4",
        "description": "NASA Mars surface gallery — different resolutions and scenes",
        "source": "NASA Images API",
    },
    {
        "file": "04_desert_rock_analog.mp4",
        "description": "Earth desert/geology analog — tests generalization beyond Mars palette",
        "source": "NASA Images API",
    },
    {
        "file": "05_rock_labeled_variety.mp4",
        "description": "Random labeled rock scenes — mixed scales and backgrounds",
        "source": "Training set (labeled rocks only)",
    },
    {
        "file": "06_terrain_negative_test.mp4",
        "description": "Mostly empty terrain — precision test (should detect few/no rocks)",
        "source": "Negative training frames",
    },
    {
        "file": "07_rover_pan_simulation.mp4",
        "description": "Simulated rover forward motion via image pan — camera angle change",
        "source": "Synthetic pan from Mars images",
    },
]


def fetch_nasa_images(query: str, count: int, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = "https://images-api.nasa.gov/search"
    params = {"q": query, "media_type": "image", "page_size": min(count, 100)}
    items = requests.get(url, params=params, timeout=60).json().get("collection", {}).get("items", [])

    saved: list[Path] = []
    for item in items:
        if len(saved) >= count:
            break
        try:
            meta = requests.get(item["href"], timeout=30).json()
            img_urls: list[str] = []
            if isinstance(meta, list):
                if meta and isinstance(meta[0], str):
                    img_urls = [u for u in meta if u.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            for img_url in img_urls[:1]:
                ext = Path(img_url.split("?")[0]).suffix or ".jpg"
                dest = output_dir / f"{query[:20].replace(' ', '_')}_{len(saved):03d}{ext}"
                data = requests.get(img_url, timeout=60)
                data.raise_for_status()
                dest.write_bytes(data.content)
                saved.append(dest)
        except Exception:
            continue
    return saved


def images_to_video(image_paths: list[Path], dest: Path, fps: int = 8, hold: int = 4) -> bool:
    if not image_paths:
        return False
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    out_w, out_h = int(w * scale), int(h * scale)

    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        return False

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (out_w, out_h))
        for _ in range(hold):
            writer.write(frame)
    writer.release()
    return dest.exists()


def build_pan_video(image_paths: list[Path], dest: Path, fps: int = 15) -> bool:
    """Simulate rover motion by panning across wide images."""
    if not image_paths:
        return False
    img = cv2.imread(str(image_paths[0]))
    if img is None:
        return False
    h, w = img.shape[:2]
    if w < 640:
        return images_to_video(image_paths[:1], dest, fps=fps, hold=30)

    view_w = min(640, w)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (view_w, min(480, h)))
    if not writer.isOpened():
        return False

    steps = max(30, (w - view_w) // 8)
    for i in range(steps):
        x = int((w - view_w) * i / max(steps - 1, 1))
        crop = img[: min(480, h), x : x + view_w]
        if crop.shape[1] != view_w:
            continue
        if crop.shape[0] < 480:
            pad = 480 - crop.shape[0]
            crop = cv2.copyMakeBorder(crop, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        writer.write(crop)
    writer.release()
    return dest.exists()


def download_github_mars_video(dest: Path) -> bool:
    response = requests.get(GITHUB_ZIP, timeout=300)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        if MARS_VIDEO_PATH not in archive.namelist():
            return False
        dest.write_bytes(archive.read(MARS_VIDEO_PATH))
    return True


def build_all_samples() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "Videos" / "output").mkdir(parents=True, exist_ok=True)
    staging = PROJECT_ROOT / "data" / "video_staging"
    staging.mkdir(parents=True, exist_ok=True)

    train_img = PROJECT_ROOT / "data" / "mars_rocks" / "train" / "images"
    labeled = sorted(
        p for p in train_img.glob("rock_*.jpg")
        if p.is_file()
    ) if train_img.exists() else []
    negatives = sorted(train_img.glob("neg_*.jpg")) if train_img.exists() else []

    print("[1/7] Mars rover rocks clip ...")
    download_github_mars_video(SAMPLES_DIR / "01_mars_rover_rocks.mp4")

    print("[2/7] Mars terrain slideshow ...")
    if labeled:
        images_to_video(labeled[:100], SAMPLES_DIR / "02_mars_terrain_slideshow.mp4", fps=10, hold=5)

    print("[3/7] NASA Mars surface gallery ...")
    nasa_mars = fetch_nasa_images("Mars surface rocks boulders", 35, staging / "nasa_mars")
    images_to_video(nasa_mars, SAMPLES_DIR / "03_nasa_mars_surface.mp4", fps=8, hold=4)

    print("[4/7] Desert rock analog ...")
    desert = fetch_nasa_images("desert rocks geology field arid", 35, staging / "desert")
    if len(desert) < 5:
        desert = fetch_nasa_images("geology rock outcrop arid landscape", 35, staging / "desert2")
    if len(desert) < 5 and labeled:
        desert = random.sample(labeled, min(25, len(labeled)))
    images_to_video(desert, SAMPLES_DIR / "04_desert_rock_analog.mp4", fps=8, hold=4)

    print("[5/7] Labeled rock variety ...")
    if labeled:
        random.seed(42)
        variety = random.sample(labeled, min(80, len(labeled)))
        images_to_video(variety, SAMPLES_DIR / "05_rock_labeled_variety.mp4", fps=12, hold=3)

    print("[6/7] Terrain negative test ...")
    if negatives:
        images_to_video(negatives[:40], SAMPLES_DIR / "06_terrain_negative_test.mp4", fps=8, hold=5)
    elif nasa_mars:
        images_to_video(nasa_mars[10:30], SAMPLES_DIR / "06_terrain_negative_test.mp4", fps=8, hold=5)

    print("[7/7] Rover pan simulation ...")
    pan_src = nasa_mars or labeled
    if pan_src:
        build_pan_video(pan_src[:5], SAMPLES_DIR / "07_rover_pan_simulation.mp4")

    # Legacy symlinks/copies for older scripts
    legacy_map = {
        "mars_rocks_sample.mp4": "01_mars_rover_rocks.mp4",
        "mars_terrain_dataset.mp4": "02_mars_terrain_slideshow.mp4",
    }
    for old, new in legacy_map.items():
        src = SAMPLES_DIR / new
        if src.exists():
            shutil_copy(src, LEGACY_DIR / old)

    manifest_path = SAMPLES_DIR / "manifest.json"
    existing = {p.name for p in SAMPLES_DIR.glob("*.mp4")}
    manifest_path.write_text(
        json.dumps([m for m in MANIFEST if m["file"] in existing], indent=2),
        encoding="utf-8",
    )

    print("\n=== Sample videos (Videos/samples/) ===")
    for p in sorted(SAMPLES_DIR.glob("*.mp4")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
    print("\nTest ALL samples (no code changes):")
    print("  python scripts/YOLO_Detection.py --all --save")
    print("\nTest ONE video:")
    print("  python scripts/YOLO_Detection.py --source Videos/samples/03_nasa_mars_surface.mp4 --save --show")


def shutil_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def main() -> None:
    build_all_samples()


if __name__ == "__main__":
    main()
