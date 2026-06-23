"""
Download and prepare Mars terrain rock detection datasets in YOLO format.

Sources:
  1. Roboflow Universe (labeled, bounding boxes) — requires free ROBOFLOW_API_KEY
  2. NASA Images API (raw Mars terrain imagery for your own collection)

Usage:
  set ROBOFLOW_API_KEY=your_key_here
  python scripts/download_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MARS_ROCKS_DIR = DATA_DIR / "mars_rocks"

# Roboflow public Mars rock / terrain datasets (object detection)
ROBOFLOW_DATASETS = [
    {"workspace": "srl-rock-detection-clover", "project": "detect-mars-rocks-3", "version": 2},
    {"workspace": "jiaowobaba", "project": "mars-data", "version": 1},
    {"workspace": "mars-ncpml", "project": "mars-terrain-a5loz", "version": 1},
]

# Class names mapped to unified "rock" label (class id 0)
ROCK_CLASS_ALIASES = {
    "rock",
    "rocks",
    "boulder",
    "boulders",
    "big_rock",
    "small_rock",
    "shiny rock",
    "obstacle",
    "graval-terrain",
    "hard-terrain",
}


def download_nasa_mars_images(output_dir: Path, count: int = 50) -> int:
    """Fetch raw Mars terrain images from NASA Images API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = "https://images-api.nasa.gov/search"
    params = {
        "q": "Mars surface rocks terrain",
        "media_type": "image",
        "page_size": min(count, 100),
    }

    print(f"\n[NASA] Searching Mars terrain images (target: {count}) ...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    items = response.json().get("collection", {}).get("items", [])

    saved = 0
    for item in tqdm(items, desc="NASA images"):
        if saved >= count:
            break
        try:
            metadata_url = item["href"]
            meta = requests.get(metadata_url, timeout=30).json()

            # NASA returns either a list of image URLs or metadata objects with links.
            img_urls: list[str] = []
            if isinstance(meta, list):
                if meta and isinstance(meta[0], str):
                    img_urls = [u for u in meta if u.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))]
                elif meta and isinstance(meta[0], dict):
                    for entry in meta:
                        for link in entry.get("links", []):
                            if link.get("render") == "image":
                                img_urls.append(link["href"])
            elif isinstance(meta, dict):
                for link in meta.get("links", []):
                    if link.get("render") == "image":
                        img_urls.append(link["href"])

            if not img_urls:
                continue

            # Prefer original resolution when available.
            img_url = next((u for u in img_urls if "~orig" in u), img_urls[0])
            ext = Path(img_url.split("?")[0]).suffix or ".jpg"
            dest = output_dir / f"nasa_mars_{saved:04d}{ext}"
            if dest.exists():
                saved += 1
                continue
            img_data = requests.get(img_url, timeout=60)
            img_data.raise_for_status()
            dest.write_bytes(img_data.content)
            saved += 1
        except Exception as exc:
            print(f"  skip: {exc}")
            continue

    print(f"[NASA] Saved {saved} raw images to {output_dir}")
    return saved


def _load_class_names(dataset_dir: Path) -> dict[int, str]:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return {0: "rock"}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {0: "rock"})
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    return {i: str(n).lower() for i, n in enumerate(names)}


def normalize_labels(dataset_dir: Path) -> None:
    """Keep only rock-like classes; drop craters/dunes/terrain mislabeled as rock."""
    class_names = _load_class_names(dataset_dir)
    rock_ids = {
        cid for cid, name in class_names.items()
        if name in ROCK_CLASS_ALIASES or "rock" in name or "boulder" in name
    }
    if not rock_ids:
        rock_ids = {0}

    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            lines = label_file.read_text(encoding="utf-8").strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id not in rock_ids:
                    continue
                parts[0] = "0"
                new_lines.append(" ".join(parts))
            label_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


def write_unified_data_yaml(dataset_dir: Path) -> Path:
    """Write dataset.yaml with absolute path for Ultralytics training."""
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
    return yaml_path


def merge_yolo_datasets(source_dirs: list[Path], dest_dir: Path) -> None:
    """Merge multiple YOLO-format datasets into one directory."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    for split in ("train", "valid", "test"):
        (dest_dir / split / "images").mkdir(parents=True)
        (dest_dir / split / "labels").mkdir(parents=True)

    counters = {"train": 0, "valid": 0, "test": 0}

    for src in source_dirs:
        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            lbl_dir = src / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                idx = counters[split]
                counters[split] += 1
                stem = f"{src.name}_{idx:05d}"
                shutil.copy2(img_path, dest_dir / split / "images" / f"{stem}{img_path.suffix.lower()}")
                lbl_src = lbl_dir / f"{img_path.stem}.txt"
                lbl_dest = dest_dir / split / "labels" / f"{stem}.txt"
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_dest)
                else:
                    lbl_dest.write_text("", encoding="utf-8")


GITHUB_MARS_DATASET_ZIP = (
    "https://github.com/Gson-glitch/Mars-Rock-Detection-with-ROS2-and-YOLOv11/archive/refs/heads/main.zip"
)
GITHUB_DATASET_PREFIX = (
    "Mars-Rock-Detection-with-ROS2-and-YOLOv11-main/dataset/mars rocks detection.v4i.yolov11/"
)


def download_github_mars_rocks_dataset(staging_dir: Path) -> Path | None:
    """Extract labeled Mars rock YOLO dataset bundled in a public GitHub repository."""
    print("\n[GitHub] Downloading public Mars rocks YOLO dataset ...")
    staging_dir.mkdir(parents=True, exist_ok=True)
    zip_path = staging_dir / "mars_rocks_github.zip"
    extract_root = staging_dir / "mars_rocks_github"

    if not zip_path.exists():
        response = requests.get(GITHUB_MARS_DATASET_ZIP, timeout=300)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        print(f"  Downloaded {len(response.content) / 1e6:.1f} MB")

    if extract_root.exists():
        shutil.rmtree(extract_root)

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m.startswith(GITHUB_DATASET_PREFIX)]
        if not members:
            print("  [WARN] Dataset folder not found in GitHub archive.")
            return None
        archive.extractall(staging_dir, members=members)

    source = staging_dir / GITHUB_DATASET_PREFIX.rstrip("/")
    if not (source / "train" / "images").exists():
        print(f"  [WARN] Expected train/images missing under {source}")
        return None

    print(f"  -> {source}")
    return source


def download_roboflow_datasets(api_key: str, staging_dir: Path) -> list[Path]:
    """Download labeled Mars datasets from Roboflow Universe."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    downloaded: list[Path] = []

    for spec in ROBOFLOW_DATASETS:
        ws, proj, ver = spec["workspace"], spec["project"], spec["version"]
        print(f"\n[Roboflow] Downloading {ws}/{proj} v{ver} ...")
        try:
            project = rf.workspace(ws).project(proj)
            dataset = project.version(ver).download("yolov11", location=str(staging_dir / f"{proj}_v{ver}"))
            ds_path = Path(dataset.location)
            normalize_labels(ds_path)
            downloaded.append(ds_path)
            print(f"  -> {ds_path}")
        except Exception as exc:
            print(f"  [WARN] Could not download {ws}/{proj}: {exc}")

    return downloaded


def ensure_min_splits(dataset_dir: Path) -> None:
    """Create valid/test splits from train if missing."""
    train_img = dataset_dir / "train" / "images"
    if not train_img.exists() or not any(train_img.iterdir()):
        raise RuntimeError("No training images found after download.")

    for split in ("valid", "test"):
        split_img = dataset_dir / split / "images"
        if split_img.exists() and any(split_img.iterdir()):
            continue
        print(f"[INFO] Creating '{split}' split from train data ...")
        images = sorted(train_img.iterdir())
        n = len(images)
        if split == "valid":
            subset = images[int(n * 0.85) : int(n * 0.95)] or images[-max(1, n // 10) :]
        else:
            subset = images[int(n * 0.95) :] or images[-max(1, n // 20) :]
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in subset:
            lbl = dataset_dir / "train" / "labels" / f"{img.stem}.txt"
            shutil.copy2(img, dataset_dir / split / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, dataset_dir / split / "labels" / f"{img.stem}.txt")


def print_dataset_stats(dataset_dir: Path) -> None:
    stats = {}
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        stats[split] = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    print("\n=== Dataset summary ===")
    print(json.dumps(stats, indent=2))
    print(f"Unified config: {dataset_dir / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Mars rock detection dataset")
    parser.add_argument("--nasa-count", type=int, default=50, help="Raw NASA images to collect")
    parser.add_argument("--skip-nasa", action="store_true", help="Skip NASA raw image download")
    parser.add_argument("--skip-roboflow", action="store_true", help="Skip Roboflow labeled download")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_nasa:
        download_nasa_mars_images(RAW_DIR / "nasa_mars", count=args.nasa_count)

    roboflow_paths: list[Path] = []
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    if not args.skip_roboflow:
        if not api_key:
            print(
                "\n[WARN] ROBOFLOW_API_KEY not set. Labeled bounding-box data requires a free key from "
                "https://app.roboflow.com/settings/api\n"
                "       Set it:  $env:ROBOFLOW_API_KEY='your_key'  (PowerShell)\n"
                "       Then re-run this script.\n"
            )
        else:
            staging = DATA_DIR / "roboflow_staging"
            staging.mkdir(parents=True, exist_ok=True)
            roboflow_paths = download_roboflow_datasets(api_key, staging)

    if roboflow_paths:
        if len(roboflow_paths) == 1:
            if MARS_ROCKS_DIR.exists():
                shutil.rmtree(MARS_ROCKS_DIR)
            shutil.copytree(roboflow_paths[0], MARS_ROCKS_DIR)
        else:
            merge_yolo_datasets(roboflow_paths, MARS_ROCKS_DIR)
        normalize_labels(MARS_ROCKS_DIR)
        ensure_min_splits(MARS_ROCKS_DIR)
    elif not (MARS_ROCKS_DIR / "train" / "images").exists():
        github_staging = DATA_DIR / "github_staging"
        github_path = download_github_mars_rocks_dataset(github_staging)
        if github_path:
            if MARS_ROCKS_DIR.exists():
                shutil.rmtree(MARS_ROCKS_DIR)
            shutil.copytree(github_path, MARS_ROCKS_DIR)
            normalize_labels(MARS_ROCKS_DIR)
            ensure_min_splits(MARS_ROCKS_DIR)
        else:
            print(
                "\n[ERROR] No labeled dataset available. Set ROBOFLOW_API_KEY and re-run, or manually place "
                "a YOLO dataset under data/mars_rocks/ with train/valid splits."
            )
            sys.exit(1)

    yaml_path = write_unified_data_yaml(MARS_ROCKS_DIR)
    print_dataset_stats(MARS_ROCKS_DIR)
    print(f"\nReady for training. Use: python scripts/train.py --data {yaml_path}")


if __name__ == "__main__":
    main()
