"""Shared video discovery and rock detection helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import cvzone
from ultralytics import YOLO

from detection_filters import is_likely_rock_box

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CLASS_NAMES = ["rock"]


@dataclass
class DetectionStats:
    video: str
    output: str | None = None
    frames: int = 0
    boxes_raw: int = 0
    boxes_kept: int = 0
    avg_conf: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def filtered_pct(self) -> float:
        if self.boxes_raw == 0:
            return 0.0
        return 100 * (self.boxes_raw - self.boxes_kept) / self.boxes_raw


def collect_video_paths(source: Path | None, samples_dir: Path) -> list[Path]:
    """Resolve one file, a folder, or all videos in the default samples folder."""
    if source is None:
        target = samples_dir
    elif source.is_dir():
        target = source
    elif source.is_file():
        return [source]
    else:
        raise FileNotFoundError(f"Video source not found: {source}")

    videos = sorted(
        p for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(
            f"No videos in {target}. Run: python scripts/download_videos.py\n"
            f"Or drop .mp4 files into: {samples_dir}"
        )
    return videos


def process_video(
    model: YOLO,
    source: Path,
    *,
    conf: float,
    max_area: float,
    save: bool,
    show: bool,
    output_dir: Path,
    project_root: Path,
) -> DetectionStats:
    """Run rock detection on a video and optionally save annotated output."""
    stats = DetectionStats(video=source.name)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        stats.errors.append("cannot open video")
        return stats

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{source.stem}_detected.mp4"
        stats.output = str(out_path.relative_to(project_root))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in,
            (w, h),
        )

    conf_sum = 0.0
    prev_frame_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break
        stats.frames += 1

        results = model(img, stream=True, conf=conf, iou=0.45, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                stats.boxes_raw += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                c = float(box.conf[0])
                if not is_likely_rock_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    c,
                    w,
                    h,
                    min_conf=conf,
                    max_area_ratio=max_area,
                    img=img,
                ):
                    continue
                stats.boxes_kept += 1
                conf_sum += c
                bw, bh = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, bw, bh), l=10, t=2)
                cls = int(box.cls[0])
                label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "rock"
                cvzone.putTextRect(
                    img,
                    f"{label} {math.ceil(c * 100) / 100}",
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                )

        now = time.time()
        fps = 1 / (now - prev_frame_time) if now > prev_frame_time else 0
        prev_frame_time = now
        cvzone.putTextRect(img, f"FPS {int(fps)}", (10, 30), scale=1, thickness=1)

        if writer:
            writer.write(img)
        if show:
            cv2.imshow(f"Rock Detection — {source.name}", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    stats.avg_conf = round(conf_sum / max(stats.boxes_kept, 1), 3)
    return stats
