"""Color and shape cues to distinguish rocks from Mars-like sand and terrain."""

from __future__ import annotations

import cv2
import numpy as np


def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def appearance_score(crop_bgr: np.ndarray) -> float:
    """
    Score 0–1: higher means more rock-like (textured, non-uniform sand color).

    Rocks on Mars terrain tend to be grayer/darker with sharper edges than
    uniform orange-red sand patches.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Texture: rocks have more high-frequency detail than smooth sand.
    texture = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 180.0, 1.0)

    # Color variation inside the box (sand patches are very uniform).
    sat_std = float(np.std(hsv[:, :, 1])) / 80.0
    val_std = float(np.std(hsv[:, :, 2])) / 70.0
    color_var = min((sat_std + val_std) / 2.0, 1.0)

    # Sand-like orange/red: hue ~5–25, high saturation.
    sand_mask = cv2.inRange(hsv, (5, 55, 50), (28, 255, 255))
    sand_ratio = float(np.count_nonzero(sand_mask)) / max(sand_mask.size, 1)
    non_sand = 1.0 - sand_ratio

    # Edge density — rocks usually have a closed contour boundary.
    edges = cv2.Canny(gray, 40, 120)
    edge_density = min(float(np.count_nonzero(edges)) / max(edges.size, 1) * 12.0, 1.0)

    # Compact blob shape (reject long thin sand streaks).
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri > 0 and area > 20:
            compactness = min(4.0 * np.pi * area / (peri * peri), 1.0)

    score = (
        0.30 * texture
        + 0.25 * color_var
        + 0.20 * non_sand
        + 0.15 * edge_density
        + 0.10 * compactness
    )
    return float(np.clip(score, 0.0, 1.0))


def is_rock_like_appearance(crop_bgr: np.ndarray, *, min_score: float = 0.32) -> bool:
    return appearance_score(crop_bgr) >= min_score


def appearance_score_for_box(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    crop = _safe_crop(img, x1, y1, x2, y2)
    if crop is None:
        return 0.0
    return appearance_score(crop)


def find_rock_boxes_by_appearance(
    frame_bgr: np.ndarray,
    *,
    min_area_ratio: float = 0.0008,
    max_area_ratio: float = 0.10,
    min_score: float = 0.38,
) -> list[tuple[int, int, int, int]]:
    """Segment non-sand blobs and return boxes that look rock-like by color/shape."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Non-sand: gray rocks, dark shadows on stone, bluish-gray pebbles.
    sand = cv2.inRange(hsv, (5, 50, 45), (30, 255, 255))
    dark = cv2.inRange(gray, 0, 95)
    low_sat = cv2.inRange(hsv, (0, 0, 40), (180, 70, 255))
    candidate = cv2.bitwise_or(cv2.bitwise_not(sand), cv2.bitwise_or(dark, low_sat))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    img_area = w * h

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area_ratio = (bw * bh) / max(img_area, 1)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 3.5:
            continue
        crop = _safe_crop(frame_bgr, x, y, x + bw, y + bh)
        if crop is None or not is_rock_like_appearance(crop, min_score=min_score):
            continue
        boxes.append((x, y, x + bw, y + bh))

    return boxes


def to_yolo_line(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> str:
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
