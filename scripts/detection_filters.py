"""Shared filters to reduce false-positive rock detections at inference time."""

from __future__ import annotations

import numpy as np

from rock_appearance import appearance_score_for_box, is_rock_like_appearance


def box_area_ratio(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> float:
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    return (bw * bh) / max(img_w * img_h, 1)


def is_likely_rock_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    conf: float,
    img_w: int,
    img_h: int,
    *,
    min_conf: float = 0.50,
    max_area_ratio: float = 0.12,
    min_area_ratio: float = 0.0003,
    max_aspect: float = 4.0,
    img: np.ndarray | None = None,
    min_appearance: float = 0.32,
) -> bool:
    """Reject boxes that match sand patches / terrain instead of rocks."""
    if conf < min_conf:
        return False

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = box_area_ratio(x1, y1, x2, y2, img_w, img_h)

    if area_ratio > max_area_ratio:
        return False
    if area_ratio < min_area_ratio:
        return False

    aspect = max(bw / bh, bh / bw)
    if aspect > max_aspect:
        return False

    # Large flat boxes on the ground strip are often false positives.
    bottom_touch = y2 >= int(img_h * 0.92)
    wide = bw >= int(img_w * 0.35)
    if bottom_touch and wide and conf < 0.70:
        return False

    if img is not None:
        score = appearance_score_for_box(img, x1, y1, x2, y2)
        # Low-confidence detections need stronger color/shape evidence.
        required = min_appearance + (0.12 if conf < 0.62 else 0.0)
        if score < required:
            return False

    return True
