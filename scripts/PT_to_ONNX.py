"""
Export a trained YOLO .pt model to ONNX for lighter Raspberry Pi deployment.

Usage:
  python scripts/PT_to_ONNX.py
  python scripts/PT_to_ONNX.py --model models/mars_rock_detector_m.pt
  python scripts/PT_to_ONNX.py --model models/mars_rock_detector.pt --imgsz 640

Output: models/<name>.onnx (same folder as the source .pt)

See Implementation.txt — Section 12 (ONNX) and Section 13 (Pi deployment).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mars_rock_detector_m.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO .pt weights to ONNX")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Source .pt file")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (match training)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset (12 is safe on Pi / ARM)")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Dynamic input shape (flexible but slower on Pi — leave off for deployment)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(str(args.model))
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=args.dynamic,
        half=False,  # FP32 — best compatibility on Pi CPU / onnxruntime
    )
    print(f"\nExported ONNX: {out}")
    print("On Pi: pip install onnxruntime  then use onnxruntime or ultralytics YOLO('model.onnx')")


if __name__ == "__main__":
    main()
