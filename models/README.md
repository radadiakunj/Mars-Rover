# Trained models

| File | Size | Use |
|------|------|-----|
| `mars_rock_detector.pt` | ~5.5 MB | YOLO11n — fast inference (Pi / quick tests) |
| `mars_rock_detector_m.pt` | ~40 MB | YOLO11m — best accuracy (recommended) |
| `mars_rock_detector_m.onnx` | ~80 MB | Pi deployment via `onnxruntime` |

## Usage

```bash
# PC video detection
python scripts/detect_rocks.py --video Videos/eval/08_pi_closeup_rock_test.mp4 --model models/mars_rock_detector_m.pt --conf 0.35

# Pi live camera
python scripts/inference_pi.py --model models/mars_rock_detector_m.onnx --conf 0.35 --no-filters --stream
```

## Retrain

```bash
python scripts/train.py
python scripts/PT_to_ONNX.py --model models/mars_rock_detector_m.pt
```
