"""
Real-time rock detection for Raspberry Pi 5 + IMX219-83 stereo camera.

Recommended (trained medium model, close-up rocks in front of camera):
  python scripts/inference_pi.py --model models/mars_rock_detector_m.pt --conf 0.35

Local display on Pi (HDMI monitor attached):
  python scripts/inference_pi.py --model models/mars_rock_detector_m.pt

Stream to your desk computer (open in browser):
  python scripts/inference_pi.py --model models/mars_rock_detector_m.pt --stream --host 0.0.0.0 --port 8080
  # On PC: http://<pi-ip-address>:8080

Test on PC first with the Pi-style eval clip:
  python scripts/detect_rocks.py --video Videos/eval/08_pi_closeup_rock_test.mp4 --model models/mars_rock_detector_m.pt

ONNX on Pi (after PT_to_ONNX.py on PC):
  python scripts/inference_pi.py --model models/mars_rock_detector_m.onnx --conf 0.35 --no-filters --stream

Note: YOLO11m is slower than nano on Pi 5. If FPS is too low, use mars_rock_detector.pt/.onnx instead.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_filters import is_likely_rock_box  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mars_rock_detector_m.pt"

# Latest annotated JPEG frame for MJPEG streaming.
_stream_lock = threading.Lock()
_stream_jpeg: bytes | None = None


def open_picamera2(width: int, height: int, camera_num: int):
    from picamera2 import Picamera2

    picam = Picamera2(camera_num)
    config = picam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1.0)

    def read():
        frame = picam.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release():
        picam.stop()
        picam.close()

    return read, release


def update_stream_frame(annotated_bgr) -> None:
    global _stream_jpeg
    ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    with _stream_lock:
        _stream_jpeg = buf.tobytes()


class MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # quiet logs while rover is running

    def do_GET(self):
        if self.path in ("/", "/stream"):
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()
            while True:
                with _stream_lock:
                    frame = _stream_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.033)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


def start_stream_server(host: str, port: int) -> HTTPServer:
    server = HTTPServer((host, port), MjpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Rock detection on Pi 5 + IMX219")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0.30–0.40 works well for close-up rocks on Pi)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Show all model boxes (skip color/shape filter — use for close-up Pi testing)",
    )
    parser.add_argument(
        "--camera-num",
        type=int,
        default=0,
        choices=(0, 1),
        help="IMX219 stereo: 0=left (CAM/DISP 0), 1=right (CAM/DISP 1)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="MJPEG stream for viewing on your desk computer",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Stream bind address")
    parser.add_argument("--port", type=int, default=8080, help="Stream port")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip local cv2 window (use with --stream on headless rover)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Copy models/mars_rock_detector_m.pt or .onnx to the Pi first.")
        sys.exit(1)

    try:
        read_frame, release = open_picamera2(args.width, args.height, args.camera_num)
    except ImportError:
        print("picamera2 is required on Raspberry Pi OS.")
        print("Install: sudo apt install -y python3-picamera2")
        print("For PC video testing use: python scripts/detect_rocks.py")
        sys.exit(1)
    except Exception as exc:
        print(f"Camera error: {exc}")
        print("Check /boot/firmware/config.txt has imx219 overlays for cam0/cam1.")
        sys.exit(1)

    model = YOLO(str(args.model))
    show_local = not args.no_display and not args.stream

    server = None
    if args.stream:
        server = start_stream_server(args.host, args.port)
        print(f"Stream live detections at: http://<pi-ip>:{args.port}/")
        print("Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = read_frame()
            if not ok:
                break
            results = model.predict(
                frame, conf=args.conf, iou=0.45, imgsz=640, verbose=False
            )
            annotated = frame.copy()
            h, w = annotated.shape[:2]
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if not args.no_filters and not is_likely_rock_box(
                        x1, y1, x2, y2, conf, w, h, min_conf=args.conf, img=frame
                    ):
                        continue
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"rock {conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            if args.stream:
                update_stream_frame(annotated)
            if show_local:
                cv2.imshow("Mars Rock Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        release()
        if show_local:
            cv2.destroyAllWindows()
        if server:
            server.shutdown()


if __name__ == "__main__":
    main()
