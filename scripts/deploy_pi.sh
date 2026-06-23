#!/bin/bash
# Raspberry Pi 5 deployment helper — run ON the Pi after copying project files.
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Mars Rock Detector — Pi 5 Setup ==="

# 1. System packages
sudo apt update
sudo apt install -y python3-pip python3-venv python3-picamera2 libcamera-apps

# 2. Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python-headless

# 3. Verify cameras (requires imx219 overlays in /boot/firmware/config.txt)
echo ""
echo "Checking cameras ..."
if command -v rpicam-hello &>/dev/null; then
    rpicam-hello --list-cameras || true
elif command -v libcamera-hello &>/dev/null; then
    libcamera-hello --list-cameras || true
fi

# 4. Verify model
MODEL="models/mars_rock_detector.pt"
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Copy $MODEL from your PC to the Pi first."
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Stream to your desk PC (rover moving):"
echo "  source .venv/bin/activate"
echo "  python scripts/inference_pi.py --stream --no-display --camera-num 0"
echo "  # Browser on PC: http://<pi-ip>:8080/"
echo ""
echo "Local HDMI preview on Pi:"
echo "  python scripts/inference_pi.py --camera-num 0"
echo ""
echo "Use right stereo camera:"
echo "  python scripts/inference_pi.py --stream --camera-num 1"
