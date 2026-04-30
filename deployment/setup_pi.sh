#!/usr/bin/env bash
# setup_pi.sh — Raspberry Pi 5 environment setup for plant disease detection.
#
# Run once on the Pi after cloning the repo:
#   bash deployment/setup_pi.sh
#
# Prerequisites:
#   - Raspberry Pi OS Bookworm (Python 3.11 pre-installed)
#   - Internet connection (or mobile hotspot) during setup only
#   - Camera Module connected via CSI

set -euo pipefail

echo "=== Pi Environment Setup ==="

# System packages
sudo apt update -y
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev   # required by numpy on ARM

# picamera2 (may already be installed on Bookworm)
sudo apt install -y python3-picamera2 || echo "[SKIP] picamera2 already installed or unavailable"

# Python packages
pip3 install --upgrade pip
pip3 install -r "$(dirname "$0")/requirements_pi.txt"

echo ""
echo "=== Verifying installation ==="
python3 -c "import onnxruntime; print(f'onnxruntime {onnxruntime.__version__} OK')"
python3 -c "import cv2; print(f'opencv {cv2.__version__} OK')"
python3 -c "import numpy; print(f'numpy {numpy.__version__} OK')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run live inference:"
echo "  python3 deployment/live_camera.py"
echo ""
echo "To test on a single image:"
echo "  python3 deployment/inference.py path/to/leaf.jpg"
