"""
live_camera.py — Live plant disease detection on Raspberry Pi.

Captures frames from the Pi Camera Module v2 (via picamera2) and overlays
the predicted disease class and confidence on each frame using OpenCV.

Falls back to cv2.VideoCapture(0) if picamera2 is not available
(useful for testing on a laptop with a USB webcam).

Inference runs on every Nth frame (INFER_EVERY_N_FRAMES) to maintain a
smooth display while keeping CPU load manageable on the Pi.

Usage:
    python3 deployment/live_camera.py

Requirements (Pi):
    onnxruntime, numpy, opencv-python, picamera2
    (install via: pip3 install -r deployment/requirements_pi.txt)

Press 'q' to quit.
"""

import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path so we can import inference.py sibling
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import PlantDiseaseClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INFER_EVERY_N_FRAMES: int = 3   # run inference once every N captured frames
DISPLAY_WIDTH:  int = 640
DISPLAY_HEIGHT: int = 480
WINDOW_TITLE:   str = "Plant Disease Detection — press Q to quit"

# Overlay styling
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.7
FONT_THICKNESS  = 2
TEXT_COLOR      = (255, 255, 255)   # white
BOX_COLOR       = (0, 150, 0)       # green background for normal
WARN_COLOR      = (0, 50, 200)      # orange-ish for disease
CONF_THRESHOLD  = 0.60              # below this, label shown in yellow

_DEPLOY_DIR = Path(__file__).resolve().parent
_PROJECT    = _DEPLOY_DIR.parent


# ---------------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------------

def open_picamera2(width: int, height: int):
    """Open Pi Camera via picamera2. Returns a Picamera2 instance or None."""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(1.0)  # let sensor settle
        return cam
    except Exception as e:
        print(f"[picamera2] Not available: {e}")
        return None


def capture_picamera2(cam) -> np.ndarray:
    """Capture one RGB frame from picamera2, return as BGR numpy array."""
    rgb = cam.capture_array()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def open_cv_camera(width: int, height: int):
    """Fallback: open default webcam via OpenCV."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera found via cv2.VideoCapture(0).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    inference_ms: float,
) -> np.ndarray:
    """Draw label, confidence, and latency overlay on the frame."""
    # Shorten label for display (strip species prefix like "Tomato___")
    short_label = label.split("___", 1)[1] if "___" in label else label
    short_label = short_label.replace("_", " ")

    conf_pct = confidence * 100
    color = BOX_COLOR if confidence >= CONF_THRESHOLD else (0, 200, 255)

    # Build text lines
    line1 = f"{short_label}"
    line2 = f"Conf: {conf_pct:.1f}%   Latency: {inference_ms:.0f}ms"

    # Measure text size to draw background rectangles
    (w1, h1), _ = cv2.getTextSize(line1, FONT, FONT_SCALE + 0.2, FONT_THICKNESS)
    (w2, h2), _ = cv2.getTextSize(line2, FONT, FONT_SCALE, FONT_THICKNESS)

    pad = 8
    rect_w = max(w1, w2) + pad * 2
    rect_h = h1 + h2 + pad * 3

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), color, thickness=-1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text
    cv2.putText(frame, line1, (pad, h1 + pad),
                FONT, FONT_SCALE + 0.2, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, line2, (pad, h1 + h2 + pad * 2),
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS - 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    print("Loading model...")
    classifier = PlantDiseaseClassifier()
    print(f"Classes: {classifier.class_names}")

    # Try picamera2 first, fall back to OpenCV webcam
    picam = open_picamera2(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    use_picam = picam is not None

    if not use_picam:
        print("Falling back to cv2.VideoCapture(0)...")
        cap = open_cv_camera(DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # State
    frame_idx = 0
    last_label = "Initializing..."
    last_confidence = 0.0
    last_latency_ms = 0.0

    print(f"Live feed started. Inferring every {INFER_EVERY_N_FRAMES} frames. Press Q to quit.")
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Capture frame
            if use_picam:
                frame = capture_picamera2(picam)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read error — retrying...")
                    continue

            # Run inference every N frames
            if frame_idx % INFER_EVERY_N_FRAMES == 0:
                t0 = time.perf_counter()
                last_label, last_confidence = classifier.predict(frame)
                last_latency_ms = (time.perf_counter() - t0) * 1000

            frame = draw_overlay(frame, last_label, last_confidence, last_latency_ms)
            cv2.imshow(WINDOW_TITLE, frame)

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        if use_picam:
            picam.stop()
        else:
            cap.release()
        print("Camera stopped.")


if __name__ == "__main__":
    run()
