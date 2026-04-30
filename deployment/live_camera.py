"""
live_camera.py — Live plant disease detection on Raspberry Pi.

Captures frames from the Pi Camera Module v2 (via picamera2) and overlays
the predicted disease class and confidence on each frame using OpenCV.

Falls back to cv2.VideoCapture(0) if picamera2 is not available
(useful for testing on a laptop with a USB webcam).

Press 'q' to quit.
"""

import logging
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import PlantDiseaseClassifier

# ---------------------------------------------------------------------------
# Logging setup — writes to both stdout and a log file
# ---------------------------------------------------------------------------

_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "live_camera.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_FILE), mode="w"),
    ],
)
log = logging.getLogger("live_camera")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INFER_EVERY_N_FRAMES: int = 1
DISPLAY_WIDTH:  int = 640
DISPLAY_HEIGHT: int = 480
WINDOW_TITLE:   str = "Plant Disease Detection — press Q to quit"

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.7
FONT_THICKNESS = 2
TEXT_COLOR     = (255, 255, 255)
BOX_COLOR      = (0, 150, 0)
CONF_THRESHOLD = 0.60

_DEPLOY_DIR = Path(__file__).resolve().parent
_PROJECT    = _DEPLOY_DIR.parent


# ---------------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------------

def open_picamera2(width: int, height: int):
    """Open Pi Camera via picamera2. Returns Picamera2 instance or None."""
    log.info("Attempting to open camera via picamera2...")
    try:
        from picamera2 import Picamera2
        log.debug("picamera2 module imported successfully")

        cam = Picamera2()
        log.debug(f"Picamera2 instance created: {cam}")

        # Log available camera info
        camera_info = cam.camera_properties
        log.info(f"Camera properties: {camera_info}")

        config = cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        log.debug(f"Camera config: {config}")
        cam.configure(config)
        log.debug("Camera configured")

        cam.start()
        log.info(f"picamera2 started ({width}x{height} RGB888)")
        time.sleep(1.0)  # let sensor settle
        log.debug("Sensor settle delay complete")
        return cam

    except ImportError as e:
        log.warning(f"picamera2 not installed: {e}")
        return None
    except Exception as e:
        log.error(f"picamera2 failed to open: {e}")
        log.debug(traceback.format_exc())
        return None


def capture_picamera2(cam) -> np.ndarray | None:
    """Capture one RGB frame from picamera2, return as BGR numpy array."""
    try:
        rgb = cam.capture_array()
        log.debug(f"Captured frame: shape={rgb.shape} dtype={rgb.dtype}")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.error(f"picamera2 capture failed: {e}")
        log.debug(traceback.format_exc())
        return None


def open_cv_camera(width: int, height: int):
    """Fallback: open default webcam via OpenCV."""
    log.info("Attempting to open camera via cv2.VideoCapture(0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log.error("cv2.VideoCapture(0) failed to open — no camera detected")
        raise RuntimeError(
            "No camera found via cv2.VideoCapture(0).\n"
            "Check: is the camera connected? Is it in use by another process?\n"
            "Run: libcamera-hello --list-cameras"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    backend  = cap.getBackendName()
    log.info(f"cv2.VideoCapture opened: {actual_w}x{actual_h} via {backend}")
    return cap


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    inference_ms: float,
) -> np.ndarray:
    short_label = label.split("___", 1)[1] if "___" in label else label
    short_label = short_label.replace("_", " ")

    conf_pct = confidence * 100
    color = BOX_COLOR if confidence >= CONF_THRESHOLD else (0, 200, 255)

    line1 = f"{short_label}"
    line2 = f"Conf: {conf_pct:.1f}%   Latency: {inference_ms:.0f}ms"

    (w1, h1), _ = cv2.getTextSize(line1, FONT, FONT_SCALE + 0.2, FONT_THICKNESS)
    (w2, h2), _ = cv2.getTextSize(line2, FONT, FONT_SCALE, FONT_THICKNESS)

    pad    = 8
    rect_w = max(w1, w2) + pad * 2
    rect_h = h1 + h2 + pad * 3

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), color, thickness=-1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, line1, (pad, h1 + pad),
                FONT, FONT_SCALE + 0.2, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, line2, (pad, h1 + h2 + pad * 2),
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS - 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    log.info("=== live_camera.py starting ===")
    log.info(f"Log file: {_LOG_FILE}")

    # Load model
    log.info("Loading classifier model...")
    try:
        classifier = PlantDiseaseClassifier()
        log.info(f"Model loaded. Classes: {classifier.class_names}")
    except Exception as e:
        log.critical(f"Failed to load model: {e}")
        log.debug(traceback.format_exc())
        sys.exit(1)

    # Open camera
    picam = open_picamera2(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    use_picam = picam is not None

    if not use_picam:
        log.warning("picamera2 unavailable — falling back to cv2.VideoCapture(0)")
        try:
            cap = open_cv_camera(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        except RuntimeError as e:
            log.critical(str(e))
            sys.exit(1)

    # State
    frame_idx        = 0
    read_errors      = 0
    MAX_READ_ERRORS  = 10
    last_label       = "Initializing..."
    last_confidence  = 0.0
    last_latency_ms  = 0.0

    log.info(f"Live feed started — inferring every {INFER_EVERY_N_FRAMES} frame(s). Press Q to quit.")
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # --- Capture ---
            if use_picam:
                frame = capture_picamera2(picam)
                if frame is None:
                    read_errors += 1
                    log.warning(f"picamera2 capture returned None (error #{read_errors})")
                    if read_errors >= MAX_READ_ERRORS:
                        log.critical(f"Too many consecutive capture failures ({MAX_READ_ERRORS}). Exiting.")
                        break
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    read_errors += 1
                    log.warning(
                        f"cv2.VideoCapture.read() failed (error #{read_errors}) — "
                        f"ret={ret}, frame={'None' if frame is None else frame.shape}"
                    )
                    if read_errors >= MAX_READ_ERRORS:
                        log.critical(f"Too many consecutive read failures ({MAX_READ_ERRORS}). Exiting.")
                        break
                    time.sleep(0.05)
                    continue

            read_errors = 0  # reset on successful read
            log.debug(f"Frame {frame_idx}: shape={frame.shape} dtype={frame.dtype}")

            # --- Inference ---
            if frame_idx % INFER_EVERY_N_FRAMES == 0:
                try:
                    t0 = time.perf_counter()
                    last_label, last_confidence = classifier.predict(frame)
                    last_latency_ms = (time.perf_counter() - t0) * 1000
                    log.debug(
                        f"Frame {frame_idx}: pred={last_label} "
                        f"conf={last_confidence:.3f} latency={last_latency_ms:.1f}ms"
                    )
                except Exception as e:
                    log.error(f"Inference failed on frame {frame_idx}: {e}")
                    log.debug(traceback.format_exc())

            # --- Display ---
            frame = draw_overlay(frame, last_label, last_confidence, last_latency_ms)
            cv2.imshow(WINDOW_TITLE, frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit key pressed")
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        log.critical(f"Unexpected error in main loop: {e}")
        log.debug(traceback.format_exc())
    finally:
        log.info(f"Shutting down after {frame_idx} frames")
        cv2.destroyAllWindows()
        if use_picam:
            picam.stop()
            log.info("picamera2 stopped")
        else:
            cap.release()
            log.info("cv2 capture released")
        log.info(f"Log saved to: {_LOG_FILE}")


if __name__ == "__main__":
    run()
