"""
inference.py — Single-image inference using ONNX Runtime.

Loads the INT8 quantized ONNX model and runs inference on a single image,
returning the top-1 class label and confidence score.

Usage:
    python3 deployment/inference.py path/to/leaf.jpg

Requirements (Pi):
    onnxruntime, numpy, pillow, opencv-python
    (install via: pip3 install -r deployment/requirements_pi.txt)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths — relative to this file's location (deployment/)
# ---------------------------------------------------------------------------

_DEPLOY_DIR = Path(__file__).resolve().parent
_PROJECT    = _DEPLOY_DIR.parent
_MODELS_DIR = _PROJECT / "models"

ONNX_MODEL  = _MODELS_DIR / "plant_disease.onnx"
LABELS_FILE = _MODELS_DIR / "labels.txt"

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class PlantDiseaseClassifier:
    """
    Wraps the ONNX Runtime session and label list for repeated inference calls.

    The session is created once at init time for efficiency in the live camera loop.
    """

    def __init__(
        self,
        model_path: Path = ONNX_MODEL,
        labels_path: Path = LABELS_FILE,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Run src/quantize.py on the training machine and copy the .onnx file."
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._class_names = [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Prepare an OpenCV BGR frame for inference.

        CRITICAL: This must exactly match the val_transform pipeline used during
        training (resize → ToTensor [/255] → ImageNet normalise → NCHW).

        Args:
            bgr_frame: uint8 numpy array (H, W, 3) in BGR order from cv2.

        Returns:
            float32 numpy array (1, 3, H, W) ready for ONNX Runtime.
        """
        # BGR → RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        resized = cv2.resize(rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        # uint8 [0, 255] → float32 [0, 1] → CHW
        arr = resized.astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        # ImageNet normalisation
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr[np.newaxis, ...]  # (1, 3, H, W)

    def predict(self, bgr_frame: np.ndarray) -> tuple[str, float]:
        """
        Run inference on one frame.

        Args:
            bgr_frame: uint8 numpy array (H, W, 3) from OpenCV (BGR).

        Returns:
            (class_name, confidence) — top-1 prediction with softmax confidence.
        """
        inp = self.preprocess(bgr_frame)
        logits = self._session.run(None, {self._input_name: inp})[0][0]

        # Convert logits to probabilities with softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        top_idx = int(np.argmax(probs))
        return self._class_names[top_idx], float(probs[top_idx])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 deployment/inference.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: could not read image: {image_path}")
        sys.exit(1)

    classifier = PlantDiseaseClassifier()
    label, confidence = classifier.predict(frame)

    print(f"Image     : {image_path.name}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.1f}%")


if __name__ == "__main__":
    main()
