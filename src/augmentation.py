"""
augmentation.py — Augmentation pipeline for plant disease detection training.

Defines a torchvision-compatible augmentation pipeline that simulates
real-world field conditions: varied lighting, camera angle, and leaf orientation.

All augmentations operate on PIL Images (before ToTensor) or tensors
normalised with ImageNet mean/std.

Usage (from train.py):
    from augmentation import build_train_transform, build_val_transform
    train_dataset = ImageFolder(root=..., transform=build_train_transform())
    val_dataset   = ImageFolder(root=..., transform=build_val_transform())
"""

from torchvision import transforms

# ---------------------------------------------------------------------------
# Normalisation constants (ImageNet statistics — matches pretrained weights)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Augmentation parameters
# ---------------------------------------------------------------------------

ROTATION_DEGREES  = 30
BRIGHTNESS_FACTOR = 0.2
CONTRAST_FACTOR   = 0.3
ZOOM_SCALE        = (0.9, 1.0)   # random crop range as fraction of image


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_train_transform() -> transforms.Compose:
    """
    Build the training augmentation + normalisation pipeline.

    Operations applied in order:
      1. Random horizontal and vertical flip
      2. Random rotation ±30°
      3. Random resized crop (simulates zoom)
      4. Brightness and contrast jitter
      5. ToTensor (HWC uint8 → CHW float32 [0, 1])
      6. ImageNet normalisation

    Returns:
        torchvision.transforms.Compose for training data.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=ZOOM_SCALE),
        transforms.ColorJitter(
            brightness=BRIGHTNESS_FACTOR,
            contrast=CONTRAST_FACTOR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform() -> transforms.Compose:
    """
    Build the validation/test/inference normalisation pipeline (no augmentation).

    Operations applied in order:
      1. Resize to IMAGE_SIZE
      2. ToTensor (HWC uint8 → CHW float32 [0, 1])
      3. ImageNet normalisation

    Returns:
        torchvision.transforms.Compose for validation/test/inference data.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_for_inference(image_array, target_size: tuple[int, int] = IMAGE_SIZE):
    """
    Preprocess a numpy uint8 HWC image (H, W, 3) for ONNX Runtime inference.

    CRITICAL: This must exactly match the val_transform pipeline used during
    training. Any mismatch will silently degrade inference accuracy.

    Args:
        image_array: numpy array, shape (H, W, 3), dtype uint8, values 0-255.
        target_size: (H, W) the model expects.

    Returns:
        numpy float32 array of shape (1, 3, H, W) — batch dimension + NCHW.
    """
    import cv2
    import numpy as np

    # Resize to model input size
    resized = cv2.resize(image_array, (target_size[1], target_size[0]))

    # HWC uint8 → CHW float32 [0, 1]
    arr = resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW

    # ImageNet normalisation
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    arr  = (arr - mean) / std

    return arr[np.newaxis, ...]  # add batch dimension: (1, 3, H, W)
