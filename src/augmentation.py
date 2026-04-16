"""
augmentation.py — Augmentation pipeline for plant disease detection training.

Defines a TensorFlow tf.data-compatible augmentation pipeline that simulates
real-world field conditions: varied lighting, camera angle, and leaf orientation.

All augmentations operate on float32 tensors in [0, 1] range.

Usage (from train.py):
    from src.augmentation import build_augmentation_pipeline
    augment_fn = build_augmentation_pipeline()
    train_ds = train_ds.map(lambda x, y: (augment_fn(x), y))
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# Augmentation parameters
# ---------------------------------------------------------------------------

# Rotation range in radians (~±30 degrees)
ROTATION_FACTOR = 0.083   # 30/360

# Brightness and contrast jitter range
BRIGHTNESS_DELTA = 0.2
CONTRAST_LOWER   = 0.7
CONTRAST_UPPER   = 1.3

# Zoom range: fraction of total dimension to crop
ZOOM_FACTOR = 0.1

# Gaussian noise standard deviation
NOISE_STDDEV = 0.02


# ---------------------------------------------------------------------------
# Individual augmentation ops
# ---------------------------------------------------------------------------

def random_flip(image: tf.Tensor) -> tf.Tensor:
    """Apply random horizontal and vertical flips."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def random_rotation(image: tf.Tensor) -> tf.Tensor:
    """
    Rotate image by a random angle in [-30°, +30°].

    NOTE: This standalone function is kept for reference but should NOT be
    called inside tf.data.map() — doing so re-instantiates the Keras layer on
    every image, breaking graph tracing.  Use build_augmentation_pipeline()
    instead, which creates the layer once and reuses it across all images.
    """
    rotator = tf.keras.layers.RandomRotation(
        factor=ROTATION_FACTOR,
        fill_mode="reflect",
        interpolation="bilinear",
    )
    image = rotator(tf.expand_dims(image, 0), training=True)
    return tf.squeeze(image, 0)


def random_brightness_contrast(image: tf.Tensor) -> tf.Tensor:
    """Apply random brightness delta and contrast scaling."""
    image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
    image = tf.image.random_contrast(image, lower=CONTRAST_LOWER, upper=CONTRAST_UPPER)
    return tf.clip_by_value(image, 0.0, 1.0)


def random_zoom(image: tf.Tensor, target_size: tuple[int, int] = (224, 224)) -> tf.Tensor:
    """
    Randomly zoom in by cropping a central region then resizing back.
    Simulates variable camera distance from the leaf.
    """
    h, w = target_size
    # Choose a crop size between (1-zoom_factor)*full and full size
    min_crop = int(h * (1.0 - ZOOM_FACTOR))
    crop_size = tf.random.uniform((), minval=min_crop, maxval=h, dtype=tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, [h, w], method="bilinear")
    return image


def random_gaussian_noise(image: tf.Tensor) -> tf.Tensor:
    """Add zero-mean Gaussian noise to simulate sensor noise."""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=NOISE_STDDEV)
    return tf.clip_by_value(image + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_augmentation_pipeline(
    use_rotation: bool = True,
    use_zoom: bool = True,
    use_noise: bool = False,
    target_size: tuple[int, int] = (224, 224),
) -> callable:
    """
    Build and return a single augmentation function suitable for tf.data.map().

    The RandomRotation Keras layer is instantiated ONCE here (not inside the
    returned closure) so that graph tracing works correctly when this function
    is used inside tf.map_fn or tf.data.Dataset.map().  Instantiating Keras
    preprocessing layers per image breaks graph tracing and reproducibility.

    Args:
        use_rotation: Apply random ±30° rotation (default True).
        use_zoom:     Apply random zoom crop (default True).
        use_noise:    Apply Gaussian noise (default False — off by default
                      to keep training stable; enable for extra robustness).
        target_size:  (H, W) of output images.

    Returns:
        A callable that takes a float32 image tensor [H, W, 3] in [0,1]
        and returns an augmented tensor of the same shape.
    """
    # Instantiate the rotation layer once here so it is shared across all
    # images in the tf.data pipeline, avoiding repeated layer creation that
    # breaks TF graph tracing.
    rotator = tf.keras.layers.RandomRotation(
        factor=ROTATION_FACTOR, fill_mode="reflect", interpolation="bilinear"
    )

    def augment(image: tf.Tensor) -> tf.Tensor:
        image = random_flip(image)
        image = random_brightness_contrast(image)
        if use_rotation:
            image = tf.squeeze(rotator(tf.expand_dims(image, 0), training=True), 0)
        if use_zoom:
            image = random_zoom(image, target_size=target_size)
        if use_noise:
            image = random_gaussian_noise(image)
        return image

    return augment


# ---------------------------------------------------------------------------
# Normalisation helper (used in both training and inference)
# ---------------------------------------------------------------------------

def normalize(image: tf.Tensor) -> tf.Tensor:
    """
    Scale pixel values from uint8 [0, 255] to float32 [0, 1].

    CRITICAL: This exact normalisation MUST be replicated in inference.py
    and live_camera.py. Any mismatch will silently degrade accuracy.
    """
    return tf.cast(image, tf.float32) / 255.0


def preprocess_for_inference(image: tf.Tensor, target_size: tuple[int, int] = (224, 224)) -> tf.Tensor:
    """
    Resize and normalise a single image for TFLite inference.

    Args:
        image: uint8 or float tensor of any spatial size, 3 channels.
        target_size: (H, W) the model expects.

    Returns:
        float32 tensor [1, H, W, 3] in [0, 1], with batch dimension added.
    """
    image = tf.image.resize(image, target_size, method="bilinear")
    image = normalize(image)
    return tf.expand_dims(image, 0)
