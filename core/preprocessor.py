"""Image preprocessing for drone imagery.

Ported and cleaned from the GarudaUAV tree-counter prototype. Drone images are often
washed out with uneven lighting; CLAHE contrast enhancement plus light unsharp
sharpening make objects more separable for downstream segmentation.

All functions operate on OpenCV **BGR** ``uint8`` arrays (``H x W x 3``) and preserve
the input resolution by default, so masks produced from the result map directly back
to the original image. SAM 2 performs its own internal resize/normalisation, so we do
not downscale before inference unless a caller explicitly asks via
:func:`resize_longest_edge`.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from utils.config import load_config

logger = logging.getLogger(__name__)


def resize_longest_edge(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize an image so its longest edge equals ``target_size``, keeping aspect ratio.

    Args:
        image: BGR ``uint8`` image.
        target_size: Desired length of the longest edge, in pixels.

    Returns:
        The resized image, or the original if it already matches.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest == target_size:
        return image
    scale = target_size / longest
    new_size = (int(width * scale), int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, new_size, interpolation=interpolation)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """Apply CLAHE to the L channel of the image in LAB colour space.

    Equalises local contrast without distorting colour, fixing the washed-out look of
    aerial imagery.

    Args:
        image: BGR ``uint8`` image.
        clip_limit: Contrast-limiting threshold (higher = stronger enhancement).
        tile_size: Side length of the local equalisation grid, in tiles.

    Returns:
        The contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray, strength: float) -> np.ndarray:
    """Sharpen an image with a 3x3 unsharp-mask kernel, blended by ``strength``.

    Args:
        image: BGR ``uint8`` image.
        strength: Blend weight in ``[0, 1]``; 0 returns the original, 1 is fully sharpened.

    Returns:
        The sharpened BGR image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)


def preprocess(image: np.ndarray, enhance: bool = True) -> np.ndarray:
    """Enhance a drone image for segmentation, preserving its resolution.

    Applies CLAHE then unsharp sharpening using parameters from ``config.yaml``
    (``preprocess.*``). Resolution is preserved so masks map back to the original image.

    Args:
        image: BGR ``uint8`` image.
        enhance: If ``False``, return the image unchanged (escape hatch for debugging).

    Returns:
        The enhanced BGR image (same shape as the input).
    """
    if not enhance:
        return image
    cfg = load_config()
    enhanced = apply_clahe(
        image,
        clip_limit=cfg.get("preprocess.clahe_clip_limit", 2.0),
        tile_size=cfg.get("preprocess.clahe_tile_size", 8),
    )
    enhanced = sharpen(enhanced, strength=cfg.get("preprocess.sharpen_strength", 0.3))
    logger.debug("Preprocessed image: shape=%s", enhanced.shape)
    return enhanced
