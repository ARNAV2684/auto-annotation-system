"""Visualisation helpers: overlay SAM masks and boxes onto images for review.

Operates on OpenCV **BGR** ``uint8`` arrays. Used by the CLI smoke test now and by the
backend/UI later to render annotation previews.
"""

from __future__ import annotations

import numpy as np

# Distinct, repeatable colour palette (BGR) so the same mask index keeps its colour.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (66, 135, 245), (245, 66, 96), (66, 245, 96), (245, 209, 66),
    (164, 66, 245), (66, 245, 230), (245, 135, 66), (138, 245, 66),
)


def _color_for(index: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a given mask index."""
    return _PALETTE[index % len(_PALETTE)]


def overlay_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend boolean masks onto an image as translucent coloured regions.

    Args:
        image: BGR ``uint8`` image to draw on (not modified in place).
        masks: List of boolean arrays, each shaped like the image's ``H x W``.
        alpha: Opacity of the coloured overlay in ``[0, 1]``.

    Returns:
        A new BGR image with the masks blended in.
    """
    out = image.copy()
    for i, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        color = np.array(_color_for(i), dtype=np.uint8)
        out[mask_bool] = (alpha * color + (1 - alpha) * out[mask_bool]).astype(np.uint8)
    return out


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes (and optional labels) on an image.

    Args:
        image: BGR ``uint8`` image (not modified in place).
        boxes: List of ``(x1, y1, x2, y2)`` integer boxes.
        labels: Optional per-box text labels.

    Returns:
        A new BGR image with boxes drawn.
    """
    import cv2

    out = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = _color_for(i)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if labels and i < len(labels):
            cv2.putText(
                out, labels[i], (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out
