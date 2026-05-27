"""Mask and box post-processing.

Cleans up raw segmentation output before it is shown for review or exported:
filter out tiny noise masks, de-duplicate overlapping detections with IoU-based
non-maximum suppression (NMS), and convert masks into simplified polygons (needed for
COCO-style export later).

IoU/NMS helpers are ported from the GarudaUAV tree-counter prototype and cleaned up.
Masks are boolean ``H x W`` arrays; boxes are ``(x1, y1, x2, y2)`` in pixels.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-Union of two ``(x1, y1, x2, y2)`` boxes.

    Args:
        box_a: First box.
        box_b: Second box.

    Returns:
        IoU in ``[0, 1]``; 0 if the boxes do not overlap.
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    inter = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    boxes: list[Box], scores: list[float], iou_threshold: float = 0.5
) -> list[int]:
    """Greedy non-maximum suppression: keep high-score boxes, drop overlapping ones.

    Args:
        boxes: Candidate boxes.
        scores: Confidence score per box (same length as ``boxes``).
        iou_threshold: Boxes overlapping a kept box above this IoU are suppressed.

    Returns:
        Indices of the boxes to keep, highest score first.
    """
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for idx in order:
        if all(compute_iou(boxes[idx], boxes[k]) <= iou_threshold for k in keep):
            keep.append(idx)
    return keep


def filter_by_area(mask: np.ndarray, min_area: int) -> bool:
    """Return ``True`` if a mask's pixel area meets the minimum threshold.

    Args:
        mask: Boolean ``H x W`` mask.
        min_area: Minimum number of foreground pixels to keep the mask.

    Returns:
        Whether the mask is large enough to keep.
    """
    return int(mask.sum()) >= min_area


def mask_to_polygon(mask: np.ndarray, epsilon: float = 2.0) -> list[list[int]]:
    """Convert a boolean mask into simplified polygon contour(s).

    Uses ``cv2.findContours`` (external contours) then Douglas-Peucker simplification
    to reduce point count while preserving shape.

    Args:
        mask: Boolean ``H x W`` mask.
        epsilon: Approximation tolerance in pixels (larger = fewer points).

    Returns:
        A list of polygons; each polygon is a flat ``[x1, y1, x2, y2, ...]`` list.
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[int]] = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) >= 3:  # need at least a triangle
            polygons.append(approx.reshape(-1).astype(int).tolist())
    return polygons


def mask_to_bbox(mask: np.ndarray) -> Box | None:
    """Return the tight ``(x1, y1, x2, y2)`` bounding box of a mask, or ``None`` if empty.

    Args:
        mask: Boolean ``H x W`` mask.

    Returns:
        The bounding box of the foreground pixels, or ``None`` if the mask is empty.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
