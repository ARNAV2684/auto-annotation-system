"""Lightweight, dependency-free type definitions shared across the pipeline.

Lives in its own module so utility code (e.g. ``annotation_converter``) can import
:class:`Detection` without dragging in heavy GPU dependencies (``torch``, ``sam2``,
``transformers``) that the GPU-free unit tests and the CI runner do not need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class Detection:
    """One detected-and-segmented object, passed through the pipeline.

    Attributes:
        mask: Boolean ``H x W`` segmentation mask.
        box: ``(x1, y1, x2, y2)`` pixel bounding box.
        label: Human-readable class label (e.g. ``"tree"``).
        score: Detector confidence in ``[0, 1]``.
        is_crowd: If ``True``, COCO writes RLE segmentation with ``iscrowd=1``.
    """

    mask: np.ndarray
    box: Box
    label: str
    score: float
    is_crowd: bool = False
