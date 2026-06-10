"""Lazy-singleton registry for the heavy GPU models.

SAM 2 and GroundingDINO each take a couple of seconds to load and use VRAM, so we
load them **once** at FastAPI app startup and reuse the same instances across every
request. The registry also tracks **which image is currently embedded in SAM** so
back-to-back point/box prompts on the same image skip the ~0.5 s ``set_image`` cost.

The HTTP layer talks to this registry through the FastAPI dependency injected by
:func:`get_model_registry`. Tests override that dependency with a mock so CI never
touches torch / sam2 / transformers.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

# Lazy: the heavy GPU modules (torch / sam2 / transformers) are imported INSIDE
# methods, never at module top, so test code and CI can import this file without
# the GPU stack installed. The dependency-override in tests means ModelRegistry
# is never actually instantiated in CI.
from core.preprocessor import preprocess

if TYPE_CHECKING:
    from core.trained_detector import TrainedYoloSegmenter
    from core.types import Detection

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds the loaded models + the "which image is hot in SAM" cache.

    All inference methods are coarse-grained (one call = one user action) so the
    surface the API layer talks to stays small.
    """

    def __init__(self) -> None:
        # Heavy imports happen ONLY when we actually instantiate the registry.
        # CI uses a stub registry (dependency override), so torch never imports there.
        from core.detector import GroundingDinoDetector
        from core.sam_inference import SAM2Segmenter

        logger.info("Loading SAM 2 ...")
        self.segmenter = SAM2Segmenter()
        logger.info("Loading GroundingDINO ...")
        self.detector = GroundingDinoDetector()
        self._current_image_id: str | None = None
        self._current_rgb: np.ndarray | None = None
        # Fine-tuned per-project YOLO-seg models, loaded lazily on first use
        # (a yolov8n-seg checkpoint is ~7 MB — cheap next to SAM + DINO).
        # Cached together with the weights file's mtime so a redeploy (eval
        # gate or manual rollback) is picked up without a server restart.
        self._trained: dict[str, tuple[TrainedYoloSegmenter, float]] = {}
        self._lock = threading.Lock()
        logger.info(
            "Models loaded (segmenter on %s, detector on %s).",
            self.segmenter.device,
            self.detector.device,
        )

    # ─── image loading + embedding cache ────────────────────────────────

    def load_image_rgb(self, image_path: Path) -> np.ndarray:
        """Read an image from disk, preprocess (CLAHE+sharpen), return RGB ``uint8``."""
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        bgr = preprocess(bgr)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def ensure_embedded(self, image_id: str, image_path: Path) -> bool:
        """Make sure SAM has this image's embedding loaded.

        Args:
            image_id: Server-side id for this image.
            image_path: On-disk path; only read if we need to (re)embed.

        Returns:
            ``True`` if the embedding was already cached (no work done),
            ``False`` if it was computed just now.
        """
        with self._lock:
            if self._current_image_id == image_id and self._current_rgb is not None:
                return True
            rgb = self.load_image_rgb(image_path)
            self.segmenter.set_image(rgb)
            self._current_image_id = image_id
            self._current_rgb = rgb
            return False

    def current_image_size(self) -> tuple[int, int] | None:
        """Return ``(width, height)`` of the currently embedded image, if any."""
        if self._current_rgb is None:
            return None
        height, width = self._current_rgb.shape[:2]
        return width, height

    # ─── inference: point / box / auto ──────────────────────────────────

    def predict_point(
        self, point_xy: tuple[float, float], label: int, multimask: bool
    ) -> tuple[list[np.ndarray], list[float]]:
        """Return ``(masks_best_first, scores_best_first)`` from a point click."""
        with self._lock:
            masks, scores = self.segmenter.predict_point(
                (int(point_xy[0]), int(point_xy[1])), label=label, multimask=multimask
            )
        return [m for m in masks], [float(s) for s in scores]

    def predict_points(
        self,
        points: list[tuple[float, float]],
        labels: list[int],
        multimask: bool = False,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Return ``(masks_best_first, scores_best_first)`` from several points."""
        with self._lock:
            masks, scores = self.segmenter.predict_points(
                [(int(x), int(y)) for x, y in points], labels, multimask=multimask
            )
        return [m for m in masks], [float(s) for s in scores]

    def predict_box(self, box: tuple[float, float, float, float]) -> tuple[np.ndarray, float]:
        """Return ``(mask, score)`` from a single box prompt."""
        with self._lock:
            mask, score = self.segmenter.predict_box(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            )
        return mask, score

    def auto_detect(self, image_path: Path, prompts: list[str]) -> list[Detection]:
        """Run the full Grounded-SAM chain (text -> boxes -> masks) on one image."""
        from core.detect import grounded_sam  # lazy; pulls torch transitively

        rgb = self.load_image_rgb(image_path)
        with self._lock:
            detections = grounded_sam(rgb, self.detector, self.segmenter, prompts)
            # The chain above calls set_image() internally; reflect that here.
            self._current_image_id = None  # we lost track of which id is loaded
            self._current_rgb = rgb
        return detections

    def auto_detect_trained(self, image_path: Path, project_id: str) -> list[Detection]:
        """Run a project's fine-tuned YOLO-seg model instead of Grounded-SAM.

        The image goes in raw (no CLAHE/sharpen): the model was fine-tuned on
        raw drone imagery, so enhanced input would be a train/serve mismatch.
        Resolution is identical either way, so masks share the same pixel
        space as SAM-produced annotations.
        """
        from core.trained_detector import (  # lazy; pulls ultralytics
            build_for_project,
            deployed_weights,
        )

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        weights = deployed_weights(project_id)
        if weights is None:
            raise FileNotFoundError(f"No deployed trained model for '{project_id}'.")
        mtime = weights.stat().st_mtime
        with self._lock:
            cached = self._trained.get(project_id)
            if cached is None or cached[1] != mtime:
                cached = (build_for_project(project_id), mtime)
                self._trained[project_id] = cached
            return cached[0].detect(bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor (used by FastAPI dependency injection)
# ─────────────────────────────────────────────────────────────────────────────

_registry: ModelRegistry | None = None


def init_model_registry() -> ModelRegistry:
    """Build the registry (call once at startup)."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_model_registry() -> ModelRegistry:
    """FastAPI dependency: return the loaded registry, building it on demand.

    Tests override this with a mock so the GPU stack never imports in CI.
    """
    return init_model_registry()
