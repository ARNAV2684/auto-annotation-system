"""Inference wrapper for project-specific fine-tuned YOLOv8-seg checkpoints.

Projects with deployed trained weights (``trained_models:`` in ``config.yaml``)
run auto-detect through this wrapper instead of GroundingDINO + SAM: one model
call yields boxes, labels and masks together. Everything heavy (``ultralytics``,
``torch``) is imported lazily inside :class:`TrainedYoloSegmenter`, so this
module stays importable on GPU-free CI.

The model is fed the **raw** image, not the CLAHE-enhanced one the SAM pipeline
uses — it was fine-tuned on raw drone imagery, and preprocessed input would be
a train/serve mismatch. Resolution is untouched either way, so masks land in
the same original-image pixel space as every other annotation source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.mask_processor import clamp_box_to_image, polygon_to_mask
from core.types import Detection
from utils.config import load_config

logger = logging.getLogger(__name__)


def deployed_weights(project_id: str, config=None) -> Path | None:
    """Return the deployed checkpoint path for a project, or ``None``.

    ``None`` means "no trained model" — either the project has no
    ``trained_models:`` entry or the weights file has not been deployed yet —
    and the caller should fall back to the Grounded-SAM flow.
    """
    cfg = config or load_config()
    conf = (cfg.get("trained_models") or {}).get(project_id)
    if not conf or not conf.get("weights"):
        return None
    weights = Path(conf["weights"])
    if not weights.is_absolute():
        weights = cfg.root / weights
    return weights if weights.is_file() else None


def results_to_detections(result, width: int, height: int, min_area: int = 0) -> list[Detection]:
    """Convert one Ultralytics segmentation result into :class:`Detection` rows.

    Duck-typed on the small slice of the Results API we use (``masks.xy``,
    ``boxes.xyxy`` / ``cls`` / ``conf``, ``names``) so it is unit-testable
    without ``ultralytics`` installed. Instances with degenerate polygons,
    degenerate boxes, or masks below ``min_area`` pixels are dropped.
    """
    if result.masks is None or result.boxes is None:
        return []
    detections: list[Detection] = []
    for polygon, box, cls_idx, score in zip(
        result.masks.xy,
        result.boxes.xyxy,
        result.boxes.cls,
        result.boxes.conf,
        strict=True,
    ):
        points = [(float(x), float(y)) for x, y in np.asarray(polygon)]
        if len(points) < 3:
            continue
        mask = polygon_to_mask(points, width, height)
        if int(mask.sum()) < max(min_area, 1):
            continue
        clamped = clamp_box_to_image(
            (float(box[0]), float(box[1]), float(box[2]), float(box[3])), width, height
        )
        if clamped is None:
            continue
        detections.append(
            Detection(
                mask=mask,
                box=clamped,
                label=str(result.names[int(cls_idx)]),
                score=float(score),
            )
        )
    return detections


class TrainedYoloSegmenter:
    """A loaded fine-tuned YOLOv8-seg checkpoint for one project.

    Args:
        weights: Path to the deployed ``.pt`` checkpoint.
        conf_threshold: Minimum detector confidence to keep an instance.
        imgsz: Inference image size (should match the training ``imgsz``).
        min_area: Drop masks below this pixel area (mirrors the SAM pipeline).
    """

    def __init__(
        self,
        weights: Path,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        min_area: int = 100,
    ) -> None:
        from ultralytics import YOLO  # heavy; imported only when actually used

        self.model = YOLO(str(weights))
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.min_area = int(min_area)
        logger.info("Loaded trained YOLO-seg checkpoint: %s", weights)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run segmentation on one BGR image (cv2 convention) at full quality.

        Returns detections in the input image's pixel space.
        """
        height, width = image_bgr.shape[:2]
        results = self.model.predict(
            image_bgr, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False
        )
        detections = results_to_detections(
            results[0], width, height, min_area=self.min_area
        )
        logger.info("Trained YOLO-seg: %d instances kept", len(detections))
        return detections


def build_for_project(project_id: str, config=None) -> TrainedYoloSegmenter:
    """Construct a :class:`TrainedYoloSegmenter` from a project's config entry.

    Raises:
        FileNotFoundError: the project has no deployed weights.
    """
    cfg = config or load_config()
    weights = deployed_weights(project_id, cfg)
    if weights is None:
        raise FileNotFoundError(f"No deployed trained model for project '{project_id}'.")
    conf = (cfg.get("trained_models") or {})[project_id]
    return TrainedYoloSegmenter(
        weights,
        conf_threshold=float(conf.get("conf_threshold", 0.25)),
        imgsz=int(conf.get("imgsz", 640)),
        min_area=int(cfg.get("mask.min_area", 100)),
    )
