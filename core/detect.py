"""Grounded-SAM pipeline: text prompt -> GroundingDINO boxes -> SAM 2 masks.

Chains the detector and the segmenter so that a text prompt produces precise,
cleaned-up masks automatically (no human clicking). Used by the CLI and, later, the
backend's auto-detect route.

CLI::

    python -m core.detect [image] --prompt tree --prompt "palm tree"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.detector import Box, GroundingDinoDetector
from core.mask_processor import filter_by_area, nms
from core.sam_inference import SAM2Segmenter
from utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected-and-segmented object."""

    mask: np.ndarray  # boolean H x W
    box: Box          # (x1, y1, x2, y2)
    label: str
    score: float


def grounded_sam(
    image_rgb: np.ndarray,
    detector: GroundingDinoDetector,
    segmenter: SAM2Segmenter,
    prompts: str | list[str] | None = None,
    config=None,
) -> list[Detection]:
    """Run the full text -> boxes -> masks pipeline on one image.

    Args:
        image_rgb: RGB ``uint8`` image (``H x W x 3``).
        detector: A loaded GroundingDINO detector.
        segmenter: A loaded SAM 2 segmenter.
        prompts: Text prompt(s); defaults to ``detector.prompts`` from config.
        config: Optional config; defaults to the project config.

    Returns:
        A list of :class:`Detection` objects (deduplicated and area-filtered).
    """
    cfg = config or load_config()
    boxes, scores, labels = detector.detect(image_rgb, prompts)
    if not boxes:
        logger.info("No detections for the given prompt.")
        return []

    keep = nms(boxes, scores, iou_threshold=cfg.get("mask.nms_iou_threshold", 0.5))
    min_area = cfg.get("mask.min_area", 100)

    segmenter.set_image(image_rgb)
    detections: list[Detection] = []
    for i in keep:
        mask, _ = segmenter.predict_box(boxes[i])
        if filter_by_area(mask, min_area):
            detections.append(Detection(mask=mask, box=boxes[i], label=labels[i], score=scores[i]))
    logger.info(
        "Grounded-SAM: %d boxes kept -> %d masks after filtering", len(keep), len(detections)
    )
    return detections


def _run(image_path: Path, prompts: list[str] | None) -> None:
    """Run Grounded-SAM on one image and save an overlay to the output dir."""
    import cv2

    from core.preprocessor import preprocess
    from utils.visualizer import draw_boxes, overlay_masks

    cfg = load_config()
    out_dir = cfg.path("paths.output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    bgr = preprocess(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    detector = GroundingDinoDetector()
    segmenter = SAM2Segmenter()
    detections = grounded_sam(rgb, detector, segmenter, prompts, cfg)

    overlay = overlay_masks(bgr, [d.mask for d in detections])
    overlay = draw_boxes(
        overlay,
        [d.box for d in detections],
        [f"{d.label} {d.score:.2f}" for d in detections],
    )
    out_path = out_dir / f"{image_path.stem}_grounded.jpg"
    cv2.imwrite(str(out_path), overlay)
    logger.info("Saved %s (%d objects)", out_path, len(detections))


def main() -> None:
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Grounded-SAM: detect + segment by text prompt.")
    parser.add_argument("image", nargs="?", help="Path to an image (defaults to a sample).")
    parser.add_argument("--prompt", action="append", help="Text prompt(s); repeatable.")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
    else:
        samples = sorted(load_config().path("paths.sample_images").glob("*.JPG"))
        if not samples:
            raise SystemExit("No image given and no samples found in data/sample_images/.")
        image_path = samples[0]
    _run(image_path, args.prompt)


if __name__ == "__main__":
    main()
