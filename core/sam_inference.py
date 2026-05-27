"""SAM 2 inference engine.

Wraps Meta's Segment Anything Model 2 to provide the three prompt modes the annotation
tool needs:

* **automatic** - segment everything in the image (``SAM2AutomaticMaskGenerator``)
* **point** - a click ``(x, y)`` -> mask (``SAM2ImagePredictor``)
* **box** - a rough box ``(x1, y1, x2, y2)`` -> refined mask

The expensive image embedding is computed once per image (via :meth:`set_image`); each
subsequent point/box prompt on that image is then cheap and near-instant.

Run the CLI smoke test::

    python -m core.sam_inference [path/to/drone_image.jpg]
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.config import load_config
from utils.device import resolve_device

logger = logging.getLogger(__name__)


class SAM2Segmenter:
    """Loads SAM 2 once and exposes automatic / point / box segmentation."""

    def __init__(self, config=None, device: str | None = None) -> None:
        """Build the SAM 2 model and predictor on the chosen device.

        Args:
            config: Optional :class:`~utils.config.Config`; defaults to the project config.
            device: Optional device override (``"cuda"``/``"cpu"``); defaults to auto.
        """
        self.cfg = config or load_config()
        self.device = resolve_device(device)
        checkpoint = str(self.cfg.path("sam.checkpoint"))
        model_cfg = self.cfg.get("sam.model_cfg")
        logger.info("Loading SAM 2 (%s) on %s", model_cfg, self.device)
        self.model = build_sam2(model_cfg, checkpoint, device=self.device)
        self._predictor = SAM2ImagePredictor(self.model)
        self._auto_generator: SAM2AutomaticMaskGenerator | None = None
        self._image_is_set = False

    @contextlib.contextmanager
    def _inference_context(self):
        """Run inference under ``inference_mode`` (and bfloat16 autocast on CUDA)."""
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Compute and cache the image embedding for subsequent point/box prompts.

        Args:
            image_rgb: RGB ``uint8`` image (``H x W x 3``).
        """
        with self._inference_context():
            self._predictor.set_image(image_rgb)
        self._image_is_set = True
        logger.debug("Image embedding computed: shape=%s", image_rgb.shape)

    def predict_point(
        self, point_xy: tuple[int, int], label: int = 1, multimask: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment the object at a clicked point. Requires :meth:`set_image` first.

        Args:
            point_xy: ``(x, y)`` pixel coordinate of the click.
            label: 1 for a foreground point, 0 for background.
            multimask: If ``True``, return up to 3 candidate masks ranked by score.

        Returns:
            ``(masks, scores)`` sorted best-first; ``masks`` is ``(N, H, W)`` bool.
        """
        self._require_image()
        with self._inference_context():
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([point_xy], dtype=np.float32),
                point_labels=np.array([label], dtype=np.int32),
                multimask_output=multimask,
            )
        order = np.argsort(scores)[::-1]
        return masks[order].astype(bool), scores[order]

    def predict_box(
        self, box_xyxy: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Segment the object within a bounding box. Requires :meth:`set_image` first.

        Args:
            box_xyxy: ``(x1, y1, x2, y2)`` pixel box.

        Returns:
            ``(mask, score)`` where ``mask`` is a single ``H x W`` bool array.
        """
        self._require_image()
        with self._inference_context():
            masks, scores, _ = self._predictor.predict(
                box=np.array(box_xyxy, dtype=np.float32),
                multimask_output=False,
            )
        return masks[0].astype(bool), float(scores[0])

    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """Automatically segment everything in the image.

        Args:
            image_rgb: RGB ``uint8`` image (``H x W x 3``).

        Returns:
            A list of mask records, each with ``segmentation`` (bool ``H x W``),
            ``bbox`` (``[x, y, w, h]``), ``area``, ``predicted_iou`` and
            ``stability_score`` keys.
        """
        if self._auto_generator is None:
            auto = self.cfg.get("sam.auto", {}) or {}
            self._auto_generator = SAM2AutomaticMaskGenerator(
                self.model,
                points_per_side=auto.get("points_per_side", 32),
                pred_iou_thresh=auto.get("pred_iou_thresh", 0.85),
                stability_score_thresh=auto.get("stability_score_thresh", 0.92),
                min_mask_region_area=auto.get("min_mask_region_area", 100),
            )
        with self._inference_context():
            return self._auto_generator.generate(image_rgb)

    def _require_image(self) -> None:
        if not self._image_is_set:
            raise RuntimeError("Call set_image() before point/box prediction.")


def _smoke_test(image_path: Path) -> None:
    """Run a quick end-to-end check: load image -> SAM 2 -> save overlays."""
    import cv2

    from core.preprocessor import preprocess
    from utils.visualizer import overlay_masks

    cfg = load_config()
    output_dir = cfg.path("paths.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    bgr = preprocess(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    segmenter = SAM2Segmenter()
    logger.info("Device: %s", segmenter.device)

    # 1) Automatic "segment everything"
    records = segmenter.generate(rgb)
    logger.info("Automatic mode: %d masks", len(records))
    top = sorted(records, key=lambda r: r["area"], reverse=True)[:40]
    auto_overlay = overlay_masks(bgr, [r["segmentation"] for r in top])
    auto_path = output_dir / f"{image_path.stem}_auto.jpg"
    cv2.imwrite(str(auto_path), auto_overlay)

    # 2) Point prompt at the image centre
    h, w = rgb.shape[:2]
    segmenter.set_image(rgb)
    masks, scores = segmenter.predict_point((w // 2, h // 2))
    logger.info("Point mode: best score=%.3f", float(scores[0]))
    point_overlay = overlay_masks(bgr, [masks[0]])
    point_path = output_dir / f"{image_path.stem}_point.jpg"
    cv2.imwrite(str(point_path), point_overlay)

    if torch.cuda.is_available():
        logger.info("Peak VRAM: %.0f MB", torch.cuda.max_memory_allocated() / 1e6)
    logger.info("Saved overlays:\n  %s\n  %s", auto_path, point_path)


def main() -> None:
    """CLI entry point for the smoke test."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SAM 2 smoke test on a drone image.")
    parser.add_argument("image", nargs="?", help="Path to an image (defaults to a sample).")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
    else:
        samples = sorted(load_config().path("paths.sample_images").glob("*.JPG"))
        if not samples:
            raise SystemExit("No image given and no samples found in data/sample_images/.")
        image_path = samples[0]
    _smoke_test(image_path)


if __name__ == "__main__":
    main()
