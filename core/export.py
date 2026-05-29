"""End-to-end annotation export CLI.

Runs the Grounded-SAM chain on one or more images, then exports the resulting masks
to YOLO, COCO, and Pascal VOC under a timestamped run directory, with validation
and a stats summary.

CLI::

    python -m core.export [image_or_dir] --prompt tree [--prompt "palm tree"]
                          [--format yolo|coco|voc|all] [--no-zip]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from core.annotation_converter import ExportImage, build_writers
from core.detect import grounded_sam
from core.detector import GroundingDinoDetector
from core.preprocessor import preprocess
from core.sam_inference import SAM2Segmenter
from utils.config import load_config
from utils.file_handler import make_run_dir, write_stats, zip_directory
from utils.validators import validate_coco_file, validate_voc_file, validate_yolo_file

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _gather_images(image_or_dir: Path) -> list[Path]:
    """Return a sorted list of image paths from a file or a directory."""
    if image_or_dir.is_file():
        return [image_or_dir]
    if image_or_dir.is_dir():
        return sorted(p for p in image_or_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(image_or_dir)


def _run_pipeline_on_image(
    path: Path,
    detector: GroundingDinoDetector,
    segmenter: SAM2Segmenter,
    prompts: list[str] | None,
    cfg,
) -> ExportImage:
    """Read an image, run Grounded-SAM, return an :class:`ExportImage`."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    bgr = preprocess(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detections = grounded_sam(rgb, detector, segmenter, prompts, cfg)
    height, width = rgb.shape[:2]
    return ExportImage(file_name=path.name, width=width, height=height, detections=detections)


def _validate_outputs(
    formats: list[str], writers, run_dir: Path, num_classes: int, yolo_mode: str
) -> None:
    """Re-parse exported files to catch coordinate/structure bugs at export time."""
    if "yolo" in formats:
        yolo_dir = writers["yolo"].output_dir
        for txt in yolo_dir.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            validate_yolo_file(txt, num_classes=num_classes, mode=yolo_mode)
    if "coco" in formats:
        coco_path = writers["coco"].output_dir / "annotations.json"
        if coco_path.exists():
            validate_coco_file(coco_path)
    if "voc" in formats:
        for xml in writers["voc"].output_dir.glob("*.xml"):
            validate_voc_file(xml)
    logger.info("Validation passed for: %s", ", ".join(formats))


def run_export(
    images: list[Path],
    prompts: list[str] | None,
    formats: list[str],
    config=None,
    make_zip: bool = True,
) -> Path:
    """Run the full pipeline + export on a list of images.

    Args:
        images: Image paths.
        prompts: Text prompt(s) for GroundingDINO; ``None`` uses the configured defaults.
        formats: Subset of ``{"yolo", "coco", "voc"}`` to write.
        config: Optional :class:`~utils.config.Config`.
        make_zip: If ``True``, also produce ``exports.zip`` alongside the run folder.

    Returns:
        The run directory containing all exports.
    """
    cfg = config or load_config()
    run_dir = make_run_dir(cfg.path("paths.output_dir") / "exports")
    writers = build_writers(run_dir, formats=formats, config=cfg)

    detector = GroundingDinoDetector(config=cfg)
    segmenter = SAM2Segmenter(config=cfg)

    all_detections = []
    export_images: list[ExportImage] = []
    for path in images:
        export_image = _run_pipeline_on_image(path, detector, segmenter, prompts, cfg)
        export_images.append(export_image)
        all_detections.extend(export_image.detections)
        for writer in writers.values():
            writer.add_image(export_image)

    for writer in writers.values():
        writer.finalize()

    write_stats(run_dir / "stats.json", export_images, all_detections)
    _validate_outputs(
        list(writers.keys()),
        writers,
        run_dir,
        num_classes=len(cfg.get("classes", [])),
        yolo_mode=cfg.get("yolo.mode", "segmentation"),
    )

    if make_zip:
        # Zip lives as a SIBLING of the run dir (never inside the dir being archived).
        zip_directory(run_dir, run_dir.parent / run_dir.name)
    logger.info("Export complete: %s", run_dir)
    return run_dir


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run Grounded-SAM and export annotations.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Image file or directory (defaults to data/sample_images/).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Text prompt for the detector; repeatable.",
    )
    parser.add_argument(
        "--format",
        choices=["yolo", "coco", "voc", "all"],
        default="all",
        help="Annotation format(s) to write.",
    )
    parser.add_argument("--no-zip", action="store_true", help="Skip packaging exports.zip.")
    args = parser.parse_args()

    cfg = load_config()
    if args.image:
        images = _gather_images(Path(args.image))
    else:
        images = _gather_images(cfg.path("paths.sample_images"))
    if not images:
        raise SystemExit("No images found to export.")

    formats = ["yolo", "coco", "voc"] if args.format == "all" else [args.format]
    run_export(images, args.prompt, formats, cfg, make_zip=not args.no_zip)


if __name__ == "__main__":
    main()
