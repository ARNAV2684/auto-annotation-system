"""Filesystem helpers for export runs: timestamped run dirs, stats, and zip packaging."""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def make_run_dir(root: Path, prefix: str = "run") -> Path:
    """Create and return a fresh timestamped subdirectory under ``root``.

    The folder name is timestamped to the second. If two runs land in the same
    second (a fast double-click, back-to-back tests, or concurrent users), a
    numeric suffix (``_1``, ``_2`` …) is appended so each run still gets its own
    directory instead of raising ``FileExistsError``.

    Args:
        root: Parent directory (created if missing).
        prefix: Prefix for the new folder name.

    Returns:
        The created directory (e.g. ``outputs/exports/run_20260529_184501``).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suffix = 0
    while True:
        name = base if suffix == 0 else f"{base}_{suffix}"
        run_dir = root / name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1


def write_stats(path: Path, images, detections) -> None:
    """Write a small ``stats.json`` summarising an export run.

    Args:
        path: Output path.
        images: Iterable of :class:`~core.annotation_converter.ExportImage`.
        detections: Iterable of :class:`~core.detect.Detection`.
    """
    class_counter: Counter = Counter()
    crowd_counter = 0
    areas: list[float] = []
    for det in detections:
        class_counter[det.label] += 1
        if det.is_crowd:
            crowd_counter += 1
        areas.append(float(int(det.mask.sum())))
    stats = {
        "num_images": len(list(images)),
        "num_detections": len(areas),
        "num_crowd_detections": crowd_counter,
        "per_class_counts": dict(class_counter),
        "area_summary": {
            "min": min(areas) if areas else 0.0,
            "max": max(areas) if areas else 0.0,
            "mean": (sum(areas) / len(areas)) if areas else 0.0,
        },
    }
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote stats.json: %d images, %d detections", stats["num_images"], len(areas))


def zip_directory(directory: Path, zip_path: Path) -> Path:
    """Create ``zip_path.zip`` packaging the entire ``directory`` (no archive root).

    Args:
        directory: Directory to compress.
        zip_path: Target zip path **without** the ``.zip`` extension (``shutil`` adds it).

    Returns:
        The created ``.zip`` path.
    """
    directory = Path(directory)
    archive = shutil.make_archive(str(zip_path), "zip", root_dir=str(directory))
    logger.info("Packaged %s -> %s", directory, archive)
    return Path(archive)
