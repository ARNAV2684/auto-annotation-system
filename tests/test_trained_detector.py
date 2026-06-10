"""Unit tests for :mod:`core.trained_detector` (GPU-free).

:class:`TrainedYoloSegmenter` itself needs ``ultralytics`` + a checkpoint, so it
is exercised manually on the GPU machine. The conversion helper is duck-typed
on the Ultralytics Results API, so these tests drive it with plain namespaces.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.trained_detector import deployed_weights, results_to_detections
from utils.config import Config

_NAMES = {0: "cracks", 1: "potholes", 2: "ravelling"}


def _square(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _result(polygons, boxes, classes, scores, names=_NAMES) -> SimpleNamespace:
    return SimpleNamespace(
        masks=SimpleNamespace(xy=polygons),
        boxes=SimpleNamespace(xyxy=boxes, cls=classes, conf=scores),
        names=names,
    )


# ─── results_to_detections ────────────────────────────────────────────────


def test_converts_instances_to_detections() -> None:
    result = _result(
        polygons=[_square(2, 2, 12, 12), _square(20, 20, 30, 28)],
        boxes=[[2, 2, 12, 12], [20, 20, 30, 28]],
        classes=[0, 2],
        scores=[0.9, 0.6],
    )
    detections = results_to_detections(result, width=40, height=40)
    assert [d.label for d in detections] == ["cracks", "ravelling"]
    assert [d.score for d in detections] == [0.9, 0.6]
    assert detections[0].box == (2, 2, 12, 12)
    for d in detections:
        assert d.mask.shape == (40, 40) and d.mask.dtype == bool
        assert d.mask.sum() > 0


def test_degenerate_polygon_is_skipped() -> None:
    result = _result(
        polygons=[np.zeros((0, 2), dtype=np.float32), _square(2, 2, 12, 12)],
        boxes=[[0, 0, 5, 5], [2, 2, 12, 12]],
        classes=[1, 1],
        scores=[0.8, 0.7],
    )
    detections = results_to_detections(result, width=40, height=40)
    assert len(detections) == 1 and detections[0].label == "potholes"


def test_min_area_filter_drops_speck_masks() -> None:
    result = _result(
        polygons=[_square(0, 0, 2, 2), _square(5, 5, 30, 30)],
        boxes=[[0, 0, 2, 2], [5, 5, 30, 30]],
        classes=[0, 0],
        scores=[0.9, 0.9],
    )
    detections = results_to_detections(result, width=40, height=40, min_area=100)
    assert len(detections) == 1 and detections[0].box == (5, 5, 30, 30)


def test_box_is_clamped_to_image_bounds() -> None:
    result = _result(
        polygons=[_square(30, 30, 39, 39)],
        boxes=[[30, 30, 55, 61]],  # spills past the 40x40 image
        classes=[2],
        scores=[0.5],
    )
    detections = results_to_detections(result, width=40, height=40)
    assert detections[0].box == (30, 30, 40, 40)


def test_no_masks_yields_empty_list() -> None:
    empty = SimpleNamespace(masks=None, boxes=None, names=_NAMES)
    assert results_to_detections(empty, width=40, height=40) == []


# ─── deployed_weights ─────────────────────────────────────────────────────


def _cfg(trained_models: dict, root: Path) -> Config:
    return Config({"trained_models": trained_models}, root)


def test_deployed_weights_returns_existing_relative_path(tmp_path: Path) -> None:
    weights = tmp_path / "models" / "deployed.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake")
    cfg = _cfg({"roads": {"weights": "models/deployed.pt"}}, tmp_path)
    assert deployed_weights("roads", cfg) == weights


def test_deployed_weights_none_when_file_missing(tmp_path: Path) -> None:
    cfg = _cfg({"roads": {"weights": "models/deployed.pt"}}, tmp_path)
    assert deployed_weights("roads", cfg) is None


def test_deployed_weights_none_for_unconfigured_project(tmp_path: Path) -> None:
    cfg = _cfg({"roads": {"weights": "models/deployed.pt"}}, tmp_path)
    assert deployed_weights("vegetation", cfg) is None
    assert deployed_weights("roads", Config({}, tmp_path)) is None
