"""Unit tests for the before/after matcher in :mod:`training.compare_baseline`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.types import Detection
from training.compare_baseline import (
    load_ground_truth,
    mask_iou,
    match_instances,
    summarise,
)


def _det(label: str, y0: int, y1: int, score: float = 0.9, size: int = 40) -> Detection:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, y0:y1] = True
    return Detection(mask=mask, box=(y0, y0, y1, y1), label=label, score=score)


def test_mask_iou_identical_and_disjoint() -> None:
    a = _det("x", 0, 10).mask
    b = _det("x", 20, 30).mask
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == 0.0
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 0.0


def test_match_counts_tp_fp_fn() -> None:
    truths = [_det("cracks", 0, 10), _det("potholes", 20, 30)]
    predictions = [
        _det("cracks", 0, 10),  # perfect match -> TP
        _det("cracks", 30, 39),  # no overlapping crack -> FP
    ]
    tallies = match_instances(truths, predictions)
    assert tallies["cracks"] == {"tp": 1, "fp": 1, "fn": 0}
    assert tallies["potholes"] == {"tp": 0, "fp": 0, "fn": 1}  # missed entirely


def test_match_requires_same_class() -> None:
    truths = [_det("cracks", 0, 10)]
    predictions = [_det("potholes", 0, 10)]  # right place, wrong label
    tallies = match_instances(truths, predictions)
    assert tallies["cracks"]["fn"] == 1
    assert tallies["potholes"]["fp"] == 1


def test_each_truth_matches_at_most_once() -> None:
    truths = [_det("cracks", 0, 10)]
    predictions = [_det("cracks", 0, 10, score=0.9), _det("cracks", 0, 10, score=0.5)]
    tallies = match_instances(truths, predictions)
    assert tallies["cracks"] == {"tp": 1, "fp": 1, "fn": 0}


def test_summarise_precision_recall_f1() -> None:
    rows = summarise({"cracks": {"tp": 3, "fp": 1, "fn": 1}})
    label, precision, recall, f1 = rows[0]
    assert label == "cracks"
    assert precision == 0.75 and recall == 0.75 and f1 == 0.75
    assert rows[-1][0] == "all"


def test_load_ground_truth_denormalises_polygons(tmp_path: Path) -> None:
    label_file = tmp_path / "img.txt"
    # One pothole: a square covering 25%-75% of a 40x40 image.
    label_file.write_text("1 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    truths = load_ground_truth(
        label_file, ["cracks", "potholes", "ravelling"], width=40, height=40
    )
    assert len(truths) == 1
    assert truths[0].label == "potholes"
    assert truths[0].mask.shape == (40, 40)
    assert 350 < truths[0].mask.sum() < 500  # ~20x20 square
