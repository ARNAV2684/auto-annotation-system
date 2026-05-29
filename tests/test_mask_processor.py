"""Tests for mask/box post-processing (``core.mask_processor``). GPU-free; runs in CI."""

import numpy as np

from core.mask_processor import (
    clamp_box_to_image,
    compute_iou,
    filter_by_area,
    mask_to_bbox,
    mask_to_polygon,
    nms,
)


def test_iou_identical_boxes_is_one() -> None:
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap_is_between() -> None:
    assert 0.0 < compute_iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_nms_suppresses_overlapping_lower_score() -> None:
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (100, 100, 110, 110)]
    scores = [0.9, 0.8, 0.7]
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_nms_empty_input() -> None:
    assert nms([], []) == []


def test_filter_by_area() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3, :3] = True  # area = 9
    assert filter_by_area(mask, 5) is True
    assert filter_by_area(mask, 20) is False


def test_mask_to_polygon_returns_valid_polygon() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    polygons = mask_to_polygon(mask)
    assert len(polygons) >= 1
    assert len(polygons[0]) >= 6  # at least 3 (x, y) points


def test_mask_to_bbox_of_square() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:16] = True  # rows 5..14, cols 5..15
    assert mask_to_bbox(mask) == (5.0, 5.0, 15.0, 14.0)


def test_mask_to_bbox_empty_is_none() -> None:
    assert mask_to_bbox(np.zeros((5, 5), dtype=bool)) is None


# ─────────────────────────────────────────────────────────────────────────────
# clamp_box_to_image  (guards against GroundingDINO out-of-bounds predictions)
# ─────────────────────────────────────────────────────────────────────────────

def test_clamp_box_passes_through_in_bounds() -> None:
    assert clamp_box_to_image((10.0, 20.0, 100.0, 200.0), 1280, 720) == (10, 20, 100, 200)


def test_clamp_box_clips_overshoot_to_image_bounds() -> None:
    # ymax overshoots height (1299 > 1280) — exactly the bug from the 24-image batch run.
    assert clamp_box_to_image((62.0, 1100.0, 369.0, 1299.0), 1280, 1280) == (62, 1100, 369, 1280)


def test_clamp_box_clips_negative_origin_to_zero() -> None:
    assert clamp_box_to_image((-5.0, -10.0, 50.0, 60.0), 1280, 720) == (0, 0, 50, 60)


def test_clamp_box_returns_none_for_fully_outside() -> None:
    # Both edges past the right side -> after clamp both become 1280 -> degenerate.
    assert clamp_box_to_image((1290.0, 100.0, 1300.0, 200.0), 1280, 720) is None


def test_clamp_box_returns_none_for_inverted_box() -> None:
    # x2 < x1 even before clamping
    assert clamp_box_to_image((100.0, 50.0, 90.0, 60.0), 1280, 720) is None


def test_clamp_box_handles_float_input_truncating_to_int() -> None:
    # 1279.99 should truncate to 1279 (below width 1280)
    assert clamp_box_to_image((10.5, 20.5, 1279.99, 200.4), 1280, 720) == (10, 20, 1279, 200)
