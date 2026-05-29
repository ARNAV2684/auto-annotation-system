"""Tests for the annotation converters.

GPU-free; runs in CI. Uses synthetic ``Detection`` objects with known masks/boxes so
coordinates can be asserted exactly. Each writer has at least 5 unit tests (brief §8.1),
and each format also has a round-trip test (export -> re-parse -> match).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from core.annotation_converter import (
    CocoWriter,
    ExportImage,
    VocWriter,
    YoloWriter,
    _normalise_label,
    build_class_index,
    decode_coco_rle,
    encode_coco_rle,
    resolve_class_id,
)
from core.types import Detection
from utils.validators import (
    ValidationError,
    validate_coco_file,
    validate_voc_file,
    validate_yolo_file,
)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test fixture: a 100x100 image with two known square detections
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["tree", "palm_tree", "shrub"]


def _square_mask(height: int, width: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


@pytest.fixture
def sample_image() -> ExportImage:
    """A 100x100 image with two known square detections (one normal, one crowd)."""
    height, width = 100, 100
    mask_tree = _square_mask(height, width, 10, 30, 10, 30)        # 20x20 at (10,10)
    mask_palm = _square_mask(height, width, 60, 80, 60, 80)        # 20x20 at (60,60)
    detections = [
        Detection(mask=mask_tree, box=(10, 10, 30, 30), label="tree", score=0.9),
        Detection(
            mask=mask_palm, box=(60, 60, 80, 80), label="palm tree", score=0.8, is_crowd=True
        ),
    ]
    return ExportImage(file_name="test.jpg", width=width, height=height, detections=detections)


# ─────────────────────────────────────────────────────────────────────────────
# Class index + label normalisation
# ─────────────────────────────────────────────────────────────────────────────

def test_build_class_index_assigns_zero_based_ids() -> None:
    assert build_class_index(CLASSES) == {"tree": 0, "palm_tree": 1, "shrub": 2}


def test_normalise_label_handles_space_and_case() -> None:
    assert _normalise_label("Palm Tree") == "palm_tree"
    assert _normalise_label("  TREE ") == "tree"


# ─────────────────────────────────────────────────────────────────────────────
# COCO RLE encoder / decoder
# ─────────────────────────────────────────────────────────────────────────────

def test_rle_all_false_is_single_background_run() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    rle = encode_coco_rle(mask)
    assert rle == {"size": [4, 4], "counts": [16]}


def test_rle_all_true_starts_with_zero_then_full_run() -> None:
    mask = np.ones((3, 5), dtype=bool)
    rle = encode_coco_rle(mask)
    assert rle == {"size": [3, 5], "counts": [0, 15]}


def test_rle_column_major_order() -> None:
    # 2x2 with TL=True, others False -> column-major flat: [1, 0, 0, 0]
    mask = np.array([[True, False], [False, False]])
    rle = encode_coco_rle(mask)
    # Leading 0 (zero bg) + 1 fg + 3 bg
    assert rle == {"size": [2, 2], "counts": [0, 1, 3]}


def test_rle_roundtrip_recovers_arbitrary_mask() -> None:
    rng = np.random.default_rng(42)
    mask = rng.random((37, 53)) > 0.4
    rle = encode_coco_rle(mask)
    decoded = decode_coco_rle(rle)
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, mask)


def test_rle_counts_sum_equals_image_pixels() -> None:
    mask = _square_mask(20, 25, 5, 15, 7, 17)
    rle = encode_coco_rle(mask)
    assert sum(rle["counts"]) == 20 * 25


# ─────────────────────────────────────────────────────────────────────────────
# YOLO writer (segmentation by default)
# ─────────────────────────────────────────────────────────────────────────────

def test_yolo_writes_one_txt_per_image_and_classes(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    writer = YoloWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    assert (tmp_path / "test.txt").exists()
    assert (tmp_path / "classes.txt").read_text(encoding="utf-8").splitlines() == CLASSES


def test_yolo_segmentation_line_starts_with_correct_class_id(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    writer = YoloWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    text = (tmp_path / "test.txt").read_text(encoding="utf-8")
    class_ids = {int(line.split()[0]) for line in text.strip().splitlines()}
    assert class_ids == {0, 1}  # tree, palm_tree


def test_yolo_segmentation_coords_are_normalised(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    writer = YoloWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    for line in (tmp_path / "test.txt").read_text(encoding="utf-8").strip().splitlines():
        coords = [float(v) for v in line.split()[1:]]
        assert len(coords) >= 6 and len(coords) % 2 == 0
        assert all(0.0 <= v <= 1.0 for v in coords)


def test_yolo_detection_mode_writes_four_normalised_coords(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    writer = YoloWriter(tmp_path, CLASSES, mode="detection")
    writer.add_image(sample_image)
    writer.finalize()
    lines = (tmp_path / "test.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # one per detection
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        cx, cy, w, h = (float(v) for v in parts[1:])
        assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))


def test_yolo_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="YOLO mode"):
        YoloWriter(Path("."), CLASSES, mode="bogus")


def test_yolo_unknown_label_is_skipped(tmp_path: Path) -> None:
    mask = _square_mask(100, 100, 10, 30, 10, 30)
    image = ExportImage(
        file_name="x.jpg",
        width=100,
        height=100,
        detections=[Detection(mask=mask, box=(10, 10, 30, 30), label="alien", score=0.5)],
    )
    writer = YoloWriter(tmp_path, CLASSES)
    writer.add_image(image)
    writer.finalize()
    assert (tmp_path / "x.txt").read_text(encoding="utf-8").strip() == ""


def test_yolo_roundtrip_passes_validator(tmp_path: Path, sample_image: ExportImage) -> None:
    writer = YoloWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    assert validate_yolo_file(tmp_path / "test.txt", num_classes=len(CLASSES)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# COCO writer (polygon + RLE)
# ─────────────────────────────────────────────────────────────────────────────

def _write_coco(tmp_path: Path, sample_image: ExportImage, crowd_classes=None) -> dict:
    writer = CocoWriter(tmp_path, CLASSES, crowd_classes=crowd_classes or [])
    writer.add_image(sample_image)
    writer.finalize()
    return json.loads((tmp_path / "annotations.json").read_text(encoding="utf-8"))


def test_coco_structure_has_required_top_level_keys(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    data = _write_coco(tmp_path, sample_image)
    assert set(data.keys()) >= {"images", "annotations", "categories"}
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 2


def test_coco_categories_use_config_class_order(tmp_path: Path, sample_image: ExportImage) -> None:
    data = _write_coco(tmp_path, sample_image)
    assert [(c["id"], c["name"]) for c in data["categories"]] == [
        (0, "tree"), (1, "palm_tree"), (2, "shrub")
    ]


def test_coco_bbox_is_xywh_absolute_not_xyxy(tmp_path: Path, sample_image: ExportImage) -> None:
    """COCO bbox is [x, y, width, height] in absolute pixels, NOT [x1, y1, x2, y2]."""
    data = _write_coco(tmp_path, sample_image)
    tree = next(a for a in data["annotations"] if a["category_id"] == 0)
    x, y, w, h = tree["bbox"]
    # Tree mask was rows 10..29, cols 10..29 -> bbox tight to mask
    assert 0.0 <= x <= 11.0 and 0.0 <= y <= 11.0
    assert 18.0 <= w <= 21.0 and 18.0 <= h <= 21.0


def test_coco_non_crowd_uses_polygon_segmentation(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    data = _write_coco(tmp_path, sample_image)
    tree = next(a for a in data["annotations"] if a["category_id"] == 0)
    assert tree["iscrowd"] == 0
    assert isinstance(tree["segmentation"], list)
    assert all(isinstance(poly, list) and len(poly) >= 6 for poly in tree["segmentation"])


def test_coco_crowd_detection_uses_rle_segmentation(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    data = _write_coco(tmp_path, sample_image)
    palm = next(a for a in data["annotations"] if a["category_id"] == 1)
    assert palm["iscrowd"] == 1
    seg = palm["segmentation"]
    assert isinstance(seg, dict)
    assert seg["size"] == [100, 100]
    assert isinstance(seg["counts"], list)


def test_coco_rle_roundtrip_recovers_mask(tmp_path: Path, sample_image: ExportImage) -> None:
    data = _write_coco(tmp_path, sample_image)
    palm = next(a for a in data["annotations"] if a["category_id"] == 1)
    decoded = decode_coco_rle(palm["segmentation"])
    expected = _square_mask(100, 100, 60, 80, 60, 80)
    assert np.array_equal(decoded, expected)


def test_coco_crowd_classes_config_flips_iscrowd(tmp_path: Path) -> None:
    """A normal detection becomes crowd if its class is in coco.crowd_classes."""
    mask = _square_mask(100, 100, 10, 30, 10, 30)
    image = ExportImage(
        file_name="x.jpg",
        width=100,
        height=100,
        detections=[
            Detection(mask=mask, box=(10, 10, 30, 30), label="tree", score=0.9, is_crowd=False)
        ],
    )
    data = _write_coco(tmp_path, image, crowd_classes=["tree"])
    ann = data["annotations"][0]
    assert ann["iscrowd"] == 1
    assert isinstance(ann["segmentation"], dict)


def test_coco_area_equals_mask_pixel_count(tmp_path: Path, sample_image: ExportImage) -> None:
    data = _write_coco(tmp_path, sample_image)
    tree = next(a for a in data["annotations"] if a["category_id"] == 0)
    assert tree["area"] == pytest.approx(400.0)  # 20x20 mask


def test_coco_roundtrip_passes_validator(tmp_path: Path, sample_image: ExportImage) -> None:
    writer = CocoWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    num_images, num_annotations = validate_coco_file(tmp_path / "annotations.json")
    assert (num_images, num_annotations) == (1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Pascal VOC writer
# ─────────────────────────────────────────────────────────────────────────────

def _voc_root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def test_voc_root_tag_is_annotation(tmp_path: Path, sample_image: ExportImage) -> None:
    writer = VocWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    root = _voc_root(tmp_path / "test.xml")
    assert root.tag == "annotation"
    assert root.findtext("filename") == "test.jpg"


def test_voc_size_block_has_width_height_depth(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    writer = VocWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    size = _voc_root(tmp_path / "test.xml").find("size")
    assert size is not None
    assert int(size.findtext("width", "0")) == 100
    assert int(size.findtext("height", "0")) == 100
    assert size.findtext("depth") == "3"


def test_voc_writes_one_object_per_detection(
    tmp_path: Path, sample_image: ExportImage
) -> None:
    writer = VocWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    objects = _voc_root(tmp_path / "test.xml").findall("object")
    assert len(objects) == 2
    names = {obj.findtext("name") for obj in objects}
    assert names == {"tree", "palm_tree"}


def test_voc_bndbox_ordering_is_valid(tmp_path: Path, sample_image: ExportImage) -> None:
    writer = VocWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    for obj in _voc_root(tmp_path / "test.xml").findall("object"):
        bnd = obj.find("bndbox")
        assert bnd is not None
        xmin = int(bnd.findtext("xmin", "0"))
        ymin = int(bnd.findtext("ymin", "0"))
        xmax = int(bnd.findtext("xmax", "0"))
        ymax = int(bnd.findtext("ymax", "0"))
        assert xmin < xmax and ymin < ymax


def test_voc_database_from_config(tmp_path: Path, sample_image: ExportImage) -> None:
    writer = VocWriter(tmp_path, CLASSES, database="TestDB")
    writer.add_image(sample_image)
    writer.finalize()
    source = _voc_root(tmp_path / "test.xml").find("source")
    assert source is not None
    assert source.findtext("database") == "TestDB"


def test_voc_roundtrip_passes_validator(tmp_path: Path, sample_image: ExportImage) -> None:
    writer = VocWriter(tmp_path, CLASSES)
    writer.add_image(sample_image)
    writer.finalize()
    assert validate_voc_file(tmp_path / "test.xml") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Validators: rejection cases
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_yolo_rejects_out_of_range_class_id(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("5 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_yolo_file(bad, num_classes=3)


def test_validate_yolo_rejects_unnormalised_coords(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("0 1.2 0.3 0.4 0.5 0.6 0.7\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_yolo_file(bad, num_classes=3)


def test_validate_coco_rejects_unknown_category(tmp_path: Path) -> None:
    bad = {
        "images": [{"id": 1, "file_name": "x.jpg", "width": 10, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 99, "bbox": [0, 0, 1, 1], "area": 1.0,
             "iscrowd": 0, "segmentation": [[0, 0, 1, 0, 1, 1]]}
        ],
        "categories": [{"id": 0, "name": "tree", "supercategory": "object"}],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_coco_file(p)


def test_validate_voc_rejects_inverted_bndbox(tmp_path: Path) -> None:
    bad = """<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>x.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object>
    <name>tree</name>
    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
  </object>
</annotation>
"""
    p = tmp_path / "bad.xml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_voc_file(p)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_class_id  (recovers detections like "tree palm tree", "tree palm")
# ─────────────────────────────────────────────────────────────────────────────

INDEX = build_class_index(CLASSES)


def test_resolve_class_id_exact_match_after_normalisation() -> None:
    assert resolve_class_id("tree", INDEX) == 0
    assert resolve_class_id("Palm Tree", INDEX) == 1
    assert resolve_class_id("  PALM TREE  ", INDEX) == 1


def test_resolve_class_id_prefers_longest_substring_match() -> None:
    # 'tree palm tree' contains both 'tree' and 'palm tree'; pick the longer.
    assert resolve_class_id("tree palm tree", INDEX) == 1  # palm_tree


def test_resolve_class_id_falls_back_to_shorter_when_longer_absent() -> None:
    # 'tree palm' does NOT contain 'palm tree' (wrong order); falls back to 'tree'.
    assert resolve_class_id("tree palm", INDEX) == 0


def test_resolve_class_id_returns_none_for_unknown() -> None:
    # 'palm' alone matches neither 'palm_tree' (needs 'palm tree') nor 'tree' nor 'shrub'.
    assert resolve_class_id("palm", INDEX) is None
    assert resolve_class_id("vehicle", INDEX) is None


def test_resolve_class_id_handles_underscore_or_space_classes() -> None:
    # palm_tree class can be reached via space form 'palm tree'
    assert resolve_class_id("palm tree", INDEX) == 1
    # exact underscore form also works
    assert resolve_class_id("palm_tree", INDEX) == 1
