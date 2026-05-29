"""Export-file validators.

Re-parse each annotation file we just wrote and assert it is structurally sound and
coordinates fall in the expected ranges. The brief flags wrong coordinates as a
"High"-severity risk (§10): downstream training silently corrupts on bad numbers,
so we catch it at export time, not weeks later.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when an exported annotation file fails validation."""


# ─────────────────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────────────────

def validate_yolo_file(path: Path, num_classes: int, mode: str = "segmentation") -> int:
    """Re-parse a YOLO ``.txt``; return the number of valid lines.

    Args:
        path: Path to the ``.txt`` file.
        num_classes: Total class count; class ids must be in ``[0, num_classes)``.
        mode: ``"segmentation"`` (polygon coords) or ``"detection"`` (cx cy w h).

    Raises:
        ValidationError: If any line has the wrong shape, out-of-range class id, or
        coordinates outside ``[0, 1]``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    line_count = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValidationError(f"{path}:{lineno}: non-integer class id") from exc
        if not (0 <= class_id < num_classes):
            raise ValidationError(
                f"{path}:{lineno}: class id {class_id} out of range [0, {num_classes})"
            )
        coords = [float(x) for x in parts[1:]]
        if mode == "detection":
            if len(coords) != 4:
                raise ValidationError(
                    f"{path}:{lineno}: detection needs 4 coords, got {len(coords)}"
                )
        else:
            if len(coords) < 6 or len(coords) % 2 != 0:
                raise ValidationError(
                    f"{path}:{lineno}: segmentation needs an even count >= 6 of coords"
                )
        for value in coords:
            if not (0.0 <= value <= 1.0):
                raise ValidationError(
                    f"{path}:{lineno}: coord {value} outside normalised range [0, 1]"
                )
        line_count += 1
    return line_count


# ─────────────────────────────────────────────────────────────────────────────
# COCO
# ─────────────────────────────────────────────────────────────────────────────

def validate_coco_file(path: Path) -> tuple[int, int]:
    """Re-parse a COCO ``annotations.json``; return ``(num_images, num_annotations)``.

    Raises:
        ValidationError: On missing keys, unknown category ids, or bad bboxes.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValidationError(f"{path}: missing top-level key '{key}'")
    image_ids = {image["id"] for image in data["images"]}
    category_ids = {cat["id"] for cat in data["categories"]}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            raise ValidationError(f"annotation {ann['id']}: unknown image_id {ann['image_id']}")
        if ann["category_id"] not in category_ids:
            raise ValidationError(
                f"annotation {ann['id']}: unknown category_id {ann['category_id']}"
            )
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValidationError(f"annotation {ann['id']}: bbox must be [x, y, w, h]")
        x, y, w, h = bbox
        if w < 0 or h < 0:
            raise ValidationError(f"annotation {ann['id']}: negative bbox width/height")
        if x < 0 or y < 0:
            raise ValidationError(f"annotation {ann['id']}: bbox origin must be >= 0")
        seg = ann.get("segmentation")
        if ann.get("iscrowd") == 1:
            if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
                raise ValidationError(
                    f"annotation {ann['id']}: crowd needs RLE segmentation"
                )
        else:
            if not isinstance(seg, list):
                raise ValidationError(
                    f"annotation {ann['id']}: non-crowd needs polygon segmentation"
                )
    return len(data["images"]), len(data["annotations"])


# ─────────────────────────────────────────────────────────────────────────────
# Pascal VOC
# ─────────────────────────────────────────────────────────────────────────────

def validate_voc_file(path: Path) -> int:
    """Re-parse a VOC ``.xml``; return the number of ``<object>`` entries.

    Raises:
        ValidationError: On bad XML, missing size, or invalid bndbox ordering/range.
    """
    path = Path(path)
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValidationError(f"{path}: malformed XML ({exc})") from exc
    root = tree.getroot()
    if root.tag != "annotation":
        raise ValidationError(f"{path}: root tag must be <annotation>")
    size = root.find("size")
    if size is None:
        raise ValidationError(f"{path}: missing <size>")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValidationError(f"{path}: width/height must be positive")
    object_count = 0
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValidationError(f"{path}: <object> missing <bndbox>")
        xmin = int(bnd.findtext("xmin", "0"))
        ymin = int(bnd.findtext("ymin", "0"))
        xmax = int(bnd.findtext("xmax", "0"))
        ymax = int(bnd.findtext("ymax", "0"))
        if not (xmin < xmax and ymin < ymax):
            raise ValidationError(f"{path}: bndbox ordering invalid ({xmin},{ymin},{xmax},{ymax})")
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            raise ValidationError(f"{path}: bndbox outside image bounds")
        object_count += 1
    return object_count
