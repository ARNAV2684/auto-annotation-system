"""Annotation exporters: YOLO (.txt), COCO (.json), Pascal VOC (.xml).

Converts the in-memory ``Detection`` objects produced by the pipeline into the three
industry-standard annotation formats so downstream training tools can consume them.

Design rules (so coordinate bugs are tested out of existence — brief §10):

* YOLO and COCO coordinate conventions are documented next to the code that emits them.
* RLE encoding is plain-NumPy "uncompressed" COCO RLE (column-major run lengths) —
  no ``pycocotools`` dependency; tested with a round-trip decoder.
* All public writers share a small ``BaseWriter`` interface: ``add_image`` while
  iterating, ``finalize`` to flush dataset-level files.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.mask_processor import mask_to_bbox, mask_to_polygon
from core.types import Detection
from utils.config import load_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data carriers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExportImage:
    """One image's annotations, ready to hand to a writer."""

    file_name: str
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Class-name <-> integer-id mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_class_index(classes: list[str]) -> dict[str, int]:
    """Return a deterministic ``{class_name: id}`` map (0-based, list order)."""
    return {name: i for i, name in enumerate(classes)}


def normalise_label(label: str) -> str:
    """Canonical class / label form: lowercase, trimmed, whitespace runs -> ``_``.

    Used everywhere a free-text label is turned into a class name (the export
    writers, the annotation PATCH route, and the project store) so a class that
    is *registered* and a label that is *applied* always agree — e.g. typing
    ``"Big House"`` yields ``"big_house"`` on both paths.
    """
    return "_".join(label.strip().lower().split())


# Backwards-compatible private alias (the writers call this name).
_normalise_label = normalise_label


def resolve_export_classes(
    projects: list[dict],
    project_id: str | None = None,
    *,
    crowd_classes: list[str] | None = None,
    legacy_classes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Decide which class list (and crowd classes) an export should use.

    This is the single policy point for "which classes does this export cover".
    It is a pure function over an explicit ``projects`` list (supplied by the
    caller from the live :class:`~app.services.project_store.ProjectStore`), so
    newly-registered classes are honoured and the logic is trivially testable.

    * ``project_id`` given -> that project's ``classes`` (raises :class:`KeyError`
      if no project with that id exists). Class ids are scoped to this one
      project, which is what downstream single-purpose training expects
      (e.g. a road-defects YOLO with ``0=pothole, 1=crack, ...``).
    * ``project_id`` is ``None`` -> the de-duplicated **union** of every
      project's classes, first-seen order preserved, so an annotation is never
      silently dropped just because no project was named.
    * Empty ``projects`` -> fall back to ``legacy_classes``.

    Args:
        projects: List of project dicts (each has ``id`` and ``classes``).
        project_id: Optional project id to scope the export to.
        crowd_classes: Global crowd-class list (crowd is a per-label property).
        legacy_classes: Fallback class list when ``projects`` is empty.

    Returns:
        ``(classes, crowd_classes)``.

    Raises:
        KeyError: If ``project_id`` is given but no such project exists.
    """
    crowd = list(crowd_classes or [])

    if project_id is not None:
        match = next((p for p in projects if p.get("id") == project_id), None)
        if match is None:
            raise KeyError(project_id)
        return list(match.get("classes", []) or []), crowd

    if projects:
        # Dict preserves first-seen insertion order while de-duplicating.
        seen: dict[str, None] = {}
        for project in projects:
            for class_name in project.get("classes", []) or []:
                seen.setdefault(class_name, None)
        return list(seen.keys()), crowd

    return list(legacy_classes or []), crowd


def resolve_class_id(label: str, class_index: dict[str, int]) -> int | None:
    """Map a free-text detector label to a class id, robust to GroundingDINO quirks.

    GroundingDINO sometimes returns merged phrases like ``"tree palm tree"`` when the
    prompt contains multiple phrases sharing a word. Strict equality drops these
    detections. We try:

    1. Exact match after normalisation (``"palm tree"`` -> ``"palm_tree"``).
    2. Substring match against class names (with underscores treated as spaces),
       preferring the *longest* class name so e.g. ``"tree palm tree"`` matches
       ``palm_tree`` instead of ``tree``.

    Args:
        label: Free-text label from the detector.
        class_index: ``{class_name: id}`` map (typically from :func:`build_class_index`).

    Returns:
        The class id, or ``None`` if no class could be resolved.
    """
    normalised = " ".join(label.strip().lower().split())
    underscored = normalised.replace(" ", "_")
    if underscored in class_index:
        return class_index[underscored]
    # Longest class name first so 'palm_tree' is preferred over 'tree' inside
    # a phrase like 'tree palm tree'.
    for class_name in sorted(class_index, key=len, reverse=True):
        phrase = class_name.replace("_", " ")
        if phrase in normalised:
            return class_index[class_name]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# COCO uncompressed RLE  (decoder is exported for tests / consumers)
# ─────────────────────────────────────────────────────────────────────────────

def encode_coco_rle(mask: np.ndarray) -> dict:
    """Encode a boolean mask as an uncompressed COCO RLE dict.

    The ``counts`` list is column-major run lengths starting with the background
    (value 0). If the first pixel is foreground we prepend a zero-length background
    run, per COCO convention.

    Args:
        mask: Boolean ``H x W`` mask.

    Returns:
        ``{"size": [H, W], "counts": [int, ...]}`` — compatible with
        ``pycocotools.mask.decode``.
    """
    height, width = mask.shape
    flat = mask.astype(np.uint8).flatten(order="F")  # column-major
    if flat.size == 0:
        return {"size": [int(height), int(width)], "counts": []}
    transitions = np.flatnonzero(np.diff(flat).astype(bool))
    if transitions.size == 0:
        total = int(flat.size)
        counts = [total] if flat[0] == 0 else [0, total]
        return {"size": [int(height), int(width)], "counts": counts}
    boundaries = np.concatenate(([-1], transitions, [flat.size - 1]))
    runs = np.diff(boundaries).astype(int).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return {"size": [int(height), int(width)], "counts": runs}


def decode_coco_rle(rle: dict) -> np.ndarray:
    """Inverse of :func:`encode_coco_rle` (used by tests and validators)."""
    height, width = rle["size"]
    counts = list(rle["counts"])
    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    value = np.uint8(0)
    for count in counts:
        flat[pos : pos + count] = value
        pos += count
        value = np.uint8(1 - value)
    return flat.reshape((height, width), order="F").astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

class BaseWriter:
    """Shared interface for the three format writers."""

    def __init__(self, output_dir: Path, classes: list[str]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
        self.class_index = build_class_index(classes)

    def add_image(self, image: ExportImage) -> None:
        raise NotImplementedError

    def finalize(self) -> None:  # pragma: no cover - default is a no-op
        return None

    def _class_id(self, label: str) -> int | None:
        """Resolve a detection label to a class id, or ``None`` if unknown."""
        class_id = resolve_class_id(label, self.class_index)
        if class_id is None:
            logger.warning("Unknown class label '%s' (not in config.classes); skipping.", label)
        return class_id


class YoloWriter(BaseWriter):
    """YOLO writer: one ``.txt`` per image (segmentation polygons by default).

    Line format (segmentation):  ``class_id  x1 y1 x2 y2 ... xn yn``
    Line format (detection):     ``class_id  cx cy w h``

    All coordinates are normalised to ``[0, 1]`` of image dimensions. Multiple polygons
    for one mask are written as separate lines with the same class (Ultralytics convention).
    Also writes a dataset-level ``classes.txt`` and a minimal ``data.yaml``.
    """

    def __init__(
        self,
        output_dir: Path,
        classes: list[str],
        mode: str = "segmentation",
        polygon_epsilon: float = 2.0,
    ) -> None:
        super().__init__(output_dir, classes)
        if mode not in {"segmentation", "detection"}:
            raise ValueError(f"YOLO mode must be 'segmentation' or 'detection', got {mode!r}")
        self.mode = mode
        self.polygon_epsilon = polygon_epsilon
        self._image_count = 0

    def add_image(self, image: ExportImage) -> None:
        lines: list[str] = []
        for det in image.detections:
            cid = self._class_id(det.label)
            if cid is None:
                continue
            if self.mode == "detection":
                line = self._detection_line(cid, det.box, image.width, image.height)
                if line:
                    lines.append(line)
            else:
                lines.extend(self._segmentation_lines(cid, det.mask, image.width, image.height))
        out_path = self.output_dir / (Path(image.file_name).stem + ".txt")
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._image_count += 1

    def finalize(self) -> None:
        (self.output_dir / "classes.txt").write_text(
            "\n".join(self.classes) + "\n", encoding="utf-8"
        )
        # Minimal Ultralytics data.yaml for convenience
        data_yaml = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(self.classes))
        data_yaml += f"nc: {len(self.classes)}\n"
        (self.output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
        logger.info("YOLO: %d images written to %s", self._image_count, self.output_dir)

    @staticmethod
    def _detection_line(class_id: int, box, width: int, height: int) -> str:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def _segmentation_lines(
        self, class_id: int, mask: np.ndarray, width: int, height: int
    ) -> list[str]:
        polygons = mask_to_polygon(mask, epsilon=self.polygon_epsilon)
        lines: list[str] = []
        for polygon in polygons:
            if len(polygon) < 6:
                continue  # need >= 3 points
            normalised = [
                f"{(polygon[i] / width):.6f} {(polygon[i + 1] / height):.6f}"
                for i in range(0, len(polygon), 2)
            ]
            lines.append(f"{class_id} " + " ".join(normalised))
        return lines


class CocoWriter(BaseWriter):
    """COCO writer: aggregates all images into a single ``annotations.json``.

    - Polygon segmentation with ``iscrowd=0`` for normal objects.
    - RLE segmentation with ``iscrowd=1`` for objects flagged as crowd (either
      ``Detection.is_crowd`` or whose class is in ``coco.crowd_classes``).
    - ``bbox`` is absolute ``[x, y, w, h]`` (COCO convention — NOT ``[x1, y1, x2, y2]``).
    """

    def __init__(
        self,
        output_dir: Path,
        classes: list[str],
        crowd_classes: list[str] | None = None,
        supercategory: str = "object",
        polygon_epsilon: float = 2.0,
    ) -> None:
        super().__init__(output_dir, classes)
        self.crowd_classes = {_normalise_label(c) for c in (crowd_classes or [])}
        self.supercategory = supercategory
        self.polygon_epsilon = polygon_epsilon
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._next_image_id = 1
        self._next_annotation_id = 1

    def add_image(self, image: ExportImage) -> None:
        image_id = self._next_image_id
        self._next_image_id += 1
        self._images.append(
            {
                "id": image_id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            }
        )
        for det in image.detections:
            cid = self._class_id(det.label)
            if cid is None:
                continue
            self._annotations.append(self._build_annotation(image_id, cid, det))

    def finalize(self) -> None:
        categories = [
            {"id": i, "name": name, "supercategory": self.supercategory}
            for i, name in enumerate(self.classes)
        ]
        coco = {
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }
        out_path = self.output_dir / "annotations.json"
        out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        logger.info(
            "COCO: %d images, %d annotations -> %s",
            len(self._images),
            len(self._annotations),
            out_path,
        )

    def _build_annotation(self, image_id: int, class_id: int, det: Detection) -> dict:
        is_crowd = det.is_crowd or _normalise_label(det.label) in self.crowd_classes
        bbox_xyxy = mask_to_bbox(det.mask) or det.box
        x1, y1, x2, y2 = bbox_xyxy
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        area = float(int(det.mask.sum()))
        annotation = {
            "id": self._next_annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 1 if is_crowd else 0,
            "score": float(det.score),
        }
        if is_crowd:
            annotation["segmentation"] = encode_coco_rle(det.mask)
        else:
            polygons = mask_to_polygon(det.mask, epsilon=self.polygon_epsilon)
            annotation["segmentation"] = [
                [float(v) for v in poly] for poly in polygons if len(poly) >= 6
            ]
        self._next_annotation_id += 1
        return annotation


class VocWriter(BaseWriter):
    """Pascal VOC writer: one ``.xml`` per image with ``<bndbox>`` per object.

    Bounding-box coordinates are 0-indexed pixels (modern VOC consumers accept this).
    Shapes (polygons/masks) are NOT stored — VOC is a boxes-only format.
    """

    def __init__(
        self,
        output_dir: Path,
        classes: list[str],
        database: str = "GarudaUAV",
    ) -> None:
        super().__init__(output_dir, classes)
        self.database = database
        self._image_count = 0

    def add_image(self, image: ExportImage) -> None:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = str(self.output_dir.name)
        ET.SubElement(root, "filename").text = image.file_name
        ET.SubElement(root, "path").text = image.file_name
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = self.database
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image.width)
        ET.SubElement(size, "height").text = str(image.height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"
        for det in image.detections:
            if self._class_id(det.label) is None:
                continue
            self._append_object(root, det)
        # ET.indent landed in 3.9, available in our 3.12
        ET.indent(root, space="  ")
        tree = ET.ElementTree(root)
        out_path = self.output_dir / (Path(image.file_name).stem + ".xml")
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        self._image_count += 1

    def finalize(self) -> None:
        logger.info("VOC: %d files written to %s", self._image_count, self.output_dir)

    @staticmethod
    def _append_object(root: ET.Element, det: Detection) -> None:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = _normalise_label(det.label)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = det.box
        ET.SubElement(bndbox, "xmin").text = str(int(x1))
        ET.SubElement(bndbox, "ymin").text = str(int(y1))
        ET.SubElement(bndbox, "xmax").text = str(int(x2))
        ET.SubElement(bndbox, "ymax").text = str(int(y2))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build all three writers from config
# ─────────────────────────────────────────────────────────────────────────────

def build_writers(
    output_root: Path,
    formats: list[str] | None = None,
    config=None,
    *,
    classes: list[str] | None = None,
    crowd_classes: list[str] | None = None,
) -> dict[str, BaseWriter]:
    """Construct only the requested format writers, configured from ``config.yaml``.

    Args:
        output_root: Folder to create per-format subdirs under.
        formats: Subset of ``{"yolo", "coco", "voc"}``; ``None`` means all three.
        config: Optional :class:`~utils.config.Config`; defaults to the project config.
        classes: Explicit class list to use (e.g. one project's classes). When
            ``None``, falls back to the flat legacy ``classes`` config key — the
            caller is expected to resolve project scoping via
            :func:`resolve_export_classes`.
        crowd_classes: Explicit crowd-class list. When ``None``, falls back to
            ``coco.crowd_classes``.

    Returns:
        ``{format: writer}`` only for the requested formats (no empty dirs created
        for formats the user did not ask for).
    """
    cfg = config or load_config()
    if classes is None:
        classes = cfg.get("classes", []) or []
    if crowd_classes is None:
        crowd_classes = cfg.get("coco.crowd_classes", []) or []
    polygon_epsilon = float(cfg.get("mask.polygon_epsilon", 2.0))
    requested = set(formats) if formats else {"yolo", "coco", "voc"}
    factories: dict[str, BaseWriter] = {}
    if "yolo" in requested:
        factories["yolo"] = YoloWriter(
            output_root / "yolo",
            classes,
            mode=cfg.get("yolo.mode", "segmentation"),
            polygon_epsilon=polygon_epsilon,
        )
    if "coco" in requested:
        factories["coco"] = CocoWriter(
            output_root / "coco",
            classes,
            crowd_classes=crowd_classes,
            supercategory=cfg.get("coco.supercategory", "object"),
            polygon_epsilon=polygon_epsilon,
        )
    if "voc" in requested:
        factories["voc"] = VocWriter(
            output_root / "voc",
            classes,
            database=cfg.get("voc.database", "GarudaUAV"),
        )
    return factories
