"""Filesystem-backed session store.

The backend keeps every uploaded image, every persisted mask, and every
annotation under a single directory tree::

    outputs/sessions/default/
      ├── images/<image_id>.<ext>
      ├── masks/<annotation_id>.npz       (compressed boolean mask)
      └── annotations.json                (the source of truth - metadata + polygons)

The "default" name is the only session for v1 (single user, single browser).
Multi-session is out of scope.

This module is **deliberately framework-agnostic** — it knows nothing about
FastAPI. The HTTP layer in ``app/api/routes.py`` calls these functions. That
keeps the session store testable on its own and easy to reuse later (e.g. a
batch CLI wrapping the API).
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image

from utils.config import load_config

logger = logging.getLogger(__name__)

SESSION_NAME = "default"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
ANNOTATIONS_FILE = "annotations.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (the on-disk truth; the API layer adapts these to Pydantic).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageRow:
    """One uploaded image."""

    image_id: str
    file_name: str
    width: int
    height: int
    done: bool = False  # marked "reviewed / done" by the user


@dataclass
class AnnotationRow:
    """One annotation row, persisted in annotations.json."""

    annotation_id: str
    image_id: str
    polygon: list[list[float]]
    box: tuple[float, float, float, float]
    label: str
    score: float
    is_crowd: bool = False
    approved: bool | None = None
    source: str = "auto"
    mask_path: str | None = None  # relative to session dir, e.g. "masks/ann_xxx.npz"


@dataclass
class SessionData:
    """Whole-session snapshot read from / written to ``annotations.json``."""

    images: list[ImageRow] = field(default_factory=list)
    annotations: list[AnnotationRow] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SessionStore
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """Filesystem session manager. Thread-safe for the single-user case."""

    def __init__(self, root: Path | None = None) -> None:
        """Create the store rooted at ``root`` (defaults to ``paths.output_dir/sessions``)."""
        cfg = load_config()
        sessions_root = root or (cfg.path("paths.output_dir") / "sessions")
        self.dir = Path(sessions_root) / SESSION_NAME
        self.images_dir = self.dir / "images"
        self.masks_dir = self.dir / "masks"
        self.annotations_path = self.dir / ANNOTATIONS_FILE
        self._lock = threading.Lock()
        self._ensure_dirs()
        self._data = self._load()

    # ─── public API ──────────────────────────────────────────────────────

    def wipe(self) -> None:
        """Delete the entire session directory and recreate empty structure."""
        with self._lock:
            if self.dir.exists():
                shutil.rmtree(self.dir, ignore_errors=True)
            self._ensure_dirs()
            self._data = SessionData()
            self._persist()
            logger.info("Session wiped: %s", self.dir)

    # ─── image operations ────────────────────────────────────────────────

    def save_image(self, file: IO[bytes], original_name: str) -> ImageRow:
        """Persist an uploaded image and return its metadata row."""
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {suffix!r}")
        data = file.read()
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        with self._lock:
            image_id = self._make_image_id()
            out_path = self.images_dir / f"{image_id}{suffix}"
            out_path.write_bytes(data)
            row = ImageRow(image_id=image_id, file_name=original_name, width=width, height=height)
            self._data.images.append(row)
            self._persist()
        logger.info("Saved image %s (%dx%d) -> %s", image_id, width, height, out_path.name)
        return row

    def list_images(self) -> list[ImageRow]:
        with self._lock:
            return list(self._data.images)

    def get_image(self, image_id: str) -> ImageRow | None:
        with self._lock:
            return next((i for i in self._data.images if i.image_id == image_id), None)

    def update_image(self, image_id: str, done: bool | None = None) -> ImageRow | None:
        """Patch image-level fields (currently just ``done``). Returns the row or None."""
        with self._lock:
            row = next((i for i in self._data.images if i.image_id == image_id), None)
            if row is None:
                return None
            if done is not None:
                row.done = done
            self._persist()
            return row

    def get_image_path(self, image_id: str) -> Path | None:
        """Return the on-disk path for an image, or ``None`` if missing."""
        with self._lock:
            row = next((i for i in self._data.images if i.image_id == image_id), None)
        if row is None:
            return None
        for path in self.images_dir.glob(f"{image_id}.*"):
            return path
        return None

    # ─── annotation operations ───────────────────────────────────────────

    def add_annotation(self, row: AnnotationRow, mask: np.ndarray | None = None) -> AnnotationRow:
        """Append an annotation row; optionally persist its boolean mask as ``.npz``."""
        with self._lock:
            if mask is not None:
                mask_path = self.masks_dir / f"{row.annotation_id}.npz"
                np.savez_compressed(mask_path, mask=mask.astype(bool))
                row.mask_path = str(mask_path.relative_to(self.dir).as_posix())
            self._data.annotations.append(row)
            self._persist()
        return row

    def list_annotations(self, image_id: str | None = None) -> list[AnnotationRow]:
        with self._lock:
            if image_id is None:
                return list(self._data.annotations)
            return [a for a in self._data.annotations if a.image_id == image_id]

    def get_annotation(self, annotation_id: str) -> AnnotationRow | None:
        with self._lock:
            return next(
                (a for a in self._data.annotations if a.annotation_id == annotation_id), None
            )

    def update_annotation(
        self,
        annotation_id: str,
        approved: bool | None = None,
        label: str | None = None,
        is_crowd: bool | None = None,
    ) -> AnnotationRow | None:
        """Patch fields of an annotation; return the updated row (or None if missing)."""
        with self._lock:
            row = next(
                (a for a in self._data.annotations if a.annotation_id == annotation_id), None
            )
            if row is None:
                return None
            if approved is not None:
                row.approved = approved
            if label is not None:
                row.label = label
            if is_crowd is not None:
                row.is_crowd = is_crowd
            self._persist()
            return row

    def delete_annotation(self, annotation_id: str) -> bool:
        """Remove an annotation and its mask file. Return True if found, False otherwise."""
        with self._lock:
            for i, row in enumerate(self._data.annotations):
                if row.annotation_id == annotation_id:
                    if row.mask_path:
                        mask_file = self.dir / row.mask_path
                        mask_file.unlink(missing_ok=True)
                    self._data.annotations.pop(i)
                    self._persist()
                    return True
        return False

    def bulk_update_annotations(
        self,
        image_id: str,
        annotation_ids: list[str],
        action: str,
    ) -> tuple[int, list[tuple[str, str]]]:
        """Apply one of approve/reject/delete to many annotations atomically.

        Returns ``(updated_count, errors)`` where ``errors`` is a list of
        ``(annotation_id, reason)`` for ids that didn't belong to ``image_id``
        or didn't exist. The whole batch is one persist().
        """
        if action not in ("approve", "reject", "delete"):
            raise ValueError(f"Unknown bulk action: {action!r}")
        wanted = set(annotation_ids)
        updated = 0
        errors: list[tuple[str, str]] = []
        with self._lock:
            index_by_id = {a.annotation_id: i for i, a in enumerate(self._data.annotations)}
            for ann_id in annotation_ids:
                idx = index_by_id.get(ann_id)
                if idx is None:
                    errors.append((ann_id, "unknown annotation_id"))
                    continue
                row = self._data.annotations[idx]
                if row.image_id != image_id:
                    errors.append((ann_id, "annotation does not belong to image"))
            if action == "delete":
                kept: list[AnnotationRow] = []
                for row in self._data.annotations:
                    if row.annotation_id in wanted and row.image_id == image_id:
                        if row.mask_path:
                            (self.dir / row.mask_path).unlink(missing_ok=True)
                        updated += 1
                    else:
                        kept.append(row)
                self._data.annotations = kept
            else:
                target = True if action == "approve" else False
                for row in self._data.annotations:
                    if row.annotation_id in wanted and row.image_id == image_id:
                        row.approved = target
                        updated += 1
            if updated or errors:
                self._persist()
        return updated, errors

    def delete_annotations_for_image(self, image_id: str, source: str | None = None) -> int:
        """Delete every annotation for one image (optionally filtered by ``source``).

        Returns the number of rows removed. Used by ``auto-detect`` to replace
        previous auto detections without touching the user's manual ones.
        """
        removed = 0
        with self._lock:
            kept: list[AnnotationRow] = []
            for row in self._data.annotations:
                drop = row.image_id == image_id and (source is None or row.source == source)
                if drop:
                    if row.mask_path:
                        (self.dir / row.mask_path).unlink(missing_ok=True)
                    removed += 1
                else:
                    kept.append(row)
            self._data.annotations = kept
            if removed:
                self._persist()
        return removed

    def load_mask(self, annotation_id: str) -> np.ndarray | None:
        """Load the persisted boolean mask for an annotation, if any."""
        row = self.get_annotation(annotation_id)
        if row is None or row.mask_path is None:
            return None
        mask_file = self.dir / row.mask_path
        if not mask_file.exists():
            return None
        with np.load(mask_file) as bundle:
            return bundle["mask"].astype(bool)

    # ─── id generators (monotonic, sortable) ─────────────────────────────

    def _make_image_id(self) -> str:
        return f"img_{int(time.time() * 1000):013d}_{len(self._data.images):04d}"

    def make_annotation_id(self) -> str:
        with self._lock:
            return f"ann_{int(time.time() * 1000):013d}_{len(self._data.annotations):05d}"

    # ─── internal: load / persist ────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> SessionData:
        if not self.annotations_path.exists():
            return SessionData()
        try:
            raw = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt %s — starting fresh", self.annotations_path)
            return SessionData()
        images = [ImageRow(**row) for row in raw.get("images", [])]
        annotations = [AnnotationRow(**row) for row in raw.get("annotations", [])]
        return SessionData(images=images, annotations=annotations)

    def _persist(self) -> None:
        payload = {
            "images": [asdict(i) for i in self._data.images],
            "annotations": [asdict(a) for a in self._data.annotations],
        }
        self.annotations_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
