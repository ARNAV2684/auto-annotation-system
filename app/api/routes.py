"""HTTP routes for the Auto Annotation backend.

Every route implements one row of ``docs/api_contract.md``. Heavy work
(segmentation, detection) is delegated to :class:`~app.services.model_registry`;
persistence is delegated to :class:`~app.services.session_store.SessionStore`.

Routes are deliberately thin — they translate JSON ↔ dataclasses ↔ Pydantic,
hand off, and re-serialise. That keeps them easy to test and easy to swap out
later (e.g. for a CLI or gRPC frontend).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response

from app.api.schemas import (
    AddClassRequest,
    Annotation,
    AnnotationPatch,
    AutoDetectRequest,
    BoxRequest,
    BulkAnnotationError,
    BulkAnnotationRequest,
    BulkAnnotationResponse,
    CreateProjectRequest,
    EmbedResponse,
    ExportRequest,
    ExportResponse,
    HealthResponse,
    ImagePatch,
    ImageRecord,
    MultiPointRequest,
    PointRequest,
    PolygonRequest,
    PreviewResponse,
    Project,
    StatsResponse,
)
from app.services.model_registry import ModelRegistry, get_model_registry
from app.services.project_store import ProjectStore
from app.services.session_store import AnnotationRow, ImageRow, SessionStore
from core.annotation_converter import (
    ExportImage,
    build_class_index,
    build_writers,
    normalise_label,
    resolve_class_id,
    resolve_export_classes,
)
from core.mask_processor import mask_to_bbox, mask_to_polygon, polygon_to_mask
from core.types import Detection
from utils.config import load_config
from utils.file_handler import make_run_dir, write_stats, zip_directory
from utils.validators import validate_coco_file, validate_voc_file, validate_yolo_file

logger = logging.getLogger(__name__)

# Routers — split only by path-prefix for clarity; all served under /api.
router = APIRouter(prefix="/api")

_IMAGE_CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
    ".webp": "image/webp",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────

_session_singleton: SessionStore | None = None
_project_store_singleton: ProjectStore | None = None


def get_session_store() -> SessionStore:
    """FastAPI dependency: a process-wide :class:`SessionStore` singleton."""
    global _session_singleton
    if _session_singleton is None:
        _session_singleton = SessionStore()
    return _session_singleton


def get_project_store() -> ProjectStore:
    """FastAPI dependency: a process-wide :class:`ProjectStore` singleton."""
    global _project_store_singleton
    if _project_store_singleton is None:
        _project_store_singleton = ProjectStore()
    return _project_store_singleton


StoreDep = Annotated[SessionStore, Depends(get_session_store)]
ProjectStoreDep = Annotated[ProjectStore, Depends(get_project_store)]
ModelsDep = Annotated[ModelRegistry, Depends(get_model_registry)]


# ─────────────────────────────────────────────────────────────────────────────
# Conversion helpers (dataclass row -> Pydantic shape)
# ─────────────────────────────────────────────────────────────────────────────

def _annotation_from_row(row: AnnotationRow) -> Annotation:
    return Annotation(
        annotation_id=row.annotation_id,
        image_id=row.image_id,
        polygon=row.polygon,
        box=tuple(row.box),
        label=row.label,
        score=row.score,
        is_crowd=row.is_crowd,
        approved=row.approved,
        source=row.source,
    )


def _image_record(image: ImageRow, num_detections: int, num_approved: int) -> ImageRecord:
    return ImageRecord(
        image_id=image.image_id,
        file_name=image.file_name,
        width=image.width,
        height=image.height,
        num_detections=num_detections,
        num_approved=num_approved,
        done=image.done,
        tags=list(image.tags),
    )


def _polygon_from_mask(mask: np.ndarray, epsilon: float | None = None) -> list[list[float]]:
    """Convert a boolean mask to one or more flat polygons (floats).

    ``epsilon`` overrides the Douglas-Peucker simplification tolerance (driven by
    the Smart Select "Simplify" slider); ``None`` uses the config default.
    """
    if epsilon is None:
        epsilon = float(load_config().get("mask.polygon_epsilon", 2.0))
    return [[float(v) for v in poly] for poly in mask_to_polygon(mask, epsilon=epsilon)]


def _persist_annotation(
    store: SessionStore,
    image_id: str,
    mask: np.ndarray,
    score: float,
    label: str,
    source: str,
    epsilon: float | None = None,
) -> AnnotationRow:
    """Build an :class:`AnnotationRow` from a mask + metadata, persist it, return it."""
    polygon = _polygon_from_mask(mask, epsilon=epsilon)
    bbox = mask_to_bbox(mask)
    if bbox is None:
        ys, xs = np.where(mask)
        bbox = (float(xs.min()) if xs.size else 0.0,
                float(ys.min()) if ys.size else 0.0,
                float(xs.max()) if xs.size else 0.0,
                float(ys.max()) if ys.size else 0.0)
    row = AnnotationRow(
        annotation_id=store.make_annotation_id(),
        image_id=image_id,
        polygon=polygon,
        box=tuple(float(v) for v in bbox),
        label=label,
        score=float(score),
        source=source,
    )
    return store.add_annotation(row, mask=mask)


def _ensure_image_loaded(
    models: ModelRegistry, store: SessionStore, image_id: str
) -> Path:
    """Resolve ``image_id`` to a path and make sure SAM has its embedding ready."""
    path = store.get_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    models.ensure_embedded(image_id, path)
    return path


def _counts_for_image(store: SessionStore, image_id: str) -> tuple[int, int]:
    anns = store.list_annotations(image_id=image_id)
    return len(anns), sum(1 for a in anns if a.approved is True)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap liveness probe. Does NOT load models if they aren't already loaded."""
    from app.services import model_registry as registry_module

    loaded = registry_module._registry is not None
    device = registry_module._registry.segmenter.device if loaded else "unknown"
    return HealthResponse(status="ok", device=device, models_loaded=loaded)


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=ImageRecord,
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(
    store: StoreDep,
    file: Annotated[UploadFile, File(description="Drone image (JPG/PNG/BMP/TIFF/WebP)")],
) -> ImageRecord:
    """Save one uploaded image and return its metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    data = await file.read()
    try:
        row = store.save_image(_BytesIO(data), file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _image_record(row, num_detections=0, num_approved=0)


@router.get("/images", response_model=list[ImageRecord])
def list_images(store: StoreDep) -> list[ImageRecord]:
    """List every uploaded image with its detection / approval counts."""
    out: list[ImageRecord] = []
    for image in store.list_images():
        total, approved = _counts_for_image(store, image.image_id)
        out.append(_image_record(image, total, approved))
    return out


@router.patch("/images/{image_id}", response_model=ImageRecord)
def patch_image(image_id: str, body: ImagePatch, store: StoreDep) -> ImageRecord:
    """Update image-level fields — currently the 'done / reviewed' flag."""
    row = store.update_image(image_id, done=body.done, tags=body.tags)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    total, approved = _counts_for_image(store, image_id)
    return _image_record(row, total, approved)


@router.get("/images/{image_id}/file")
def get_image_file(image_id: str, store: StoreDep) -> FileResponse:
    """Return the raw image bytes so the canvas can draw it."""
    path = store.get_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    media_type = _IMAGE_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/images/{image_id}/embed", response_model=EmbedResponse)
def embed_image(image_id: str, store: StoreDep, models: ModelsDep) -> EmbedResponse:
    """Force SAM to compute (or reuse) the embedding for this image."""
    path = store.get_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    cached = models.ensure_embedded(image_id, path)
    return EmbedResponse(ok=True, cached=cached)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/images/{image_id}/infer/point",
    response_model=list[Annotation],
    status_code=status.HTTP_201_CREATED,
)
def infer_point(
    image_id: str, body: PointRequest, store: StoreDep, models: ModelsDep
) -> list[Annotation]:
    """Segment the object at a click and persist the resulting annotation(s)."""
    _ensure_image_loaded(models, store, image_id)
    masks, scores = models.predict_point(
        (body.x, body.y), label=body.label, multimask=body.multimask
    )
    rows = [
        _persist_annotation(store, image_id, mask, score, label="unlabeled", source="point")
        for mask, score in zip(masks, scores, strict=True)
    ]
    return [_annotation_from_row(r) for r in rows]


@router.post("/images/{image_id}/infer/preview", response_model=PreviewResponse)
def infer_preview(
    image_id: str, body: MultiPointRequest, store: StoreDep, models: ModelsDep
) -> PreviewResponse:
    """Smart Select preview: segment from accumulated points WITHOUT persisting.

    Used for the hover ghost and the live in-progress mask, so it must stay cheap
    and side-effect free.
    """
    _ensure_image_loaded(models, store, image_id)
    points = [(p.x, p.y) for p in body.points]
    labels = [p.label for p in body.points]
    masks, scores = models.predict_points(points, labels, multimask=body.multimask)
    mask, score = masks[0], scores[0]
    bbox = mask_to_bbox(mask) or (0.0, 0.0, 0.0, 0.0)
    return PreviewResponse(
        polygon=_polygon_from_mask(mask, epsilon=body.simplify),
        box=tuple(float(v) for v in bbox),
        score=float(score),
    )


@router.post(
    "/images/{image_id}/infer/points",
    response_model=list[Annotation],
    status_code=status.HTTP_201_CREATED,
)
def infer_points(
    image_id: str, body: MultiPointRequest, store: StoreDep, models: ModelsDep
) -> list[Annotation]:
    """Smart Select commit: persist the best mask from the accumulated points."""
    _ensure_image_loaded(models, store, image_id)
    points = [(p.x, p.y) for p in body.points]
    labels = [p.label for p in body.points]
    masks, scores = models.predict_points(points, labels, multimask=body.multimask)
    row = _persist_annotation(
        store, image_id, masks[0], scores[0],
        label="unlabeled", source="point", epsilon=body.simplify,
    )
    return [_annotation_from_row(row)]


@router.post(
    "/images/{image_id}/infer/box",
    response_model=Annotation,
    status_code=status.HTTP_201_CREATED,
)
def infer_box(
    image_id: str, body: BoxRequest, store: StoreDep, models: ModelsDep
) -> Annotation:
    """Segment the object inside a bounding box and persist the annotation."""
    if body.x2 <= body.x1 or body.y2 <= body.y1:
        raise HTTPException(status_code=422, detail="Box must satisfy x2>x1 and y2>y1.")
    _ensure_image_loaded(models, store, image_id)
    mask, score = models.predict_box((body.x1, body.y1, body.x2, body.y2))
    row = _persist_annotation(store, image_id, mask, score, label="unlabeled", source="box")
    return _annotation_from_row(row)


@router.post(
    "/images/{image_id}/annotations/polygon",
    response_model=Annotation,
    status_code=status.HTTP_201_CREATED,
)
def create_polygon(image_id: str, body: PolygonRequest, store: StoreDep) -> Annotation:
    """Persist a hand-drawn polygon as an annotation (rasterised to a mask).

    This is the manual escape hatch for objects SAM can't grab — the outline is
    filled into a mask so the annotation stores and exports exactly like a SAM one.
    """
    image = store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    mask = polygon_to_mask(
        [(float(x), float(y)) for x, y in body.points], image.width, image.height
    )
    if not mask.any():
        raise HTTPException(status_code=422, detail="Polygon is empty or degenerate.")
    label = normalise_label(body.label) if body.label else "unlabeled"
    row = _persist_annotation(store, image_id, mask, score=1.0, label=label, source="polygon")
    return _annotation_from_row(row)


@router.post(
    "/images/{image_id}/annotations/copy-from/{source_id}",
    response_model=list[Annotation],
    status_code=status.HTTP_201_CREATED,
)
def copy_annotations(image_id: str, source_id: str, store: StoreDep) -> list[Annotation]:
    """Copy every annotation from ``source_id`` onto ``image_id`` (Repeat previous)."""
    if store.get_image(image_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    if store.get_image(source_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown source image: {source_id}")
    rows = store.copy_annotations(source_id, image_id)
    return [_annotation_from_row(r) for r in rows]


@router.post(
    "/images/{image_id}/auto-detect",
    response_model=list[Annotation],
    status_code=status.HTTP_201_CREATED,
)
def auto_detect(
    image_id: str, body: AutoDetectRequest, store: StoreDep, models: ModelsDep
) -> list[Annotation]:
    """Run GroundingDINO + SAM. Replaces previous 'auto' annotations for this image."""
    path = store.get_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    store.delete_annotations_for_image(image_id, source="auto")
    detections: list[Detection] = models.auto_detect(path, body.prompts)
    rows = [
        _persist_annotation(store, image_id, d.mask, d.score, label=d.label, source="auto")
        for d in detections
    ]
    return [_annotation_from_row(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Annotations CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/images/{image_id}/annotations", response_model=list[Annotation])
def list_annotations(image_id: str, store: StoreDep) -> list[Annotation]:
    if store.get_image(image_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    return [_annotation_from_row(r) for r in store.list_annotations(image_id=image_id)]


_MAX_LABEL_LEN = 64


@router.patch("/annotations/{annotation_id}", response_model=Annotation)
def patch_annotation(
    annotation_id: str, body: AnnotationPatch, store: StoreDep
) -> Annotation:
    """Update an annotation. The label is normalised but NOT checked against a
    fixed class list — projects let the user pick from different lists, so any
    sensible non-empty label is accepted. The frontend dropdown is the policy.
    """
    if body.label is not None:
        normalised = normalise_label(body.label)
        if not normalised:
            raise HTTPException(status_code=422, detail="Label cannot be empty.")
        if len(normalised) > _MAX_LABEL_LEN:
            raise HTTPException(
                status_code=422,
                detail=f"Label too long (max {_MAX_LABEL_LEN} characters).",
            )
        body.label = normalised
    row = store.update_annotation(
        annotation_id,
        approved=body.approved,
        label=body.label,
        is_crowd=body.is_crowd,
    )
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown annotation_id: {annotation_id}"
        )
    return _annotation_from_row(row)


@router.delete(
    "/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response
)
def delete_annotation(annotation_id: str, store: StoreDep) -> Response:
    if not store.delete_annotation(annotation_id):
        raise HTTPException(
            status_code=404, detail=f"Unknown annotation_id: {annotation_id}"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/images/{image_id}/annotations/bulk",
    response_model=BulkAnnotationResponse,
)
def bulk_annotations(
    image_id: str, body: BulkAnnotationRequest, store: StoreDep
) -> BulkAnnotationResponse:
    """Apply approve / reject / delete to many annotations of one image."""
    if store.get_image(image_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")
    updated, errors = store.bulk_update_annotations(
        image_id=image_id,
        annotation_ids=body.annotation_ids,
        action=body.action,
    )
    return BulkAnnotationResponse(
        updated=updated,
        errors=[BulkAnnotationError(annotation_id=aid, reason=msg) for aid, msg in errors],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/export", response_model=ExportResponse, status_code=status.HTTP_201_CREATED
)
def export(
    body: ExportRequest, store: StoreDep, project_store: ProjectStoreDep
) -> ExportResponse:
    cfg = load_config()
    formats = (
        ["yolo", "coco", "voc"]
        if "all" in body.formats
        else [f for f in body.formats if f != "all"]
    )
    if not formats:
        raise HTTPException(status_code=422, detail="At least one format must be requested.")

    # Decide which class list this export covers (one project, or the union of
    # all projects when none is named), reading the LIVE project store so a
    # class registered moments ago is exportable. This is the fix for "blank
    # exports": previously every label was matched against the legacy 3-class
    # list, so any road-defect / infrastructure label silently resolved to nothing.
    try:
        classes, crowd_classes = resolve_export_classes(
            project_store.list_projects(),
            body.project_id,
            crowd_classes=cfg.get("coco.crowd_classes", []) or [],
            legacy_classes=cfg.get("classes", []) or [],
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=422, detail=f"Unknown project_id: {body.project_id}"
        ) from exc
    class_index = build_class_index(classes)

    run_dir = make_run_dir(cfg.path("paths.output_dir") / "exports")
    writers = build_writers(
        run_dir, formats=formats, config=cfg, classes=classes, crowd_classes=crowd_classes
    )

    export_images: list[ExportImage] = []
    exported_detections: list[Detection] = []
    num_dropped = 0
    dropped_labels: set[str] = set()
    for image in store.list_images():
        rows = [
            r for r in store.list_annotations(image_id=image.image_id)
            if (not body.only_approved) or r.approved is True
        ]
        if not rows:
            continue
        detections: list[Detection] = []
        for row in rows:
            mask = store.load_mask(row.annotation_id)
            if mask is None:
                # Polygon-only fallback (shouldn't normally happen).
                continue
            # Skip-and-report any label outside the chosen class list, so the
            # writers only ever see resolvable labels and the UI can warn the user.
            if resolve_class_id(row.label, class_index) is None:
                num_dropped += 1
                dropped_labels.add(row.label)
                continue
            detections.append(
                Detection(
                    mask=mask, box=tuple(int(v) for v in row.box),  # type: ignore[arg-type]
                    label=row.label, score=row.score, is_crowd=row.is_crowd,
                )
            )
        if not detections:
            continue
        ei = ExportImage(
            file_name=image.file_name,
            width=image.width,
            height=image.height,
            detections=detections,
        )
        export_images.append(ei)
        exported_detections.extend(detections)
        for w in writers.values():
            w.add_image(ei)

    for w in writers.values():
        w.finalize()

    write_stats(run_dir / "stats.json", export_images, exported_detections)

    # Validate every file we wrote — class ids must fall within the chosen list.
    if "yolo" in formats:
        for txt in (run_dir / "yolo").glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            validate_yolo_file(
                txt, num_classes=len(classes),
                mode=cfg.get("yolo.mode", "segmentation"),
            )
    if "coco" in formats and (run_dir / "coco" / "annotations.json").exists():
        validate_coco_file(run_dir / "coco" / "annotations.json")
    if "voc" in formats:
        for xml in (run_dir / "voc").glob("*.xml"):
            validate_voc_file(xml)

    zip_directory(run_dir, run_dir.parent / run_dir.name)

    return ExportResponse(
        run_id=run_dir.name,
        num_images=len(export_images),
        num_annotations_exported=len(exported_detections),
        download_url=f"/api/exports/{run_dir.name}.zip",
        num_dropped=num_dropped,
        dropped_labels=sorted(dropped_labels),
    )


@router.get("/exports/{run_id}.zip")
def download_export(run_id: str) -> FileResponse:
    cfg = load_config()
    zip_path = (cfg.path("paths.output_dir") / "exports" / run_id).with_suffix(".zip")
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Stats + session admin
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/projects", response_model=list[Project])
def list_projects(store: ProjectStoreDep) -> list[Project]:
    """Return every project (UI preset) from the live project store.

    The store seeds itself from ``config.yaml`` on first run and then owns the
    list, so this reflects any groups/classes the user created at runtime.
    """
    return [Project(**p) for p in store.list_projects()]


@router.post("/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(body: CreateProjectRequest, store: ProjectStoreDep) -> Project:
    """Create a brand-new group (project)."""
    try:
        project = store.create_project(
            label=body.label,
            classes=body.classes,
            default_prompts=body.default_prompts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Project(**project)


@router.post("/projects/{project_id}/classes", response_model=Project)
def add_project_class(
    project_id: str, body: AddClassRequest, store: ProjectStoreDep
) -> Project:
    """Register a class (label) under an existing group."""
    try:
        project = store.add_class(project_id, body.name)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown project_id: {project_id}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Project(**project)


@router.get("/stats", response_model=StatsResponse)
def stats(store: StoreDep) -> StatsResponse:
    annotations = store.list_annotations()
    num_approved = sum(1 for a in annotations if a.approved is True)
    num_rejected = sum(1 for a in annotations if a.approved is False)
    num_pending = sum(1 for a in annotations if a.approved is None)
    per_class: dict[str, int] = {}
    for a in annotations:
        per_class[a.label] = per_class.get(a.label, 0) + 1
    return StatsResponse(
        num_images=len(store.list_images()),
        num_annotations=len(annotations),
        num_approved=num_approved,
        num_rejected=num_rejected,
        num_pending=num_pending,
        per_class_counts=per_class,
    )


@router.delete(
    "/session", status_code=status.HTTP_204_NO_CONTENT, response_class=Response
)
def delete_session(store: StoreDep) -> Response:
    store.wipe()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ─────────────────────────────────────────────────────────────────────────────
# Small internal helper (kept out of the public surface)
# ─────────────────────────────────────────────────────────────────────────────

def _BytesIO(data: bytes):  # noqa: N802 (matches stdlib name for clarity)
    """Wrap raw bytes so SessionStore.save_image's IO[bytes] contract is satisfied."""
    import io

    return io.BytesIO(data)


__all__ = ["router", "get_session_store"]
