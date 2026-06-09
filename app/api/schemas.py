"""Pydantic request / response schemas for the backend API.

These are the runtime implementation of ``docs/api_contract.md`` — the single
source of truth shared with the frontend. Every shape sent over the wire is
defined here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A polygon is a flat list of pixel coordinates: [x1, y1, x2, y2, ...].
# A mask may have one or more disjoint polygons.
Polygon = list[float]
Box = tuple[float, float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: Literal["ok", "degraded"]
    device: str = Field(description="Torch device the models live on, e.g. 'cuda' / 'cpu'.")
    models_loaded: bool = Field(description="True once SAM 2 + GroundingDINO are ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

class ImageRecord(BaseModel):
    """Persisted image metadata, returned on upload and in listings."""

    image_id: str
    file_name: str
    width: int
    height: int
    num_detections: int = 0
    num_approved: int = 0
    done: bool = False
    tags: list[str] = Field(default_factory=list)


class ImagePatch(BaseModel):
    """Body for PATCH /api/images/{id}. All fields optional."""

    done: bool | None = None
    tags: list[str] | None = None


class EmbedResponse(BaseModel):
    """Returned by /api/images/{id}/embed."""

    ok: bool
    cached: bool = Field(
        description="True if the embedding was already loaded (no GPU work was done)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation (shared shape returned by every inference / CRUD endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class Annotation(BaseModel):
    """One detected-and-segmented object as exposed over the wire.

    Pixel masks are intentionally NOT sent; clients receive simplified polygons
    that render identically on a canvas.
    """

    annotation_id: str
    image_id: str
    polygon: list[Polygon] = Field(
        description="List of polygons; flat [x, y] pairs in absolute image pixels."
    )
    box: Box = Field(description="(x1, y1, x2, y2) absolute pixels.")
    label: str
    score: float = Field(ge=0.0, le=1.0)
    is_crowd: bool = False
    approved: bool | None = Field(
        default=None, description="null = pending review; true = kept; false = rejected."
    )
    source: Literal["auto", "point", "box"] = "auto"


# ─────────────────────────────────────────────────────────────────────────────
# Inference requests
# ─────────────────────────────────────────────────────────────────────────────

class PointRequest(BaseModel):
    """Body for /api/images/{id}/infer/point."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: Literal[0, 1] = Field(
        default=1, description="1 = foreground (include this), 0 = background (exclude)."
    )
    multimask: bool = Field(
        default=False,
        description="If True return up to 3 candidate masks ranked by score; else just the best.",
    )


class BoxRequest(BaseModel):
    """Body for /api/images/{id}/infer/box."""

    model_config = ConfigDict(populate_by_name=True)

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)


class PointSpec(BaseModel):
    """One include/exclude point for Smart Select."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: Literal[0, 1] = Field(
        default=1, description="1 = include (foreground), 0 = exclude (background)."
    )


class MultiPointRequest(BaseModel):
    """Body for the Smart Select preview/commit endpoints."""

    points: list[PointSpec] = Field(
        min_length=1, description="Accumulated include/exclude points."
    )
    multimask: bool = Field(default=False)
    simplify: float | None = Field(
        default=None,
        ge=0.0,
        description="Polygon simplification tolerance (px). None = config default.",
    )


class PreviewResponse(BaseModel):
    """Returned by /api/images/{id}/infer/preview — geometry only, not persisted."""

    polygon: list[Polygon] = Field(description="Flat [x, y] pairs in absolute image pixels.")
    box: Box
    score: float = Field(ge=0.0, le=1.0)


class AutoDetectRequest(BaseModel):
    """Body for /api/images/{id}/auto-detect."""

    prompts: list[str] = Field(
        min_length=1,
        description="One or more text prompts (e.g. ['tree', 'palm tree']).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation CRUD
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationPatch(BaseModel):
    """Body for PATCH /api/annotations/{id}. All fields optional."""

    approved: bool | None = None
    label: str | None = None
    is_crowd: bool | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Bulk annotation actions
# ─────────────────────────────────────────────────────────────────────────────

BulkAction = Literal["approve", "reject", "delete"]


class BulkAnnotationRequest(BaseModel):
    """Body for POST /api/images/{id}/annotations/bulk.

    One round-trip to approve, reject, or delete many annotations on a single
    image. Ids not belonging to ``image_id`` (or that don't exist) are reported
    back in ``errors`` but do NOT abort the rest of the batch.
    """

    action: BulkAction = Field(description="One of approve / reject / delete.")
    annotation_ids: list[str] = Field(
        min_length=1, description="Annotation ids to act on (must belong to the image)."
    )


class BulkAnnotationError(BaseModel):
    annotation_id: str
    reason: str


class BulkAnnotationResponse(BaseModel):
    updated: int = Field(description="Number of annotations successfully changed.")
    errors: list[BulkAnnotationError] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

ExportFormat = Literal["yolo", "coco", "voc", "all"]


class ExportRequest(BaseModel):
    """Body for POST /api/export."""

    formats: list[ExportFormat] = Field(
        default_factory=lambda: ["all"],
        description="Any subset of yolo/coco/voc, or ['all'].",
    )
    only_approved: bool = Field(
        default=True,
        description="If True (default), export only annotations with approved=True.",
    )
    project_id: str | None = Field(
        default=None,
        description=(
            "If set, scope the export to this project's class list; annotations "
            "whose label is outside it are skipped and reported in the response. "
            "If omitted, the union of every project's classes is used so nothing "
            "is silently dropped."
        ),
    )


class ExportResponse(BaseModel):
    """Returned by POST /api/export."""

    run_id: str
    num_images: int
    num_annotations_exported: int
    download_url: str
    num_dropped: int = Field(
        default=0,
        description="Annotations skipped because their label is not in the export class list.",
    )
    dropped_labels: list[str] = Field(
        default_factory=list,
        description="Sorted unique labels that were skipped (for a UI warning).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    """Returned by GET /api/stats."""

    num_images: int
    num_annotations: int
    num_approved: int
    num_rejected: int
    num_pending: int
    per_class_counts: dict[str, int]


# ─────────────────────────────────────────────────────────────────────────────
# Projects (UI presets — multiple class lists + default detection prompts)
# ─────────────────────────────────────────────────────────────────────────────

class Project(BaseModel):
    """A named class-list + default-prompt preset for the annotation UI.

    Projects are pure UI presets — they don't change storage. The same image
    can carry annotations from multiple projects (a tree mask and a culvert
    mask). Switching projects only changes which labels the dropdowns show and
    what default text fills the auto-detect box.
    """

    id: str = Field(description="Stable URL-safe id, e.g. 'road_defects'.")
    label: str = Field(description="Human-readable label, e.g. 'Road Defects'.")
    classes: list[str] = Field(
        description="Class labels available in the UI dropdown for this project."
    )
    default_prompts: list[str] = Field(
        default_factory=list,
        description="Text phrases pre-filled in the auto-detect box.",
    )


class CreateProjectRequest(BaseModel):
    """Body for POST /api/projects — create a brand-new group."""

    label: str = Field(description="Human-readable group name, e.g. 'Buildings'.")
    classes: list[str] = Field(
        default_factory=list, description="Optional initial class labels."
    )
    default_prompts: list[str] = Field(
        default_factory=list, description="Optional auto-detect prompt presets."
    )


class AddClassRequest(BaseModel):
    """Body for POST /api/projects/{id}/classes — register a class under a group."""

    name: str = Field(description="Class label to register, e.g. 'house'.")


# ─────────────────────────────────────────────────────────────────────────────
# Simple acknowledgement
# ─────────────────────────────────────────────────────────────────────────────

class Ok(BaseModel):
    ok: bool = True
