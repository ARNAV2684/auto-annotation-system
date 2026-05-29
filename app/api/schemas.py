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


class ExportResponse(BaseModel):
    """Returned by POST /api/export."""

    run_id: str
    num_images: int
    num_annotations_exported: int
    download_url: str


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
# Simple acknowledgement
# ─────────────────────────────────────────────────────────────────────────────

class Ok(BaseModel):
    ok: bool = True
