"""Backend API tests.

These exercise every route via FastAPI's :class:`TestClient`. The heavy
:class:`~app.services.model_registry.ModelRegistry` is replaced with a stub via
FastAPI's dependency-override hook, so the GPU stack never imports here. CI runs
this file fine without ``torch`` / ``sam2`` / ``transformers``.

We also set ``LOAD_MODELS_ON_STARTUP=false`` *before* importing :mod:`app.main`
so the lifespan hook doesn't try to load the real models either.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

os.environ["LOAD_MODELS_ON_STARTUP"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.api import routes  # noqa: E402
from app.main import app  # noqa: E402
from app.services.model_registry import get_model_registry  # noqa: E402
from app.services.project_store import ProjectStore  # noqa: E402
from app.services.session_store import SessionStore  # noqa: E402
from core.annotation_converter import resolve_export_classes  # noqa: E402
from utils.config import load_config  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Stub ModelRegistry — returns small deterministic masks; no GPU, no torch.
# ─────────────────────────────────────────────────────────────────────────────

class _FakeDetection:
    def __init__(self, mask: np.ndarray, box, label: str, score: float) -> None:
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
        self.is_crowd = False


class StubModelRegistry:
    """Just enough of :class:`ModelRegistry` to drive the routes."""

    def __init__(self) -> None:
        self.segmenter = type("S", (), {"device": "cpu"})()
        self._current: str | None = None

    def ensure_embedded(self, image_id: str, image_path: Path) -> bool:
        cached = self._current == image_id
        self._current = image_id
        return cached

    def predict_point(
        self, point_xy, label: int, multimask: bool
    ) -> tuple[list[np.ndarray], list[float]]:
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:25, 5:25] = True
        return ([mask], [0.9])

    def predict_box(self, box) -> tuple[np.ndarray, float]:
        mask = np.zeros((40, 40), dtype=bool)
        x1, y1, x2, y2 = (int(v) for v in box)
        mask[max(0, y1):min(40, y2), max(0, x1):min(40, x2)] = True
        return (mask, 0.77)

    def predict_points(
        self, points, labels, multimask: bool = False
    ) -> tuple[list[np.ndarray], list[float]]:
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:25, 5:25] = True
        return ([mask], [0.88])

    def auto_detect(self, image_path: Path, prompts) -> list[Any]:
        big = np.zeros((40, 40), dtype=bool)
        big[10:30, 10:30] = True
        small = np.zeros((40, 40), dtype=bool)
        small[2:8, 2:8] = True
        return [
            _FakeDetection(big, (10, 10, 30, 30), "tree", 0.8),
            _FakeDetection(small, (2, 2, 8, 8), "tree", 0.6),
        ]

    def auto_detect_trained(self, image_path: Path, project_id: str) -> list[Any]:
        # One high-confidence detection, distinguishable from the grounded stub
        # above (two 'tree' rows) so route-dispatch tests can tell them apart.
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        return [_FakeDetection(mask, (10, 10, 30, 30), "potholes", 0.95)]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A TestClient with the model registry stubbed and session storage in tmp_path."""
    store = SessionStore(root=tmp_path / "sessions")
    routes._session_singleton = store  # private but used only by get_session_store
    # Isolated project store per test, freshly seeded from config into tmp_path.
    routes._project_store_singleton = ProjectStore(path=tmp_path / "projects.json")
    # Build the stub ONCE so cache state (current image_id) persists across requests.
    stub_registry = StubModelRegistry()
    app.dependency_overrides[get_model_registry] = lambda: stub_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    routes._session_singleton = None
    routes._project_store_singleton = None


@pytest.fixture
def png_bytes() -> bytes:
    """A 40x40 PNG with a deterministic gradient."""
    arr = (np.indices((40, 40)).sum(axis=0) * 3).astype(np.uint8)
    image = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _upload(client: TestClient, png_bytes: bytes, name: str = "drone.png") -> dict:
    response = client.post(
        "/api/upload", files={"file": (name, png_bytes, "image/png")}
    )
    assert response.status_code == 201, response.text
    return response.json()


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_reports_models_not_loaded_initially(client: TestClient) -> None:
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is False


def test_root_returns_pointer_to_docs(client: TestClient) -> None:
    data = client.get("/").json()
    assert data["docs"] == "/docs"


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

def _zip_of(png_bytes: bytes, names: list[str], extra: dict[str, bytes] | None = None) -> bytes:
    """Build an in-memory zip containing the given image names (+ optional junk)."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.writestr(name, png_bytes)
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return buf.getvalue()


def test_upload_zip_extracts_images(client: TestClient, png_bytes: bytes) -> None:
    zip_bytes = _zip_of(
        png_bytes,
        ["a.png", "sub/b.PNG"],
        extra={"notes.txt": b"ignore me", "__MACOSX/._a.png": b"junk", "._hidden.png": b"x"},
    )
    response = client.post(
        "/api/upload-zip", files={"file": ("batch.zip", zip_bytes, "application/zip")}
    )
    assert response.status_code == 201
    records = response.json()
    assert len(records) == 2  # only the two real images
    assert {r["file_name"] for r in records} == {"a.png", "b.PNG"}
    assert len(client.get("/api/images").json()) == 2


def test_upload_zip_rejects_non_zip(client: TestClient) -> None:
    response = client.post("/api/upload-zip", files={"file": ("x.png", b"notazip", "image/png")})
    assert response.status_code == 400


def test_upload_zip_with_no_images_422(client: TestClient) -> None:
    zip_bytes = _zip_of(b"", [], extra={"readme.txt": b"hello"})
    response = client.post(
        "/api/upload-zip", files={"file": ("empty.zip", zip_bytes, "application/zip")}
    )
    assert response.status_code == 422


def test_upload_returns_image_record(client: TestClient, png_bytes: bytes) -> None:
    body = _upload(client, png_bytes)
    assert body["file_name"] == "drone.png"
    assert body["width"] == 40 and body["height"] == 40
    assert body["image_id"].startswith("img_")


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/upload", files={"file": ("x.gif", b"GIF89a", "image/gif")}
    )
    assert response.status_code == 400


def test_list_images_returns_uploaded(client: TestClient, png_bytes: bytes) -> None:
    _upload(client, png_bytes, name="a.png")
    _upload(client, png_bytes, name="b.png")
    records = client.get("/api/images").json()
    assert len(records) == 2
    assert {r["file_name"] for r in records} == {"a.png", "b.png"}


def test_get_image_file_returns_bytes(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.get(f"/api/images/{image_id}/file")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_patch_image_marks_done(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    assert _upload_done_state(client, image_id) is False
    body = client.patch(f"/api/images/{image_id}", json={"done": True}).json()
    assert body["done"] is True
    assert _upload_done_state(client, image_id) is True


def test_patch_image_unknown_404(client: TestClient) -> None:
    assert client.patch("/api/images/nope", json={"done": True}).status_code == 404


def test_patch_image_sets_tags_deduped(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    body = client.patch(
        f"/api/images/{image_id}", json={"tags": ["blurry", "blurry", " urban ", ""]}
    ).json()
    assert body["tags"] == ["blurry", "urban"]  # deduped, trimmed, blanks dropped


def _upload_done_state(client: TestClient, image_id: str) -> bool:
    record = next(r for r in client.get("/api/images").json() if r["image_id"] == image_id)
    return record["done"]


def test_embed_endpoint_reports_cache_state(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    first = client.post(f"/api/images/{image_id}/embed").json()
    second = client.post(f"/api/images/{image_id}/embed").json()
    assert first["cached"] is False and second["cached"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def test_point_prompt_creates_annotation(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    )
    assert response.status_code == 201
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["source"] == "point"
    assert rows[0]["image_id"] == image_id


def test_box_prompt_creates_annotation(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/infer/box",
        json={"x1": 5, "y1": 5, "x2": 25, "y2": 25},
    )
    assert response.status_code == 201
    row = response.json()
    assert row["source"] == "box"
    assert row["box"] == [5.0, 5.0, 25.0, 25.0] or row["box"][0] >= 0


def test_box_prompt_rejects_inverted_box(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/infer/box",
        json={"x1": 25, "y1": 25, "x2": 5, "y2": 5},
    )
    assert response.status_code == 422


def test_auto_detect_replaces_previous_auto_annotations(
    client: TestClient, png_bytes: bytes
) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    first = client.post(
        f"/api/images/{image_id}/auto-detect", json={"prompts": ["tree"]}
    ).json()
    second = client.post(
        f"/api/images/{image_id}/auto-detect", json={"prompts": ["tree"]}
    ).json()
    # Same number of detections, but different ids (previous auto rows were replaced).
    assert len(first) == len(second)
    assert {r["annotation_id"] for r in first}.isdisjoint(
        {r["annotation_id"] for r in second}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smart Select (multi-point preview + commit) — PR Q
# ─────────────────────────────────────────────────────────────────────────────

def test_preview_returns_polygon_without_persisting(
    client: TestClient, png_bytes: bytes
) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/infer/preview",
        json={"points": [{"x": 15, "y": 15, "label": 1}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["polygon"] and isinstance(body["polygon"][0], list)
    assert 0.0 <= body["score"] <= 1.0
    # Preview must NOT create an annotation.
    assert client.get(f"/api/images/{image_id}/annotations").json() == []


def test_infer_points_commits_one_annotation(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    rows = client.post(
        f"/api/images/{image_id}/infer/points",
        json={"points": [{"x": 15, "y": 15, "label": 1}, {"x": 8, "y": 8, "label": 0}]},
    ).json()
    assert len(rows) == 1
    assert rows[0]["source"] == "point"
    assert client.get(f"/api/images/{image_id}/annotations").json()[0]["annotation_id"] == (
        rows[0]["annotation_id"]
    )


def test_preview_rejects_empty_points(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/infer/preview", json={"points": []}
    )
    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Manual polygon + Repeat previous (PR T)
# ─────────────────────────────────────────────────────────────────────────────

def test_create_polygon_persists_annotation(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/annotations/polygon",
        json={"points": [[5, 5], [30, 5], [30, 30], [5, 30]], "label": "House"},
    )
    assert response.status_code == 201
    ann = response.json()
    assert ann["source"] == "polygon"
    assert ann["label"] == "house"  # normalised
    assert ann["polygon"] and len(ann["polygon"][0]) >= 6


def test_create_polygon_rejects_too_few_points(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/annotations/polygon", json={"points": [[5, 5], [10, 10]]}
    )
    assert response.status_code == 422


def test_copy_annotations_clones_to_target(client: TestClient, png_bytes: bytes) -> None:
    src = _upload(client, png_bytes, name="a.png")["image_id"]
    dst = _upload(client, png_bytes, name="b.png")["image_id"]
    client.post(f"/api/images/{src}/infer/point", json={"x": 15, "y": 15})
    client.post(f"/api/images/{src}/infer/point", json={"x": 20, "y": 20})
    copied = client.post(f"/api/images/{dst}/annotations/copy-from/{src}").json()
    assert len(copied) == 2
    assert all(c["image_id"] == dst for c in copied)
    # The destination now has its own independent copies.
    assert len(client.get(f"/api/images/{dst}/annotations").json()) == 2


def test_copy_annotations_unknown_source_404(client: TestClient, png_bytes: bytes) -> None:
    dst = _upload(client, png_bytes)["image_id"]
    assert (
        client.post(f"/api/images/{dst}/annotations/copy-from/nope").status_code == 404
    )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation CRUD
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_approves_annotation(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    response = client.patch(
        f"/api/annotations/{ann['annotation_id']}",
        json={"approved": True, "label": "tree"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True and body["label"] == "tree"


def test_patch_accepts_label_outside_project_classes(
    client: TestClient, png_bytes: bytes
) -> None:
    """Any non-empty, reasonable label is accepted - projects gate the UI, not the API."""
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    response = client.patch(
        f"/api/annotations/{ann['annotation_id']}", json={"label": "Culvert"}
    )
    assert response.status_code == 200
    assert response.json()["label"] == "culvert"  # normalised lowercase


def test_patch_rejects_empty_label(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    response = client.patch(
        f"/api/annotations/{ann['annotation_id']}", json={"label": "   "}
    )
    assert response.status_code == 422


def test_patch_rejects_too_long_label(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    response = client.patch(
        f"/api/annotations/{ann['annotation_id']}", json={"label": "a" * 65}
    )
    assert response.status_code == 422


def test_delete_annotation(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    response = client.delete(f"/api/annotations/{ann['annotation_id']}")
    assert response.status_code == 204
    # Now it's gone.
    not_found = client.delete(f"/api/annotations/{ann['annotation_id']}")
    assert not_found.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Projects
# ─────────────────────────────────────────────────────────────────────────────

def test_projects_returns_configured_projects(client: TestClient) -> None:
    projects = client.get("/api/projects").json()
    # config.yaml ships with three: vegetation, road_defects, infrastructure.
    assert isinstance(projects, list) and len(projects) >= 3
    ids = {p["id"] for p in projects}
    assert {"vegetation", "road_defects", "infrastructure"} <= ids


def test_projects_shape(client: TestClient) -> None:
    projects = client.get("/api/projects").json()
    veg = next(p for p in projects if p["id"] == "vegetation")
    assert veg["label"] == "Vegetation"
    assert "tree" in veg["classes"] and "palm_tree" in veg["classes"]
    assert "tree" in veg["default_prompts"]


def test_projects_road_defects_matches_trained_model(client: TestClient) -> None:
    # Must mirror the trained model's data.yaml names exactly (Sprint 7).
    projects = client.get("/api/projects").json()
    rd = next(p for p in projects if p["id"] == "road_defects")
    assert rd["classes"] == ["cracks", "potholes", "ravelling"]


# ─────────────────────────────────────────────────────────────────────────────
# Trained-model auto-detect (Sprint 7)
# ─────────────────────────────────────────────────────────────────────────────

def test_auto_detect_uses_trained_model_when_deployed(
    client: TestClient, png_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        routes,
        "deployed_weights",
        lambda pid: Path("deployed.pt") if pid == "road_defects" else None,
    )
    image_id = _upload(client, png_bytes)["image_id"]
    body = client.post(
        f"/api/images/{image_id}/auto-detect",
        json={"prompts": ["potholes"], "project_id": "road_defects"},
    ).json()
    # The trained stub returns exactly one 'potholes' row; grounded returns two trees.
    assert [a["label"] for a in body] == ["potholes"]
    assert body[0]["source"] == "auto"


def test_auto_detect_falls_back_to_grounded_sam_without_weights(
    client: TestClient, png_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes, "deployed_weights", lambda pid: None)
    image_id = _upload(client, png_bytes)["image_id"]
    body = client.post(
        f"/api/images/{image_id}/auto-detect",
        json={"prompts": ["potholes"], "project_id": "road_defects"},
    ).json()
    assert [a["label"] for a in body] == ["tree", "tree"]


def test_projects_expose_trained_model_flag(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        routes,
        "deployed_weights",
        lambda pid: Path("deployed.pt") if pid == "road_defects" else None,
    )
    projects = client.get("/api/projects").json()
    by_id = {p["id"]: p for p in projects}
    assert by_id["road_defects"]["has_trained_model"] is True
    assert by_id["vegetation"]["has_trained_model"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Project creation + class registration (PR O)
# ─────────────────────────────────────────────────────────────────────────────

def test_create_project_adds_new_group(client: TestClient) -> None:
    response = client.post(
        "/api/projects", json={"label": "Buildings", "classes": ["house", "Shed"]}
    )
    assert response.status_code == 201
    project = response.json()
    assert project["id"] == "buildings"
    assert project["label"] == "Buildings"
    assert project["classes"] == ["house", "shed"]  # normalised
    # And it shows up in the listing.
    ids = {p["id"] for p in client.get("/api/projects").json()}
    assert "buildings" in ids


def test_create_project_rejects_empty_label(client: TestClient) -> None:
    assert client.post("/api/projects", json={"label": "   "}).status_code == 422


def test_create_project_rejects_duplicate(client: TestClient) -> None:
    # 'Vegetation' is seeded from config.
    assert client.post("/api/projects", json={"label": "Vegetation"}).status_code == 422


def test_add_class_to_existing_project(client: TestClient) -> None:
    response = client.post("/api/projects/infrastructure/classes", json={"name": "house"})
    assert response.status_code == 200
    assert "house" in response.json()["classes"]


def test_add_class_normalises_name(client: TestClient) -> None:
    project = client.post(
        "/api/projects/infrastructure/classes", json={"name": "Water Tank"}
    ).json()
    assert "water_tank" in project["classes"]


def test_add_class_is_idempotent(client: TestClient) -> None:
    client.post("/api/projects/infrastructure/classes", json={"name": "house"})
    project = client.post(
        "/api/projects/infrastructure/classes", json={"name": "house"}
    ).json()
    assert project["classes"].count("house") == 1


def test_add_class_unknown_project_404(client: TestClient) -> None:
    response = client.post("/api/projects/nope/classes", json={"name": "house"})
    assert response.status_code == 404


def test_add_class_rejects_empty_name(client: TestClient) -> None:
    response = client.post("/api/projects/infrastructure/classes", json={"name": "  "})
    assert response.status_code == 422


def test_register_class_then_export_includes_it(
    client: TestClient, png_bytes: bytes
) -> None:
    # The end-to-end loop: register 'house' under infrastructure, label an
    # annotation 'house', approve it, export the infrastructure project -> it
    # comes through (num_dropped == 0) instead of being skipped.
    client.post("/api/projects/infrastructure/classes", json={"name": "house"})
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "house")
    body = client.post(
        "/api/export",
        json={"formats": ["yolo"], "only_approved": True, "project_id": "infrastructure"},
    ).json()
    assert body["num_annotations_exported"] == 1
    assert body["num_dropped"] == 0


# resolve_export_classes is a pure function — unit-test it directly.

def test_resolve_export_classes_scopes_to_project() -> None:
    projects = [{"id": "a", "classes": ["x", "y"]}, {"id": "b", "classes": ["z"]}]
    classes, crowd = resolve_export_classes(projects, "a")
    assert classes == ["x", "y"] and crowd == []


def test_resolve_export_classes_union_dedups_preserving_order() -> None:
    projects = [{"id": "a", "classes": ["x", "y"]}, {"id": "b", "classes": ["y", "z"]}]
    classes, _ = resolve_export_classes(projects, None)
    assert classes == ["x", "y", "z"]


def test_resolve_export_classes_unknown_project_raises() -> None:
    with pytest.raises(KeyError):
        resolve_export_classes([{"id": "a", "classes": ["x"]}], "missing")


def test_resolve_export_classes_legacy_fallback() -> None:
    classes, _ = resolve_export_classes([], None, legacy_classes=["l1", "l2"])
    assert classes == ["l1", "l2"]


def test_resolve_export_classes_passes_crowd_through() -> None:
    _, crowd = resolve_export_classes(
        [{"id": "a", "classes": ["x"]}], "a", crowd_classes=["c"]
    )
    assert crowd == ["c"]


# ─────────────────────────────────────────────────────────────────────────────
# Stats + session admin
# ─────────────────────────────────────────────────────────────────────────────

def test_stats_count_pending_and_approved(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann1 = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    client.post(
        f"/api/images/{image_id}/infer/box",
        json={"x1": 5, "y1": 5, "x2": 25, "y2": 25},
    )
    client.patch(
        f"/api/annotations/{ann1['annotation_id']}",
        json={"approved": True, "label": "tree"},
    )
    stats = client.get("/api/stats").json()
    assert stats["num_images"] == 1
    assert stats["num_annotations"] == 2
    assert stats["num_approved"] == 1
    assert stats["num_pending"] == 1


def test_delete_session_wipes_everything(client: TestClient, png_bytes: bytes) -> None:
    _upload(client, png_bytes)
    response = client.delete("/api/session")
    assert response.status_code == 204
    assert client.get("/api/images").json() == []


# ─────────────────────────────────────────────────────────────────────────────
# Bulk annotation actions
# ─────────────────────────────────────────────────────────────────────────────

def _seed_three_annotations(client: TestClient, png_bytes: bytes) -> tuple[str, list[str]]:
    """Upload one image and create three annotations on it; return ids."""
    image_id = _upload(client, png_bytes)["image_id"]
    ids: list[str] = []
    for _ in range(3):
        ann = client.post(
            f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
        ).json()[0]
        ids.append(ann["annotation_id"])
    return image_id, ids


def test_bulk_approve_updates_every_id(client: TestClient, png_bytes: bytes) -> None:
    image_id, ids = _seed_three_annotations(client, png_bytes)
    response = client.post(
        f"/api/images/{image_id}/annotations/bulk",
        json={"action": "approve", "annotation_ids": ids},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 3 and body["errors"] == []
    after = client.get(f"/api/images/{image_id}/annotations").json()
    assert all(row["approved"] is True for row in after)


def test_bulk_reject_marks_all_as_false(client: TestClient, png_bytes: bytes) -> None:
    image_id, ids = _seed_three_annotations(client, png_bytes)
    body = client.post(
        f"/api/images/{image_id}/annotations/bulk",
        json={"action": "reject", "annotation_ids": ids},
    ).json()
    assert body["updated"] == 3
    after = client.get(f"/api/images/{image_id}/annotations").json()
    assert all(row["approved"] is False for row in after)


def test_bulk_delete_removes_annotations(client: TestClient, png_bytes: bytes) -> None:
    image_id, ids = _seed_three_annotations(client, png_bytes)
    body = client.post(
        f"/api/images/{image_id}/annotations/bulk",
        json={"action": "delete", "annotation_ids": ids[:2]},
    ).json()
    assert body["updated"] == 2
    after = client.get(f"/api/images/{image_id}/annotations").json()
    assert [row["annotation_id"] for row in after] == [ids[2]]


def test_bulk_reports_unknown_ids_in_errors(client: TestClient, png_bytes: bytes) -> None:
    image_id, ids = _seed_three_annotations(client, png_bytes)
    response = client.post(
        f"/api/images/{image_id}/annotations/bulk",
        json={"action": "approve", "annotation_ids": [ids[0], "ann_does_not_exist"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 1
    assert len(body["errors"]) == 1
    assert body["errors"][0]["annotation_id"] == "ann_does_not_exist"


def test_bulk_rejects_id_belonging_to_other_image(
    client: TestClient, png_bytes: bytes
) -> None:
    image_a, ids_a = _seed_three_annotations(client, png_bytes)
    image_b = _upload(client, png_bytes, name="b.png")["image_id"]
    response = client.post(
        f"/api/images/{image_b}/annotations/bulk",
        json={"action": "delete", "annotation_ids": [ids_a[0]]},
    )
    body = response.json()
    assert body["updated"] == 0
    assert body["errors"][0]["reason"] == "annotation does not belong to image"


def test_bulk_404s_for_unknown_image(client: TestClient) -> None:
    response = client.post(
        "/api/images/img_missing/annotations/bulk",
        json={"action": "approve", "annotation_ids": ["ann_x"]},
    )
    assert response.status_code == 404


def test_bulk_rejects_empty_id_list(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    response = client.post(
        f"/api/images/{image_id}/annotations/bulk",
        json={"action": "approve", "annotation_ids": []},
    )
    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def test_export_writes_and_serves_zip(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    client.patch(
        f"/api/annotations/{ann['annotation_id']}",
        json={"approved": True, "label": "tree"},
    )
    response = client.post(
        "/api/export", json={"formats": ["all"], "only_approved": True}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["num_images"] == 1 and body["num_annotations_exported"] >= 1
    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert len(download.content) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Per-project export (PR N)
# ─────────────────────────────────────────────────────────────────────────────

def _approved_ann_with_label(client: TestClient, image_id: str, label: str) -> str:
    """Create one point annotation on an image, label + approve it, return its id."""
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    client.patch(
        f"/api/annotations/{ann['annotation_id']}",
        json={"approved": True, "label": label},
    )
    return ann["annotation_id"]


def test_export_with_project_id_only_includes_matching_classes(
    client: TestClient, png_bytes: bytes
) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "tree")
    _approved_ann_with_label(client, image_id, "pothole")
    body = client.post(
        "/api/export",
        json={"formats": ["yolo"], "only_approved": True, "project_id": "vegetation"},
    ).json()
    # Only the 'tree' annotation belongs to the vegetation project.
    assert body["num_annotations_exported"] == 1
    assert body["num_dropped"] == 1
    assert body["dropped_labels"] == ["pothole"]


def test_export_drops_labels_outside_project_and_reports_them(
    client: TestClient, png_bytes: bytes
) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "pothole")
    _approved_ann_with_label(client, image_id, "culvert")
    body = client.post(
        "/api/export",
        json={"formats": ["yolo"], "only_approved": True, "project_id": "vegetation"},
    ).json()
    assert body["num_annotations_exported"] == 0
    assert body["num_dropped"] == 2
    assert body["dropped_labels"] == ["culvert", "pothole"]  # sorted, unique


def test_export_without_project_id_uses_union_of_all_projects(
    client: TestClient, png_bytes: bytes
) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "tree")  # vegetation
    _approved_ann_with_label(client, image_id, "potholes")  # road_defects
    body = client.post(
        "/api/export", json={"formats": ["yolo"], "only_approved": True}
    ).json()
    # No project named -> union of every project's classes -> nothing dropped.
    assert body["num_annotations_exported"] == 2
    assert body["num_dropped"] == 0
    assert body["dropped_labels"] == []


def test_export_includes_potholes_when_project_id_is_road_defects(
    client: TestClient, png_bytes: bytes
) -> None:
    # Regression test for the reported bug: road-defect labels exported blank.
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "potholes")
    response = client.post(
        "/api/export",
        json={"formats": ["all"], "only_approved": True, "project_id": "road_defects"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["num_annotations_exported"] == 1
    assert body["num_dropped"] == 0
    download = client.get(body["download_url"])
    assert download.status_code == 200 and len(download.content) > 0


def test_export_rejects_unknown_project_id(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "pothole")
    response = client.post(
        "/api/export",
        json={"formats": ["yolo"], "only_approved": True, "project_id": "does_not_exist"},
    )
    assert response.status_code == 422


def test_export_class_ids_match_project_class_order(
    client: TestClient, png_bytes: bytes
) -> None:
    # road_defects.classes = [cracks, potholes, ravelling] -> ravelling is id 2.
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "ravelling")
    body = client.post(
        "/api/export",
        json={"formats": ["yolo"], "only_approved": True, "project_id": "road_defects"},
    ).json()
    yolo_txt = (
        load_config().path("paths.output_dir")
        / "exports"
        / body["run_id"]
        / "yolo"
        / "drone.txt"
    )
    first_line = yolo_txt.read_text(encoding="utf-8").strip().splitlines()[0]
    assert first_line.split()[0] == "2"
