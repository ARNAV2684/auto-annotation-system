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
from app.services.session_store import SessionStore  # noqa: E402
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

    def auto_detect(self, image_path: Path, prompts) -> list[Any]:
        big = np.zeros((40, 40), dtype=bool)
        big[10:30, 10:30] = True
        small = np.zeros((40, 40), dtype=bool)
        small[2:8, 2:8] = True
        return [
            _FakeDetection(big, (10, 10, 30, 30), "tree", 0.8),
            _FakeDetection(small, (2, 2, 8, 8), "tree", 0.6),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A TestClient with the model registry stubbed and session storage in tmp_path."""
    store = SessionStore(root=tmp_path / "sessions")
    routes._session_singleton = store  # private but used only by get_session_store
    # Build the stub ONCE so cache state (current image_id) persists across requests.
    stub_registry = StubModelRegistry()
    app.dependency_overrides[get_model_registry] = lambda: stub_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    routes._session_singleton = None


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


def test_projects_road_defects_has_culvert(client: TestClient) -> None:
    projects = client.get("/api/projects").json()
    rd = next(p for p in projects if p["id"] == "road_defects")
    assert "culvert" in rd["classes"]
    assert "pothole" in rd["classes"]


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
    _approved_ann_with_label(client, image_id, "pothole")  # road_defects
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
    _approved_ann_with_label(client, image_id, "pothole")
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
    # road_defects.classes = [pothole, crack, culvert, ...] -> culvert is id 2.
    image_id = _upload(client, png_bytes)["image_id"]
    _approved_ann_with_label(client, image_id, "culvert")
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
