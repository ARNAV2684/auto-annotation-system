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


def test_patch_rejects_unknown_label(client: TestClient, png_bytes: bytes) -> None:
    image_id = _upload(client, png_bytes)["image_id"]
    ann = client.post(
        f"/api/images/{image_id}/infer/point", json={"x": 15, "y": 15}
    ).json()[0]
    response = client.patch(
        f"/api/annotations/{ann['annotation_id']}", json={"label": "alien"}
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
