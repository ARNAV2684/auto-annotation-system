"""Unit tests for :mod:`app.services.training_service` (GPU-free)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.services import training_service
from app.services.session_store import AnnotationRow, SessionStore
from utils.config import Config

CLASSES = ["cracks", "potholes", "ravelling"]


def _png_bytes(size: int = 40) -> bytes:
    arr = (np.indices((size, size)).sum(axis=0) * 3).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _store_with_image(tmp_path: Path, done: bool = True) -> tuple[SessionStore, str]:
    store = SessionStore(root=tmp_path / "sessions")
    row = store.save_image(io.BytesIO(_png_bytes()), "drone.png")
    store.update_image(row.image_id, done=done)
    return store, row.image_id


def _ann(
    store: SessionStore,
    image_id: str,
    label: str,
    approved: bool | None = True,
    polygon: list[list[float]] | None = None,
) -> None:
    store.add_annotation(
        AnnotationRow(
            annotation_id=store.make_annotation_id(),
            image_id=image_id,
            polygon=polygon if polygon is not None else [[4, 4], [20, 4], [20, 20]],
            box=(4.0, 4.0, 20.0, 20.0),
            label=label,
            score=0.9,
            approved=approved,
        )
    )


# ─── snapshot_dataset ─────────────────────────────────────────────────────


def test_snapshot_exports_approved_annotations_from_done_images(tmp_path: Path) -> None:
    store, image_id = _store_with_image(tmp_path)
    _ann(store, image_id, "potholes")
    _ann(store, image_id, "cracks")
    dest = tmp_path / "snap"
    n_images, n_anns = training_service.snapshot_dataset(store, dest, CLASSES)
    assert (n_images, n_anns) == (1, 2)
    label_file = next((dest / "labels").iterdir())
    lines = label_file.read_text().strip().splitlines()
    # class ids follow the dataset order: potholes=1, cracks=0.
    assert [line.split()[0] for line in lines] == ["1", "0"]
    # Coordinates are normalised by the 40x40 image size: 4px -> 0.1.
    assert lines[0].split()[1] == "0.100000"
    assert len(list((dest / "images").iterdir())) == 1


def test_snapshot_skips_unapproved_unknown_and_degenerate(tmp_path: Path) -> None:
    store, image_id = _store_with_image(tmp_path)
    _ann(store, image_id, "potholes", approved=None)  # pending
    _ann(store, image_id, "potholes", approved=False)  # rejected
    _ann(store, image_id, "house")  # not a dataset class
    _ann(store, image_id, "cracks", polygon=[[1, 1], [2, 2]])  # < 3 points
    n_images, n_anns = training_service.snapshot_dataset(store, tmp_path / "s", CLASSES)
    assert (n_images, n_anns) == (0, 0)


def test_snapshot_skips_images_not_marked_done(tmp_path: Path) -> None:
    store, image_id = _store_with_image(tmp_path, done=False)
    _ann(store, image_id, "potholes")
    n_images, _ = training_service.snapshot_dataset(store, tmp_path / "s", CLASSES)
    assert n_images == 0


def test_snapshot_wipes_previous_contents(tmp_path: Path) -> None:
    store, image_id = _store_with_image(tmp_path)
    _ann(store, image_id, "potholes")
    dest = tmp_path / "snap"
    stale = dest / "labels" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("0 0 0 0 0 0 0\n")
    training_service.snapshot_dataset(store, dest, CLASSES)
    assert not stale.exists()


# ─── read_status / track_paths ────────────────────────────────────────────


def _track_config(tmp_path: Path) -> Config:
    return Config({"training": {"t": {"output_dir": "m"}}}, tmp_path)


def test_read_status_idle_when_no_file(tmp_path: Path) -> None:
    assert training_service.read_status("t", _track_config(tmp_path)) == {
        "state": "idle"
    }


def test_read_status_running_reports_epoch_progress(tmp_path: Path) -> None:
    cfg = _track_config(tmp_path)
    out_dir = tmp_path / "m"
    run_dir = out_dir / "jobs" / "j1"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text("header\nepoch1\nepoch2\n")
    (out_dir / "job_status.json").write_text(
        json.dumps(
            {"state": "running", "phase": "training", "run_dir": str(run_dir)}
        )
    )
    status = training_service.read_status("t", cfg)
    assert status["state"] == "running" and status["epochs_done"] == 2


def test_read_status_marks_stale_running_job_as_failed(tmp_path: Path, monkeypatch) -> None:
    cfg = _track_config(tmp_path)
    out_dir = tmp_path / "m"
    out_dir.mkdir(parents=True)
    (out_dir / "job_status.json").write_text(json.dumps({"state": "running"}))
    monkeypatch.setattr(training_service, "_STALE_AFTER_S", -1)  # everything is stale
    status = training_service.read_status("t", cfg)
    assert status["state"] == "failed"


def test_track_paths_defaults_deployed_under_output_dir(tmp_path: Path) -> None:
    out_dir, deployed = training_service.track_paths("t", _track_config(tmp_path))
    assert out_dir == tmp_path / "m"
    assert deployed == tmp_path / "m" / "deployed.pt"


def test_track_paths_honours_trained_models_block(tmp_path: Path) -> None:
    cfg = Config(
        {
            "training": {"t": {"output_dir": "m"}},
            "trained_models": {"t": {"weights": "m/live.pt"}},
        },
        tmp_path,
    )
    _, deployed = training_service.track_paths("t", cfg)
    assert deployed == tmp_path / "m" / "live.pt"
