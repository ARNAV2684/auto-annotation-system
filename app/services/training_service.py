"""Bridge between the annotation session and the offline training pipeline.

Two responsibilities, both framework-agnostic (the HTTP layer adapts them):

* **Snapshot** — export the user's reviewed work (approved annotations on
  images marked *done*) into a flat YOLO-seg ``images/`` + ``labels/`` folder.
  ``training.data_prep`` then merges that snapshot into train/val.
* **Job management** — spawn ``training.run_job`` as a **separate process**
  (a CUDA crash or OOM mid-training must never take the API server down),
  remember the live handle, and surface progress from the job's status file.

Only one job per track runs at a time. Polygons are exported from the session
store's stored geometry — no mask decoding needed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from app.services.session_store import SessionStore
from core.annotation_converter import normalise_label
from utils.config import load_config

logger = logging.getLogger(__name__)

# Live job handles, keyed by track. Server-lifetime only: after a restart we
# fall back to the status file (see _status_is_fresh).
_jobs: dict[str, subprocess.Popen] = {}

# A status file claiming "running" older than this is treated as stale (the
# job process is updating it at phase boundaries; training itself takes ~20
# minutes, so a fresh file means a job is genuinely alive).
_STALE_AFTER_S = 30 * 60


def track_paths(track: str, config=None) -> tuple[Path, Path]:
    """Return ``(output_dir, deployed_weights_path)`` for a track."""
    cfg = config or load_config()
    conf = (cfg.get("training") or {})[track]
    out_dir = cfg.root / conf["output_dir"]
    trained = (cfg.get("trained_models") or {}).get(track) or {}
    deployed = (
        cfg.root / trained["weights"] if trained.get("weights") else out_dir / "deployed.pt"
    )
    return out_dir, deployed


def dataset_class_names(track: str, config=None) -> list[str]:
    """Class names (in id order) from the track's source ``data.yaml``.

    This is the single source of truth for label -> class-id mapping: the
    snapshot must use the same ids the base dataset was labelled with.
    """
    cfg = config or load_config()
    conf = (cfg.get("training") or {})[track]
    data = yaml.safe_load(
        (cfg.root / conf["dataset_src"] / "data.yaml").read_text(encoding="utf-8")
    )
    return [str(name) for name in data["names"]]


def snapshot_dataset(
    store: SessionStore, dest: Path, class_names: list[str]
) -> tuple[int, int]:
    """Export reviewed in-app work as a flat YOLO-seg dataset under ``dest``.

    Only images marked **done** contribute — an image the user hasn't finished
    reviewing would teach the model that its unmarked defects are background.
    Within those, only **approved** annotations whose label matches a dataset
    class are written. The destination is wiped first (it's a snapshot, not an
    archive), so deleted annotations never linger.

    Returns:
        ``(num_images, num_annotations)`` actually exported.
    """
    if dest.exists():
        shutil.rmtree(dest)
    images_dir = dest / "images"
    labels_dir = dest / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    class_ids = {name: i for i, name in enumerate(class_names)}
    num_images = 0
    num_annotations = 0
    for image in store.list_images():
        if not image.done:
            continue
        lines: list[str] = []
        for ann in store.list_annotations(image_id=image.image_id):
            if ann.approved is not True:
                continue
            class_id = class_ids.get(normalise_label(ann.label))
            if class_id is None or len(ann.polygon) < 3:
                continue
            coords = []
            for x, y in ann.polygon:
                coords.append(f"{min(max(x / image.width, 0.0), 1.0):.6f}")
                coords.append(f"{min(max(y / image.height, 0.0), 1.0):.6f}")
            lines.append(f"{class_id} " + " ".join(coords))
        if not lines:
            continue
        src_path = store.get_image_path(image.image_id)
        if src_path is None:
            continue
        shutil.copy(src_path, images_dir / src_path.name)
        (labels_dir / f"{src_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        num_images += 1
        num_annotations += len(lines)
    logger.info(
        "Snapshot: %d images / %d annotations -> %s", num_images, num_annotations, dest
    )
    return num_images, num_annotations


def job_running(track: str, config=None) -> bool:
    """True if a training job for ``track`` is (very likely) alive right now."""
    handle = _jobs.get(track)
    if handle is not None and handle.poll() is None:
        return True
    return _status_is_fresh(track, config)


def spawn_job(track: str, config=None) -> str:
    """Launch ``training.run_job`` for ``track`` in the background; return its id."""
    cfg = config or load_config()
    out_dir, _ = track_paths(track, cfg)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{job_id}.log").open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "training.run_job", track, "--job-id", job_id],
            cwd=cfg.root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_file.close()  # the child holds its own handle from here
    _jobs[track] = proc
    logger.info("Spawned training job %s for '%s' (pid %d)", job_id, track, proc.pid)
    return job_id


def read_status(track: str, config=None) -> dict:
    """Current job status for the UI: the status file + live epoch progress."""
    out_dir, _ = track_paths(track, config)
    path = out_dir / "job_status.json"
    if not path.exists():
        return {"state": "idle"}
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "idle"}
    if status.get("state") == "running":
        status["epochs_done"] = _epochs_done(status.get("run_dir"))
        # A "running" file with no live process and no recent writes means the
        # job died hard (machine reboot, kill). Surface that instead of an
        # eternal spinner.
        handle = _jobs.get(track)
        alive = handle is not None and handle.poll() is None
        if not alive and not _status_is_fresh(track, config):
            status["state"] = "failed"
            status["message"] = (
                "The training process is no longer running. Check the job log "
                "under models/.../jobs/."
            )
    return status


def _status_is_fresh(track: str, config=None) -> bool:
    """True if the status file claims *running* and was written recently."""
    out_dir, _ = track_paths(track, config)
    path = out_dir / "job_status.json"
    if not path.exists():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if status.get("state") != "running":
        return False
    return (time.time() - path.stat().st_mtime) < _STALE_AFTER_S


def _epochs_done(run_dir: str | None) -> int | None:
    """Epochs completed so far, read from Ultralytics' per-epoch results.csv."""
    if not run_dir:
        return None
    results = Path(run_dir) / "results.csv"
    if not results.exists():
        return 0
    try:
        lines = results.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    return max(0, len(lines) - 1)  # first line is the header
