"""One end-to-end training job: prepare -> train -> evaluate -> gate -> deploy.

Spawned by the backend's TRAIN endpoint as a **separate process** (a CUDA crash
or OOM during training must never take the API server down), but equally
runnable by hand:

    python -m training.run_job road_defects --job-id smoke --epochs 2

Progress is written to ``<output_dir>/job_status.json`` after every phase; the
``GET /api/train/{track}/status`` route reads that file. The eval gate only
promotes the candidate to ``deployed.pt`` when it beats the live model's mask
mAP50 on the frozen test split — losing candidates are still recorded in the
registry so the attempt (and its score) stays visible.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from training import registry
from training.data_prep import prepare
from utils.config import load_config

STATUS_FILE = "job_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_job(track: str, job_id: str, epochs_override: int | None = None) -> None:
    """Run the full pipeline for one track, reporting progress via the status file."""
    cfg = load_config()
    conf = (cfg.get("training") or {}).get(track)
    if conf is None:
        raise SystemExit(f"No training config for track '{track}'.")
    root = cfg.root
    out_dir = root / conf["output_dir"]
    status_path = out_dir / STATUS_FILE
    run_dir = out_dir / "jobs" / job_id
    data_yaml = root / conf["work_dir"] / "data.yaml"
    trained_conf = (cfg.get("trained_models") or {}).get(track) or {}
    deployed_path = (
        root / trained_conf["weights"]
        if trained_conf.get("weights")
        else out_dir / "deployed.pt"
    )
    epochs = (
        epochs_override if epochs_override is not None else int(conf.get("epochs", 100))
    )

    status: dict = {
        "job_id": job_id,
        "state": "running",
        "phase": "preparing",
        "message": "Building the train/val/test split",
        "started_at": _now(),
        "finished_at": None,
        "epochs_total": epochs,
        "run_dir": str(run_dir),
        "result": None,
    }
    _write_status(status_path, status)

    try:
        # ── prepare: base dataset + the in-app snapshot (test stays frozen) ──
        extra = root / conf["app_src"] if conf.get("app_src") else None
        if extra is not None and not (extra / "images").is_dir():
            extra = None
        counts = prepare(track, extra_src=extra, config=cfg)

        # ── baseline bootstrap: adopt a pre-registry deployed.pt as v1 ──────
        reg = registry.load_registry(out_dir)
        if not reg["versions"] and deployed_path.exists():
            status.update(
                phase="evaluating",
                message="Scoring the current deployed model (baseline import)",
            )
            _write_status(status_path, status)
            from training.evaluate import evaluate_weights

            baseline_metrics = evaluate_weights(deployed_path, data_yaml)
            version = registry.add_version(
                out_dir, deployed_path, baseline_metrics, note="imported baseline"
            )
            reg = registry.load_registry(out_dir)
            reg["deployed"] = version
            registry.save_registry(out_dir, reg)

        # ── train ────────────────────────────────────────────────────────────
        status.update(
            phase="training",
            message=f"Fine-tuning for {epochs} epochs on {counts['train']} images",
        )
        _write_status(status_path, status)
        from training.train_road import train

        best = train(track, epochs_override=epochs, run_dir=run_dir)

        # ── evaluate the candidate on the frozen test split ─────────────────
        status.update(phase="evaluating", message="Scoring on the held-out test split")
        _write_status(status_path, status)
        from training.evaluate import evaluate_weights

        candidate = evaluate_weights(best, data_yaml)

        # ── gate + record (+ deploy only if better) ──────────────────────────
        status.update(phase="gating", message="Comparing against the live model")
        _write_status(status_path, status)
        reg = registry.load_registry(out_dir)
        baseline = registry.deployed_metrics(reg)
        promote = registry.is_better(candidate, baseline)
        note = f"trained {epochs} epochs on {counts['train']} train images"
        version = registry.add_version(out_dir, best, candidate, note=note)
        if promote:
            registry.deploy_version(out_dir, version, deployed_path)

        status.update(
            state="done",
            phase="done",
            finished_at=_now(),
            message=(
                f"v{version} deployed (beat the live model)"
                if promote
                else f"v{version} recorded but NOT deployed (live model is better)"
            ),
            result={
                "version": version,
                "deployed": promote,
                "candidate": {k: round(v, 4) for k, v in candidate.items()},
                "baseline": baseline,
                "counts": counts,
            },
        )
        _write_status(status_path, status)
    except Exception as exc:  # noqa: BLE001 — subprocess boundary: report, don't crash silently
        status.update(
            state="failed",
            finished_at=_now(),
            message=f"{type(exc).__name__}: {exc}",
        )
        _write_status(status_path, status)
        traceback.print_exc()
        raise SystemExit(1) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one training job end-to-end.")
    parser.add_argument("track", nargs="?", default="road_defects")
    parser.add_argument("--job-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run_job(args.track, args.job_id, args.epochs)


if __name__ == "__main__":
    main()
