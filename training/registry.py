"""Versioned model registry for trained checkpoints (GPU-free).

Each training track keeps a ``registry.json`` next to its weights::

    models/road_defects/
      registry.json          # versions + which one is live
      v1/weights.pt          # immutable, one folder per version
      v2/weights.pt
      deployed.pt            # copy of the live version (what the app loads)

``deployed.pt`` is the contract the inference side (``core.trained_detector``)
already reads — deploying a version means copying its weights over that file.
Keeping versions immutable makes rollback a plain re-deploy of an older one.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_FILE = "registry.json"
GATE_METRIC = "seg_map50"  # mask mAP@50 — the headline segmentation metric


def load_registry(reg_dir: Path) -> dict:
    """Read ``registry.json`` under ``reg_dir`` (empty registry if absent/corrupt)."""
    path = reg_dir / REGISTRY_FILE
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw.get("versions"), list):
                return raw
        except (json.JSONDecodeError, OSError):
            pass
    return {"versions": [], "deployed": None}


def save_registry(reg_dir: Path, registry: dict) -> None:
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / REGISTRY_FILE).write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )


def add_version(
    reg_dir: Path, weights_src: Path, metrics: dict[str, float], note: str = ""
) -> int:
    """Copy ``weights_src`` in as the next immutable version. Returns its number."""
    registry = load_registry(reg_dir)
    version = max((v["version"] for v in registry["versions"]), default=0) + 1
    version_dir = reg_dir / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights_src, version_dir / "weights.pt")
    registry["versions"].append(
        {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "note": note,
        }
    )
    save_registry(reg_dir, registry)
    return version


def deploy_version(reg_dir: Path, version: int, deployed_path: Path) -> None:
    """Make ``version`` the live model: copy its weights over ``deployed_path``.

    Raises:
        KeyError: no such version in the registry.
    """
    registry = load_registry(reg_dir)
    if not any(v["version"] == version for v in registry["versions"]):
        raise KeyError(version)
    weights = reg_dir / f"v{version}" / "weights.pt"
    deployed_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, deployed_path)
    registry["deployed"] = version
    save_registry(reg_dir, registry)


def deployed_metrics(registry: dict) -> dict[str, float] | None:
    """Return the live version's metrics, or ``None`` if nothing is deployed."""
    deployed = registry.get("deployed")
    if deployed is None:
        return None
    entry = next((v for v in registry["versions"] if v["version"] == deployed), None)
    return dict(entry["metrics"]) if entry else None


def is_better(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
    metric: str = GATE_METRIC,
) -> bool:
    """The deploy gate: strictly better than the live model on ``metric``.

    No baseline (first ever model) always passes. Equal scores do NOT pass —
    churning the deployed model for no measured gain is all risk, no reward.
    """
    if baseline is None:
        return True
    return float(candidate.get(metric, 0.0)) > float(baseline.get(metric, 0.0))
