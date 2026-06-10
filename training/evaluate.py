"""Evaluate a trained model on the held-out test split.

    python -m training.evaluate road_defects

Prints box + mask mAP so a model can be compared against a previous one before
it is deployed into the app (the "only deploy if better" gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

from utils.config import load_config


def evaluate_weights(weights: Path, data_yaml: Path) -> dict[str, float]:
    """Validate one checkpoint on the test split; return the headline metrics.

    The returned keys (``seg_map50`` etc.) are the registry/eval-gate contract.
    """
    from ultralytics import YOLO  # heavy; imported only when actually evaluating

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="test", verbose=False)
    return {
        "box_map50": float(metrics.box.map50),
        "box_map5095": float(metrics.box.map),
        "seg_map50": float(metrics.seg.map50),
        "seg_map5095": float(metrics.seg.map),
    }


def evaluate(track: str) -> None:
    """Run validation on the test split and print the headline metrics."""
    cfg = load_config()
    tracks = cfg.get("training") or {}
    if track not in tracks:
        raise SystemExit(f"No training config for track '{track}'.")
    conf = tracks[track]
    root = cfg.root
    weights = root / conf["output_dir"] / "train" / "weights" / "best.pt"
    data_yaml = root / conf["work_dir"] / "data.yaml"
    if not weights.exists():
        raise SystemExit(f"No trained weights at {weights}. Train first.")

    metrics = evaluate_weights(weights, data_yaml)
    print(f"Test metrics for '{track}':")
    print(f"  box  mAP50:    {metrics['box_map50']:.3f}")
    print(f"  box  mAP50-95: {metrics['box_map5095']:.3f}")
    print(f"  mask mAP50:    {metrics['seg_map50']:.3f}")
    print(f"  mask mAP50-95: {metrics['seg_map5095']:.3f}")


def main() -> None:
    track = sys.argv[1] if len(sys.argv) > 1 else "road_defects"
    evaluate(track)


if __name__ == "__main__":
    main()
