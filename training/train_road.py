"""Fine-tune a YOLOv8-seg model on a prepared training track.

    python -m training.train_road road_defects

Requires ``pip install -r requirements-train.txt``. Uses the GPU (device 0) if
CUDA is available, otherwise the CPU. Run :mod:`training.data_prep` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

from utils.config import load_config


def train(
    track: str,
    epochs_override: int | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Fine-tune the configured base model on ``track`` and save the run.

    ``epochs_override`` (optional CLI arg) is handy for a quick smoke run.
    ``run_dir`` overrides where the run lands (versioned training jobs each
    get their own directory); default is ``<output_dir>/train``.

    Returns:
        Path to the run's ``best.pt`` weights.
    """
    import torch  # heavy; imported only when actually training
    from ultralytics import YOLO

    cfg = load_config()
    tracks = cfg.get("training") or {}
    if track not in tracks:
        raise SystemExit(f"No training config for track '{track}'.")
    conf = tracks[track]
    root = cfg.root
    data_yaml = root / conf["work_dir"] / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"Run `python -m training.data_prep {track}` first ({data_yaml} is missing)."
        )

    if run_dir is None:
        run_dir = root / conf["output_dir"] / "train"
    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = epochs_override if epochs_override is not None else int(conf.get("epochs", 100))
    print(f"Training '{track}' on device={device} for {epochs} epochs ...")
    model = YOLO(conf.get("base_model", "yolov8n-seg.pt"))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=int(conf.get("imgsz", 640)),
        batch=int(conf.get("batch", 8)),
        seed=int(conf.get("seed", 42)),
        device=device,
        # Blackwell (sm_120) + this torch build is unstable under AMP, and the
        # Windows multiprocessing dataloader is flaky — disable both for stability.
        amp=bool(conf.get("amp", False)),
        workers=int(conf.get("workers", 0)),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )
    best = run_dir / "weights" / "best.pt"
    print(f"Done. Best weights: {best}")
    return best


def main() -> None:
    track = sys.argv[1] if len(sys.argv) > 1 else "road_defects"
    epochs_override = int(sys.argv[2]) if len(sys.argv) > 2 else None
    train(track, epochs_override)


if __name__ == "__main__":
    main()
