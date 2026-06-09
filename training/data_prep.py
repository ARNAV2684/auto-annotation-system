"""Prepare a labelled dataset into the Ultralytics train/val/test layout.

Reads the flat ``images/`` + ``labels/`` source folder, splits the pairs
deterministically (seeded) by the configured ratios, copies them into
``images/{train,val,test}`` + ``labels/{train,val,test}`` under the work dir, and
writes the Ultralytics ``data.yaml``.

    python -m training.data_prep road_defects
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import yaml

from utils.config import load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _track_config(track: str) -> dict:
    """Return the ``training.<track>`` config block, or exit with a clear message."""
    tracks = load_config().get("training") or {}
    if track not in tracks:
        raise SystemExit(f"No training config for track '{track}'. Add it under `training:`.")
    return tracks[track]


def _pairs(src: Path) -> list[tuple[Path, Path]]:
    """Return ``(image, label)`` pairs where both files exist."""
    images_dir = src / "images"
    labels_dir = src / "labels"
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs


def prepare(track: str) -> None:
    """Build the train-ready split + data.yaml for ``track``."""
    cfg = load_config()
    conf = _track_config(track)
    root = cfg.root
    src = root / conf["dataset_src"]
    work = root / conf["work_dir"]
    ratios = conf.get("split", [0.8, 0.1, 0.1])
    seed = int(conf.get("seed", 42))

    pairs = _pairs(src)
    if not pairs:
        raise SystemExit(f"No image/label pairs found in {src}")
    random.Random(seed).shuffle(pairs)

    total = len(pairs)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    if work.exists():
        shutil.rmtree(work)
    for split, items in splits.items():
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image, label in items:
            shutil.copy(image, work / "images" / split / image.name)
            shutil.copy(label, work / "labels" / split / label.name)

    names = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]
    data_yaml = {
        "path": str(work.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    (work / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    print(
        f"Prepared '{track}': {total} pairs -> "
        f"train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}"
    )
    print(f"Dataset yaml: {work / 'data.yaml'}")


def main() -> None:
    track = sys.argv[1] if len(sys.argv) > 1 else "road_defects"
    prepare(track)


if __name__ == "__main__":
    main()
