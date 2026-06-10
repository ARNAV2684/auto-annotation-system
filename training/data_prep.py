"""Prepare a labelled dataset into the Ultralytics train/val/test layout.

Reads the flat ``images/`` + ``labels/`` source folder, splits the pairs
deterministically (seeded) by the configured ratios, copies them into
``images/{train,val,test}`` + ``labels/{train,val,test}`` under the work dir, and
writes the Ultralytics ``data.yaml``.

An optional **extra source** (the in-app annotation snapshot) can be merged in:
its pairs are split 90/10 into train/val only — **never test**. The test split
always comes from the same seeded shuffle of the base dataset, so every model
version is measured against the identical held-out set and eval-gate
comparisons stay apples-to-apples.

    python -m training.data_prep road_defects [--with-app]
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import yaml

from utils.config import load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_EXTRA_TRAIN_FRACTION = 0.9  # app-snapshot pairs go 90/10 into train/val


def _track_config(track: str, cfg) -> dict:
    """Return the ``training.<track>`` config block, or exit with a clear message."""
    tracks = cfg.get("training") or {}
    if track not in tracks:
        raise SystemExit(f"No training config for track '{track}'. Add it under `training:`.")
    return tracks[track]


def _pairs(src: Path) -> list[tuple[Path, Path]]:
    """Return ``(image, label)`` pairs where both files exist."""
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs


def prepare(track: str, extra_src: Path | None = None, config=None) -> dict[str, int]:
    """Build the train-ready split + data.yaml for ``track``.

    Args:
        track: Training track name (a key under ``training:`` in config).
        extra_src: Optional second ``images/`` + ``labels/`` source (the in-app
            snapshot) merged into train/val only.
        config: Optional config override (tests); defaults to the project config.

    Returns:
        Pair counts per split, e.g. ``{"train": 170, "val": 21, "test": 19}``.
    """
    cfg = config or load_config()
    conf = _track_config(track, cfg)
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

    # In-app snapshot: extra training signal only. The test split must stay the
    # frozen base-dataset sample or version-to-version metrics stop meaning
    # anything.
    extra_pairs = _pairs(extra_src) if extra_src else []
    if extra_pairs:
        random.Random(seed).shuffle(extra_pairs)
        cut = int(len(extra_pairs) * _EXTRA_TRAIN_FRACTION)
        cut = max(1, cut) if len(extra_pairs) > 1 else len(extra_pairs)
        splits["train"] = splits["train"] + extra_pairs[:cut]
        splits["val"] = splits["val"] + extra_pairs[cut:]

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

    counts = {split: len(items) for split, items in splits.items()}
    print(
        f"Prepared '{track}': {total} base + {len(extra_pairs)} app pairs -> "
        f"train {counts['train']}, val {counts['val']}, test {counts['test']}"
    )
    print(f"Dataset yaml: {work / 'data.yaml'}")
    return counts


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    track = args[0] if args else "road_defects"
    extra: Path | None = None
    if "--with-app" in sys.argv:
        cfg = load_config()
        conf = _track_config(track, cfg)
        if conf.get("app_src"):
            extra = cfg.root / conf["app_src"]
    prepare(track, extra_src=extra)


if __name__ == "__main__":
    main()
