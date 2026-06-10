"""Unit tests for the model registry and the data-prep merge (GPU-free)."""

from __future__ import annotations

from pathlib import Path

import yaml

from training import registry
from training.data_prep import prepare
from utils.config import Config

# ─── registry ─────────────────────────────────────────────────────────────


def _weights(tmp_path: Path, name: str = "w.pt") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake-weights")
    return path


def test_empty_registry_when_no_file(tmp_path: Path) -> None:
    reg = registry.load_registry(tmp_path)
    assert reg == {"versions": [], "deployed": None}


def test_add_version_numbers_sequentially_and_copies_weights(tmp_path: Path) -> None:
    src = _weights(tmp_path)
    assert registry.add_version(tmp_path, src, {"seg_map50": 0.5}) == 1
    assert registry.add_version(tmp_path, src, {"seg_map50": 0.6}, note="run 2") == 2
    reg = registry.load_registry(tmp_path)
    assert [v["version"] for v in reg["versions"]] == [1, 2]
    assert reg["versions"][1]["note"] == "run 2"
    assert (tmp_path / "v1" / "weights.pt").read_bytes() == b"fake-weights"
    assert (tmp_path / "v2" / "weights.pt").exists()


def test_deploy_version_copies_weights_and_flags_live(tmp_path: Path) -> None:
    registry.add_version(tmp_path, _weights(tmp_path), {"seg_map50": 0.5})
    deployed = tmp_path / "live" / "deployed.pt"
    registry.deploy_version(tmp_path, 1, deployed)
    assert deployed.read_bytes() == b"fake-weights"
    reg = registry.load_registry(tmp_path)
    assert reg["deployed"] == 1
    assert registry.deployed_metrics(reg) == {"seg_map50": 0.5}


def test_deploy_unknown_version_raises(tmp_path: Path) -> None:
    try:
        registry.deploy_version(tmp_path, 7, tmp_path / "deployed.pt")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_gate_no_baseline_always_passes() -> None:
    assert registry.is_better({"seg_map50": 0.01}, None) is True


def test_gate_requires_strictly_better() -> None:
    baseline = {"seg_map50": 0.60}
    assert registry.is_better({"seg_map50": 0.61}, baseline) is True
    assert registry.is_better({"seg_map50": 0.60}, baseline) is False
    assert registry.is_better({"seg_map50": 0.59}, baseline) is False


# ─── data_prep merge ──────────────────────────────────────────────────────


def _make_dataset(root: Path, stems: list[str], with_yaml: bool = True) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for stem in stems:
        (root / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (root / "labels" / f"{stem}.txt").write_text("0 0.1 0.1 0.2 0.2 0.1 0.2\n")
    if with_yaml:
        (root / "data.yaml").write_text(
            yaml.safe_dump({"names": ["cracks", "potholes", "ravelling"]})
        )
    return root


def _prep_config(tmp_path: Path) -> Config:
    return Config(
        {
            "training": {
                "t": {
                    "dataset_src": "base",
                    "work_dir": "work",
                    "split": [0.8, 0.1, 0.1],
                    "seed": 42,
                }
            }
        },
        tmp_path,
    )


def test_prepare_splits_base_dataset(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "base", [f"b{i}" for i in range(10)])
    counts = prepare("t", config=_prep_config(tmp_path))
    assert counts == {"train": 8, "val": 1, "test": 1}
    data = yaml.safe_load((tmp_path / "work" / "data.yaml").read_text())
    assert data["names"] == ["cracks", "potholes", "ravelling"]


def test_prepare_merges_app_snapshot_into_train_val_only(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "base", [f"b{i}" for i in range(10)])
    extra = _make_dataset(
        tmp_path / "app", [f"e{i}" for i in range(5)], with_yaml=False
    )
    counts = prepare("t", extra_src=extra, config=_prep_config(tmp_path))
    assert counts["train"] + counts["val"] == 8 + 1 + 5
    # The frozen guarantee: every test image still comes from the base dataset.
    test_images = [p.name for p in (tmp_path / "work" / "images" / "test").iterdir()]
    assert counts["test"] == 1 and all(n.startswith("b") for n in test_images)


def test_prepare_test_split_is_stable_with_or_without_extra(tmp_path: Path) -> None:
    """Adding app data must not move the held-out test set (fair eval gate)."""
    _make_dataset(tmp_path / "base", [f"b{i}" for i in range(10)])
    extra = _make_dataset(tmp_path / "app", ["e0", "e1"], with_yaml=False)
    cfg = _prep_config(tmp_path)
    prepare("t", config=cfg)
    without = sorted(p.name for p in (tmp_path / "work" / "images" / "test").iterdir())
    prepare("t", extra_src=extra, config=cfg)
    with_extra = sorted(p.name for p in (tmp_path / "work" / "images" / "test").iterdir())
    assert without == with_extra
