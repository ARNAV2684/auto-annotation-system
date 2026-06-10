"""Unit tests for :class:`~app.services.project_store.ProjectStore`.

Focus: the config-reconcile behaviour. The store seeds from config, persists to
JSON, and on every startup re-syncs built-in projects with config.yaml while
preserving runtime additions (user-registered classes and user-created groups).
GPU-free — the store only touches the filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.project_store import ProjectStore
from utils.config import Config


def _config(projects: list[dict], root: Path) -> Config:
    """A real Config wrapping just a ``projects`` block."""
    return Config({"projects": projects}, root)


_ROADS_V1 = {
    "id": "roads",
    "label": "Roads",
    "classes": ["pothole", "crack", "culvert"],
    "default_prompts": ["pothole"],
}
_ROADS_V2 = {
    "id": "roads",
    "label": "Roads",
    "classes": ["cracks", "potholes", "ravelling"],
    "default_prompts": ["potholes", "cracks"],
}


def test_seeds_from_config_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    store = ProjectStore(path=path, config=_config([_ROADS_V1], tmp_path))
    assert store.get_project("roads")["classes"] == ["pothole", "crack", "culvert"]
    assert path.exists()


def test_config_class_change_flows_into_existing_store(tmp_path: Path) -> None:
    """Editing config.yaml updates built-in projects even after first seed."""
    path = tmp_path / "projects.json"
    ProjectStore(path=path, config=_config([_ROADS_V1], tmp_path))
    store = ProjectStore(path=path, config=_config([_ROADS_V2], tmp_path))
    roads = store.get_project("roads")
    assert roads["classes"] == ["cracks", "potholes", "ravelling"]
    assert roads["default_prompts"] == ["potholes", "cracks"]


def test_user_added_class_survives_config_change(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    first = ProjectStore(path=path, config=_config([_ROADS_V1], tmp_path))
    first.add_class("roads", "manhole")
    store = ProjectStore(path=path, config=_config([_ROADS_V2], tmp_path))
    assert store.get_project("roads")["classes"] == [
        "cracks",
        "potholes",
        "ravelling",
        "manhole",
    ]


def test_user_created_group_survives_config_change(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    first = ProjectStore(path=path, config=_config([_ROADS_V1], tmp_path))
    first.create_project("Buildings", classes=["house"])
    store = ProjectStore(path=path, config=_config([_ROADS_V2], tmp_path))
    buildings = store.get_project("buildings")
    assert buildings is not None
    assert buildings["classes"] == ["house"]


def test_pre_seed_tracking_file_snaps_to_config(tmp_path: Path) -> None:
    """Files written before seed tracking existed migrate cleanly to config."""
    path = tmp_path / "projects.json"
    path.write_text(json.dumps({"projects": [_ROADS_V1]}), encoding="utf-8")
    store = ProjectStore(path=path, config=_config([_ROADS_V2], tmp_path))
    assert store.get_project("roads")["classes"] == ["cracks", "potholes", "ravelling"]


def test_new_config_project_appears_in_existing_store(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    ProjectStore(path=path, config=_config([_ROADS_V1], tmp_path))
    bridges = {"id": "bridges", "label": "Bridges", "classes": ["bridge"]}
    store = ProjectStore(path=path, config=_config([_ROADS_V2, bridges], tmp_path))
    assert store.get_project("bridges")["classes"] == ["bridge"]


def test_public_dicts_hide_seed_bookkeeping(tmp_path: Path) -> None:
    store = ProjectStore(
        path=tmp_path / "projects.json", config=_config([_ROADS_V1], tmp_path)
    )
    created = store.create_project("Buildings", classes=["house"])
    added = store.add_class("roads", "manhole")
    for project in [*store.list_projects(), store.get_project("roads"), created, added]:
        assert "seeded_classes" not in project
