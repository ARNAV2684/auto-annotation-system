"""Mutable project store.

Projects (Vegetation, Road Defects, Infrastructure, …) start life as static
presets in ``config.yaml``. Once the user can create new groups and register
custom classes at runtime, that list becomes **application state**, so it needs
a writable home that survives a restart.

This store:

* seeds itself from ``config.yaml``'s ``projects:`` block (or the legacy flat
  ``classes`` key) the first time it runs,
* then persists every change to a gitignored JSON file (``outputs/projects.json``
  by default), which becomes the single source of truth from then on.

It is the one place that answers "what projects/classes exist right now", and is
consumed by ``GET /api/projects``, the create/register routes, **and** the export
route (so a freshly-registered class is exportable immediately).

Like :class:`~app.services.session_store.SessionStore`, this module is framework
agnostic — it knows nothing about FastAPI.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from core.annotation_converter import normalise_label
from utils.config import load_config

logger = logging.getLogger(__name__)

_MAX_CLASS_LEN = 64


class ProjectStore:
    """Thread-safe, filesystem-backed list of projects (UI presets)."""

    def __init__(self, path: Path | None = None, config=None) -> None:
        """Create the store, seeding from config if the JSON file is absent.

        Args:
            path: JSON file to persist to. Defaults to ``paths.projects_store``.
            config: Optional :class:`~utils.config.Config`; defaults to the project config.
        """
        cfg = config or load_config()
        self.path = (
            Path(path) if path else cfg.path("paths.projects_store", "outputs/projects.json")
        )
        self._lock = threading.Lock()
        self._projects: list[dict] = []
        self._load_or_seed(cfg)

    # ─── public API ──────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        """Return a copy of every project dict."""
        with self._lock:
            return [dict(p) for p in self._projects]

    def get_project(self, project_id: str) -> dict | None:
        with self._lock:
            match = next((p for p in self._projects if p["id"] == project_id), None)
            return dict(match) if match else None

    def create_project(
        self,
        label: str,
        classes: list[str] | None = None,
        default_prompts: list[str] | None = None,
    ) -> dict:
        """Create a new group (project). Returns the created project dict.

        Raises:
            ValueError: empty label, or a group with the same id/label exists.
        """
        label = (label or "").strip()
        if not label:
            raise ValueError("Group name cannot be empty.")
        slug = normalise_label(label)
        if not slug:
            raise ValueError("Group name cannot be empty.")
        with self._lock:
            for project in self._projects:
                if project["id"] == slug or project["label"].strip().lower() == label.lower():
                    raise ValueError(f"A group named '{label}' already exists.")
            project = {
                "id": slug,
                "label": label,
                "classes": _dedup_normalised(classes or []),
                "default_prompts": list(default_prompts or []),
            }
            self._projects.append(project)
            self._persist()
            logger.info("Created project %s (%s)", slug, label)
            return dict(project)

    def add_class(self, project_id: str, class_name: str) -> dict:
        """Register a class (label) under an existing group. Idempotent.

        Returns the updated project dict.

        Raises:
            KeyError: no such project.
            ValueError: empty or over-long class name.
        """
        name = normalise_label(class_name)
        if not name:
            raise ValueError("Class name cannot be empty.")
        if len(name) > _MAX_CLASS_LEN:
            raise ValueError(f"Class name too long (max {_MAX_CLASS_LEN} characters).")
        with self._lock:
            project = next((p for p in self._projects if p["id"] == project_id), None)
            if project is None:
                raise KeyError(project_id)
            if name not in project["classes"]:
                project["classes"].append(name)
                self._persist()
                logger.info("Registered class '%s' under project %s", name, project_id)
            return dict(project)

    # ─── internal: load / seed / persist ─────────────────────────────────

    def _load_or_seed(self, cfg) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                projects = raw.get("projects")
                if projects:
                    self._projects = projects
                    return
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt %s — reseeding from config", self.path)
        self._projects = self._seed_from_config(cfg)
        self._persist()

    @staticmethod
    def _seed_from_config(cfg) -> list[dict]:
        raw = cfg.get("projects") or []
        if raw:
            return [
                {
                    "id": entry["id"],
                    "label": entry["label"],
                    "classes": _dedup_normalised(entry.get("classes", [])),
                    "default_prompts": list(entry.get("default_prompts", [])),
                }
                for entry in raw
            ]
        legacy = cfg.get("classes", []) or []
        return [
            {
                "id": "default",
                "label": "Default",
                "classes": _dedup_normalised(legacy),
                "default_prompts": list(legacy),
            }
        ]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"projects": self._projects}, indent=2), encoding="utf-8"
        )


def _dedup_normalised(classes: list[str]) -> list[str]:
    """Normalise every class name and drop duplicates, preserving order."""
    out: list[str] = []
    for name in classes:
        norm = normalise_label(name)
        if norm and norm not in out:
            out.append(norm)
    return out
