"""Configuration loading for the Auto Annotation System.

Loads ``config.yaml`` (path overridable via the ``CONFIG_PATH`` env var) and any
``.env`` file, exposing a single cached :class:`Config` object. No module should
read paths or thresholds directly; they should all come from here.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repo root = parent of this file's parent (utils/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Typed, dict-backed view over ``config.yaml`` with dotted-key access."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self._data = data
        self.root = root

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested value via a dotted key, e.g. ``"sam.variant"``.

        Args:
            dotted_key: Keys joined by dots, walked from the config root.
            default: Value returned if any key in the path is missing.

        Returns:
            The resolved value, or ``default`` if the path does not exist.
        """
        node: Any = self._data
        for key in dotted_key.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, dotted_key: str, default: str | None = None) -> Path:
        """Resolve a config value as a filesystem path relative to the repo root.

        Args:
            dotted_key: Dotted key whose value is a path string.
            default: Fallback path string if the key is missing.

        Returns:
            An absolute :class:`~pathlib.Path`. Absolute config values are kept as-is.
        """
        raw = self.get(dotted_key, default)
        if raw is None:
            raise KeyError(f"No path configured for '{dotted_key}'")
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


@functools.lru_cache(maxsize=1)
def load_config() -> Config:
    """Load and cache the project configuration.

    Reads ``.env`` (if present) then the YAML file pointed to by ``CONFIG_PATH``
    (default ``config.yaml`` at the repo root).

    Returns:
        A cached :class:`Config` instance shared across the process.
    """
    load_dotenv(REPO_ROOT / ".env")
    config_path = Path(os.getenv("CONFIG_PATH", REPO_ROOT / "config.yaml"))
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data, REPO_ROOT)
