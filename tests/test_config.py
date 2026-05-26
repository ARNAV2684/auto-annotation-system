"""Tests for the project configuration loader (``utils.config``)."""

from utils.config import load_config


def test_load_config_returns_object() -> None:
    """The loader returns a usable Config object."""
    cfg = load_config()
    assert cfg is not None


def test_sam_variant_is_known() -> None:
    """The configured SAM variant is one we support."""
    cfg = load_config()
    assert cfg.get("sam.variant") in {"hiera_s", "hiera_b_plus"}


def test_dotted_key_missing_returns_default() -> None:
    """A missing dotted key falls back to the provided default."""
    cfg = load_config()
    assert cfg.get("does.not.exist", "fallback") == "fallback"


def test_path_resolution_is_absolute() -> None:
    """Configured paths resolve to absolute paths under the repo root."""
    cfg = load_config()
    assert cfg.path("paths.models_dir").is_absolute()


def test_classes_is_non_empty_list() -> None:
    """The active class list is present and non-empty."""
    cfg = load_config()
    classes = cfg.get("classes")
    assert isinstance(classes, list) and len(classes) >= 1
