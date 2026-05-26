"""Tests for image preprocessing (``core.preprocessor``). GPU-free; runs in CI."""

import numpy as np

from core.preprocessor import apply_clahe, preprocess, resize_longest_edge, sharpen


def _dummy_image(height: int = 64, width: int = 128) -> np.ndarray:
    """A deterministic BGR uint8 test image."""
    rng = np.random.default_rng(0)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


def test_resize_scales_longest_edge_to_target() -> None:
    out = resize_longest_edge(_dummy_image(64, 128), 256)
    assert max(out.shape[:2]) == 256


def test_resize_preserves_aspect_ratio() -> None:
    out = resize_longest_edge(_dummy_image(64, 128), 256)
    assert abs(out.shape[0] / out.shape[1] - 64 / 128) < 0.05


def test_resize_is_noop_when_already_target() -> None:
    img = _dummy_image(100, 50)
    assert resize_longest_edge(img, 100).shape == img.shape


def test_clahe_preserves_shape_and_dtype() -> None:
    img = _dummy_image()
    out = apply_clahe(img, clip_limit=2.0, tile_size=8)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_sharpen_strength_zero_is_noop() -> None:
    img = _dummy_image()
    assert np.array_equal(sharpen(img, 0.0), img)


def test_preprocess_preserves_shape_and_dtype() -> None:
    img = _dummy_image(120, 200)
    out = preprocess(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_preprocess_enhance_false_returns_input_unchanged() -> None:
    img = _dummy_image()
    assert np.array_equal(preprocess(img, enhance=False), img)
