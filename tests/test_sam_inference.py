"""Tests for the SAM 2 engine (``core.sam_inference``).

These need ``torch`` + ``sam2`` (and a GPU for the model itself), so the whole module
is skipped where those are unavailable — e.g. on the GPU-free CI runner.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sam2")

import torch  # noqa: E402  (imported after importorskip on purpose)

from core.sam_inference import SAM2Segmenter, resolve_device  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SAM 2 model load needs a CUDA GPU"
)


def test_resolve_device_honours_explicit_choice() -> None:
    assert resolve_device("cpu") == "cpu"


@requires_cuda
def test_segmenter_loads_and_runs_all_modes() -> None:
    """The model loads and all three prompt modes return well-formed outputs."""
    segmenter = SAM2Segmenter()
    image = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)

    # Automatic
    records = segmenter.generate(image)
    assert isinstance(records, list)

    # Point
    segmenter.set_image(image)
    masks, scores = segmenter.predict_point((128, 128))
    assert masks.shape[0] == scores.shape[0] and masks.dtype == bool

    # Box
    mask, score = segmenter.predict_box((40, 40, 200, 200))
    assert mask.shape == (256, 256) and mask.dtype == bool and 0.0 <= score <= 1.0


@requires_cuda
def test_point_prediction_requires_set_image() -> None:
    segmenter = SAM2Segmenter()
    with pytest.raises(RuntimeError):
        segmenter.predict_point((10, 10))
