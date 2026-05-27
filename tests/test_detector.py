"""Tests for the GroundingDINO detector (``core.detector``).

Needs ``torch`` + ``transformers`` (and a GPU for the model), so the module is skipped
where those are unavailable — e.g. on the GPU-free CI runner.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from core.detector import GroundingDinoDetector  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GroundingDINO model load needs a CUDA GPU"
)


def test_format_prompt_joins_and_lowercases() -> None:
    assert GroundingDinoDetector.format_prompt(["Tree", "Palm Tree"]) == "tree. palm tree."


def test_format_prompt_single_string() -> None:
    assert GroundingDinoDetector.format_prompt("Pothole") == "pothole."


@requires_cuda
def test_detect_returns_aligned_lists() -> None:
    detector = GroundingDinoDetector()
    image = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)
    boxes, scores, labels = detector.detect(image, "tree")
    assert len(boxes) == len(scores) == len(labels)
