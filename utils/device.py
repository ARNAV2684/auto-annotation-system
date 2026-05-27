"""Torch device resolution shared by the SAM 2 and GroundingDINO engines."""

from __future__ import annotations

import os

import torch

from utils.config import load_config


def resolve_device(requested: str | None = None) -> str:
    """Resolve the torch device to use.

    Order of precedence: explicit ``requested`` arg, ``SAM_DEVICE`` env var, the
    ``sam.device`` config value, then ``"auto"`` (CUDA if available, else CPU).

    Args:
        requested: Explicit device string, or ``None`` to fall back to config/env.

    Returns:
        Either ``"cuda"`` or ``"cpu"``.
    """
    choice = requested or os.getenv("SAM_DEVICE") or load_config().get("sam.device", "auto")
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice
