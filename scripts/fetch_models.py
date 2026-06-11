"""One-time model download for fresh checkouts and Docker volumes.

    python -m scripts.fetch_models

Downloads the SAM 2 checkpoint configured in ``config.yaml`` into
``models/sam_checkpoints/`` if it is not already there. The other models need
no help: GroundingDINO is fetched by HuggingFace ``transformers`` on first use
(into its own cache) and the YOLOv8 base checkpoint by ``ultralytics``.

Stdlib only — runs on the host or inside the backend container, no GPU needed.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from utils.config import load_config

# Official Meta release of the checkpoint named in config.yaml (sam.checkpoint).
SAM2_CHECKPOINT_URLS = {
    "sam2.1_hiera_small.pt": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    ),
    "sam2.1_hiera_base_plus.pt": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    ),
}


def _report(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        sys.stdout.write(f"\r  {pct:3d}%  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()


def fetch_sam_checkpoint() -> Path:
    """Download the configured SAM 2 checkpoint if missing. Returns its path."""
    cfg = load_config()
    checkpoint = cfg.path("sam.checkpoint")
    if checkpoint.exists():
        print(f"SAM checkpoint already present: {checkpoint}")
        return checkpoint
    url = SAM2_CHECKPOINT_URLS.get(checkpoint.name)
    if url is None:
        raise SystemExit(
            f"No known download URL for '{checkpoint.name}'. "
            "Download it manually into models/sam_checkpoints/."
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_suffix(".part")
    print(f"Downloading {checkpoint.name} ...")
    urllib.request.urlretrieve(url, partial, reporthook=_report)  # noqa: S310 — fixed https URL
    partial.rename(checkpoint)
    print(f"\nSaved: {checkpoint}")
    return checkpoint


def main() -> None:
    fetch_sam_checkpoint()
    print(
        "GroundingDINO and the YOLOv8 base model download themselves on first "
        "use — nothing else to fetch."
    )


if __name__ == "__main__":
    main()
