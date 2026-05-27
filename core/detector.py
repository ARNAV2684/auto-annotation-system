"""GroundingDINO zero-shot object detector.

Given an image and a text prompt (e.g. ``"tree"``), GroundingDINO returns rough
bounding boxes for matching objects, with confidence scores and the matched phrase.
These boxes are then fed to SAM 2 as box-prompts to produce precise masks (the
"Grounded-SAM" pattern).

Uses the HuggingFace ``transformers`` implementation (no custom C++/CUDA build needed).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.config import load_config
from utils.device import resolve_device

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]


class GroundingDinoDetector:
    """Text-prompted object detector built on GroundingDINO."""

    def __init__(self, config=None, device: str | None = None, model_id: str | None = None) -> None:
        """Load the GroundingDINO processor and model.

        Args:
            config: Optional :class:`~utils.config.Config`; defaults to the project config.
            device: Optional device override (``"cuda"``/``"cpu"``); defaults to auto.
            model_id: Optional HuggingFace model id; defaults to the configured one.
        """
        self.cfg = config or load_config()
        self.device = resolve_device(device)
        self.model_id = model_id or self.cfg.get(
            "grounding_dino.model_id", "IDEA-Research/grounding-dino-base"
        )
        self.box_threshold = self.cfg.get("detector.box_threshold", 0.2)
        self.text_threshold = self.cfg.get("detector.text_threshold", 0.2)
        logger.info("Loading GroundingDINO (%s) on %s", self.model_id, self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def format_prompt(prompts: str | list[str]) -> str:
        """Format prompt(s) into GroundingDINO's lowercase, period-separated query string.

        Args:
            prompts: A single phrase or a list of phrases (e.g. ``["tree", "palm tree"]``).

        Returns:
            A query like ``"tree. palm tree."``.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        return ". ".join(p.strip().lower() for p in prompts) + "."

    def detect(
        self, image_rgb: np.ndarray, prompts: str | list[str] | None = None
    ) -> tuple[list[Box], list[float], list[str]]:
        """Detect objects matching the prompt in an image.

        Args:
            image_rgb: RGB ``uint8`` image (``H x W x 3``).
            prompts: Phrase(s) to detect; defaults to ``detector.prompts`` from config.

        Returns:
            ``(boxes, scores, labels)`` where ``boxes`` are ``(x1, y1, x2, y2)`` pixel
            ints, ``scores`` are floats, and ``labels`` are the matched phrases.
        """
        prompts = prompts if prompts is not None else self.cfg.get("detector.prompts", ["tree"])
        text = self.format_prompt(prompts)
        image = Image.fromarray(image_rgb)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        boxes = [tuple(int(v) for v in box) for box in results["boxes"].cpu().numpy()]
        scores = [float(s) for s in results["scores"].cpu().numpy()]
        labels = list(results.get("text_labels") or results.get("labels") or [""] * len(boxes))
        logger.info("GroundingDINO: %d boxes for prompt '%s'", len(boxes), text)
        return boxes, scores, labels
