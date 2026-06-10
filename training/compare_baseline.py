"""Before/after comparison: generic Grounded-SAM vs the fine-tuned model.

Runs both detectors over the frozen test split and scores them per class with
precision / recall / F1 at mask IoU >= 0.5 — the plain-language answer to
"what did custom training actually buy us?".

    python -m training.compare_baseline road_defects

The baseline is the exact pipeline the app used before training: GroundingDINO
prompted with the project's default prompts feeding SAM 2. Note its structural
handicap, which is the whole point: a generic detector has no prompt that means
"ravelling", so that class is unreachable for it.

Heavy imports stay inside :func:`run`, so this module is importable (and its
matching logic testable) on GPU-free CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from core.annotation_converter import normalise_label
from core.mask_processor import polygon_to_mask
from core.types import Detection
from utils.config import load_config

IOU_THRESHOLD = 0.5


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks (0.0 when both are empty)."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def load_ground_truth(
    label_file: Path, class_names: list[str], width: int, height: int
) -> list[Detection]:
    """Parse one YOLO-seg label file into ground-truth Detection rows."""
    truths: list[Detection] = []
    for line in label_file.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class id + at least 3 points
            continue
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        polygon = [
            (coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords) - 1, 2)
        ]
        mask = polygon_to_mask(polygon, width, height)
        truths.append(
            Detection(mask=mask, box=(0, 0, 0, 0), label=class_names[class_id], score=1.0)
        )
    return truths


def match_instances(
    truths: list[Detection], predictions: list[Detection]
) -> dict[str, dict[str, int]]:
    """Greedy per-class matching at mask IoU >= 0.5 -> TP / FP / FN tallies.

    Predictions are taken best-score-first; each ground-truth instance can be
    matched at most once. Returns ``{class_name: {"tp": n, "fp": n, "fn": n}}``
    covering every class present in either list.
    """
    tallies: dict[str, dict[str, int]] = {}
    for det in truths + predictions:
        tallies.setdefault(det.label, {"tp": 0, "fp": 0, "fn": 0})
    matched: set[int] = set()
    for pred in sorted(predictions, key=lambda d: -d.score):
        best_iou, best_idx = 0.0, -1
        for i, truth in enumerate(truths):
            if i in matched or truth.label != pred.label:
                continue
            iou = mask_iou(pred.mask, truth.mask)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= IOU_THRESHOLD:
            matched.add(best_idx)
            tallies[pred.label]["tp"] += 1
        else:
            tallies[pred.label]["fp"] += 1
    for i, truth in enumerate(truths):
        if i not in matched:
            tallies[truth.label]["fn"] += 1
    return tallies


def merge_tallies(
    total: dict[str, dict[str, int]], image: dict[str, dict[str, int]]
) -> None:
    for label, counts in image.items():
        slot = total.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
        for key, value in counts.items():
            slot[key] += value


def summarise(tallies: dict[str, dict[str, int]]) -> list[tuple[str, float, float, float]]:
    """Per-class ``(label, precision, recall, f1)`` rows, then an 'all' row."""
    rows: list[tuple[str, float, float, float]] = []
    grand = {"tp": 0, "fp": 0, "fn": 0}
    for label in sorted(tallies):
        counts = tallies[label]
        rows.append((label, *_prf(counts)))
        for key in grand:
            grand[key] += counts[key]
    rows.append(("all", *_prf(grand)))
    return rows


def _prf(counts: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run(track: str) -> None:
    """Score both pipelines over the test split and print markdown tables."""
    import cv2

    from core.detect import grounded_sam
    from core.detector import GroundingDinoDetector
    from core.preprocessor import preprocess
    from core.sam_inference import SAM2Segmenter
    from core.trained_detector import build_for_project

    cfg = load_config()
    conf = (cfg.get("training") or {})[track]
    work = cfg.root / conf["work_dir"]
    class_names = [
        str(n)
        for n in yaml.safe_load((work / "data.yaml").read_text(encoding="utf-8"))["names"]
    ]
    project = next(p for p in cfg.get("projects") or [] if p["id"] == track)
    prompts = list(project.get("default_prompts") or [])

    print(f"Loading models (baseline prompts: {prompts}) ...")
    detector = GroundingDinoDetector()
    segmenter = SAM2Segmenter()
    trained = build_for_project(track)

    baseline_tallies: dict[str, dict[str, int]] = {}
    trained_tallies: dict[str, dict[str, int]] = {}
    images = sorted((work / "images" / "test").iterdir())
    for i, image_path in enumerate(images, 1):
        bgr = cv2.imread(str(image_path))
        height, width = bgr.shape[:2]
        truths = load_ground_truth(
            work / "labels" / "test" / f"{image_path.stem}.txt", class_names, width, height
        )
        # Baseline: the app's pre-training path (CLAHE-enhanced image, text prompts).
        rgb = cv2.cvtColor(preprocess(bgr), cv2.COLOR_BGR2RGB)
        baseline = grounded_sam(rgb, detector, segmenter, prompts, cfg)
        for det in baseline:
            det.label = normalise_label(det.label)
        merge_tallies(baseline_tallies, match_instances(truths, baseline))
        # Trained: the deployed fine-tuned model (raw image, fixed classes).
        merge_tallies(trained_tallies, match_instances(truths, trained.detect(bgr)))
        print(f"  [{i}/{len(images)}] {image_path.name}")

    for title, tallies in (
        ("Baseline: GroundingDINO + SAM 2", baseline_tallies),
        ("Fine-tuned YOLOv8-seg (deployed)", trained_tallies),
    ):
        print(f"\n### {title}\n")
        print("| class | precision | recall | F1 |")
        print("|---|---|---|---|")
        for label, precision, recall, f1 in summarise(tallies):
            print(f"| {label} | {precision:.2f} | {recall:.2f} | {f1:.2f} |")


def main() -> None:
    run(sys.argv[1] if len(sys.argv) > 1 else "road_defects")


if __name__ == "__main__":
    main()
