# Before / after: custom training on road defects

What did fine-tuning a model on our own labelled drone data actually buy us,
compared with the generic pipeline the app started with?

## Setup

- **Task:** instance segmentation of road defects (`cracks`, `potholes`,
  `ravelling`) in nadir drone imagery.
- **Test set:** the frozen held-out split of the labelled dataset — 19 images,
  22 instances, never seen during training. The split is produced by a seeded
  shuffle (`training/data_prep.py`), so every model — past and future — is
  scored on the identical sample.
- **Before:** the app's original auto-detect — **GroundingDINO** prompted with
  the project's default text prompts (`potholes`, `cracks`) feeding **SAM 2**.
- **After:** **YOLOv8n-seg** fine-tuned for 100 epochs on 152 training images
  (the deployed model, registry v1).
- **Metric:** per-class precision / recall / F1 with a prediction counted as
  correct at mask IoU ≥ 0.5. Reproduce with:

  ```bash
  python -m training.compare_baseline road_defects
  ```

## Results

### Before — GroundingDINO + SAM 2 (generic, text-prompted)

| class | precision | recall | F1 |
|---|---|---|---|
| cracks | 0.02 | 0.20 | 0.04 |
| potholes | 0.00 | 0.00 | 0.00 |
| ravelling | 0.00 | 0.00 | 0.00 |
| *(garbage labels)* | 0.00 | 0.00 | 0.00 |
| **all** | **0.01** | **0.09** | **0.02** |

### After — fine-tuned YOLOv8-seg (deployed)

| class | precision | recall | F1 |
|---|---|---|---|
| cracks | 0.40 | 0.80 | 0.53 |
| potholes | 0.50 | 0.43 | 0.46 |
| ravelling | 0.75 | 0.50 | 0.60 |
| **all** | **0.47** | **0.61** | **0.53** |

Standard COCO-style metrics for the deployed model on the same split
(`python -m training.evaluate road_defects`): **mask mAP50 0.603**, mask
mAP50-95 0.272, box mAP50 0.525.

## Reading the numbers

- **Overall F1 went from 0.02 to 0.53.** The generic pipeline is effectively
  unusable on this task; the fine-tuned model is a genuinely useful assistant
  that finds most defects (recall 0.61) with roughly half of its suggestions
  correct — exactly what a human-in-the-loop review queue wants.
- **The generic model cannot even *aim* at this task.** It has no text prompt
  that means "ravelling" (recall is structurally zero), it confuses potholes
  with everything round and dark, and GroundingDINO's phrase merging emits
  garbage labels (`##holes`, `potholes_cracks`) that all count as false
  positives in review.
- **Class-level signal mirrors the data.** Ravelling (114 training instances)
  scores best; potholes (43 instances) score worst. The lever is labelled
  data, and the app now produces it: annotate → mark done → TRAIN.
- **The eval gate keeps this honest.** Every retrain is scored on this same
  frozen split and deployed only if it beats the live model's mask mAP50, so
  the deployed model can only ever get better.

## Where the next gains come from

1. **More pothole labels** — the weakest class is the least-represented one.
2. **More imagery diversity** (lighting, surface types, altitudes) to lift
   mask tightness (mAP50-95).
3. Re-run TRAIN from the app after each labelling session; the registry keeps
   the full version history with metrics.
