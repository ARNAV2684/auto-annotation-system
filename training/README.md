# Custom training (Sprint 7)

Offline YOLOv8-segmentation fine-tuning on labelled drone data. Runs on the GPU
machine, separate from the web app and the GPU-free CI.

## One-time setup

```bash
pip install -r requirements-train.txt
```

(PyTorch with the CUDA 12.8 build should already be installed — see the root
README. `ultralytics` reuses it.)

## Train a track

The only track with labelled data today is **road_defects** (the
`cracks / potholes / ravelling` dataset under `data/cracks_potholes_ravelling/`).

```bash
# 1. Split the labelled dataset into train/val/test + write data.yaml
python -m training.data_prep road_defects

# 2. Fine-tune YOLOv8-seg (uses the GPU if available)
python -m training.train_road road_defects

# 3. Evaluate on the held-out test split
python -m training.evaluate road_defects
```

Hyperparameters (base model, epochs, image size, batch, split, seed) live under
`training:` in `config.yaml`.

## Outputs

- Train-ready split: `data/road_defects_yolo/` (gitignored)
- Run + weights: `models/road_defects/train/weights/best.pt` (gitignored)

## Adding another track later

When labelled data for another domain (e.g. vegetation) arrives in YOLO-seg
format, add a block under `training:` in `config.yaml` mirroring `road_defects`
and run the same three commands with the new track name. No code changes needed.
```
