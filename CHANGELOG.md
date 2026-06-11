# Changelog

All notable changes to the One-Click Auto Annotation System.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are git tags on `main`.

## [1.0.0] — 2026-06-11

The Dockerization release — the original seven-sprint brief is complete.

### Added
- Backend `Dockerfile` (CUDA-enabled PyTorch, SAM 2 pinned from source,
  GroundingDINO, YOLOv8) and two-stage frontend `Dockerfile`
  (Node build → nginx serving the SPA and proxying `/api`).
- `docker-compose.yml`: one-command launch with GPU passthrough, host-mounted
  volumes for weights / sessions / datasets, a persistent HuggingFace cache
  volume, backend healthcheck and restart policy.
- `scripts/fetch_models.py`: one-time SAM-checkpoint download, runnable on the
  host or inside the container.
- "Run with Docker" README section; this changelog.

## [0.9.0] — 2026-06-10

The custom-training release — the annotate → train → deploy loop is closed.

### Added
- `training/` package: seeded train/val/test data prep with a **frozen test
  split**, YOLOv8-seg fine-tuning (Blackwell-stable settings), evaluation.
- First trained road-defects model (cracks / potholes / ravelling):
  mask mAP50 0.603 on the held-out test split.
- Trained-model auto-detect: projects with deployed weights run their own
  fine-tuned model instead of GroundingDINO + SAM; UI shows a
  custom-model badge instead of the prompts box.
- One-click TRAIN: reviewed annotations join the dataset, a background job
  trains, evaluates and **deploys only if it beats the live model**;
  versioned model registry with metrics and one-click rollback.
- Before/after evaluation report (`docs/before_after_road_defects.md`):
  generic pipeline F1 0.02 → fine-tuned model F1 0.53, plus the
  reproducible comparison script.

### Changed
- Road Defects project classes aligned to the trained model
  (`cracks`, `potholes`, `ravelling`); the project store now reconciles
  built-in projects with `config.yaml` on startup while preserving
  user-registered classes and groups.
- `ultralytics` added to runtime requirements (CI stays GPU-free).

## [0.8.1] — 2026-06-10

### Fixed
- Smart Select collection now records history so Ctrl+Z undoes a
  just-created annotation.

### Added
- Zip upload (server-side extraction) as the reliable bulk-import path.

## [0.8.0] — 2026-06-09

### Added
- Smart Select multi-collect: hover previews SAM's mask live, each click
  collects an object, Enter commits them all as annotations.
- Manual polygon tool, undo/redo history, image tags and a per-image
  "done / reviewed" flag.

## [0.7.0] — 2026-06-09

### Added
- Roboflow-style workspace: top bar with image navigator and TRAIN stub,
  left panel tabs (Images / Classes / Layers / Tags / Attributes / History /
  Raw), class-colour system, label-assist panel, "repeat previous image".

## [0.6.1] — 2026-06-09

### Fixed
- Per-project export scoping: approved annotations whose class belongs to the
  selected project always export; dropped labels are reported, never silent.

## [0.6.0] — 2026-06-08

### Added
- Projects (Vegetation / Road Defects / Infrastructure) as UI presets with
  per-project classes and default detection prompts.
- Custom labeling: create new groups and register new classes at runtime,
  persisted in a writable project store.
- Batch approve / reject, keyboard shortcuts with a help overlay.

## [0.5.1] — 2026-05-30

### Fixed
- Frontend polish and CI formatting fixes after the first end-to-end release.

## [0.5.0] — 2026-05-30

### Added
- React + Vite + Tailwind review UI: upload, konva canvas with zoom/pan,
  click = point prompt, drag = box prompt, auto-detect panel, detection list
  with approve / reject / relabel, export page with format selection and
  zip download.

## [0.4.0] — 2026-05-29

### Added
- FastAPI backend implementing the contract in `docs/api_contract.md`:
  upload, point/box inference, auto-detect, annotation CRUD, export, stats.
- Filesystem session store (images, compressed masks, `annotations.json`).
- Model registry: load models once at startup, cache the embedded image.
- GPU-free API test suite via a stubbed model registry.

## [0.3.0] — 2026-05-29

### Added
- Annotation export to YOLO (segmentation polygons), COCO (polygon + RLE
  crowd) and Pascal VOC, with per-format validators, run directories,
  stats and zip packaging. Verified on all 24 seed drone images.

## [0.2.0] — 2026-05-28

### Added
- GroundingDINO text-prompted detection (HuggingFace transformers) and the
  Grounded-SAM chain: text → boxes → SAM 2 masks.
- Mask post-processing: IoU, NMS, area filtering, polygon simplification,
  box clamping.

## [0.1.0] — 2026-05-26

### Added
- Project scaffold, config-driven design (`config.yaml`), CUDA PyTorch on an
  RTX 5060 (Blackwell), SAM 2 inference core (point / box / automatic),
  CLAHE + sharpen preprocessing, CLI smoke test, unit tests, GitHub Actions
  CI (ruff + pytest, GPU-free).
