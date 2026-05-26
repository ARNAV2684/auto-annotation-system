# One-Click Auto Annotation System

AI-assisted image annotation for **drone imagery**. A human clicks (or runs an
auto-detector); **SAM 2** suggests precise object masks; a human approves; approved
annotations export to **YOLO / COCO / Pascal VOC** for downstream model training.

Built as Project 1 of the AI & Computer Vision internship at GarudaUAV.

> **Status:** Sprint 1 — foundation (repo, environment, SAM 2 inference core).

## What it does

```
upload → SAM 2 embeds image once → prompt (GroundingDINO text | click | box)
       → masks → clean → human approve/label → export YOLO/COCO/VOC → (re)train
```

- **SAM 2** turns clicks/boxes into pixel-precise masks (class-agnostic).
- **GroundingDINO** (Sprint 2) auto-suggests boxes from a text prompt ("tree"),
  which feed SAM 2 — the "Grounded-SAM" pattern.
- **Human-in-the-loop** review gate; approved masks become training data.

## Architecture

| Layer | Tech | Location |
|-------|------|----------|
| AI engine | SAM 2, GroundingDINO, OpenCV, PyTorch | `core/` |
| Backend API | FastAPI (REST) | `app/` (Sprint 4) |
| Frontend | React + Vite | `frontend/` (Sprint 5) |
| Config | `config.yaml` (no hardcoding) | repo root |

## Environment setup

Requires **Python 3.10+** and (for GPU) an NVIDIA GPU with **CUDA 12.8+**
(this project targets an RTX 5060 / Blackwell laptop GPU, 8 GB).

```powershell
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install CUDA build of PyTorch (NOT the default PyPI/CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Install SAM 2 from source (see docs/architecture.md) and download a checkpoint
#    into models/sam_checkpoints/ (gitignored).

# 5. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Copy `.env.example` → `.env` and adjust as needed. All tunables live in `config.yaml`.

## Repository layout

```
app/      FastAPI backend (Sprint 4)
core/     AI engine: preprocessor, sam_inference, detector, mask_processor, annotation_converter
utils/    config loader, visualizer, file/IO helpers, validators
models/   SAM 2 / GroundingDINO checkpoints (gitignored)
data/     sample images & exported annotations
tests/    pytest suite
docs/     architecture, API reference, user guide
frontend/ React + Vite SPA (Sprint 5)
```

## Development

```powershell
ruff check .          # lint (line length 100, PEP 8)
pytest                # run tests
pytest --cov          # coverage (target ≥ 70%)
```

Branching: `main` (protected) ← `develop` ← `feature/*`; merges via PR.
Commits follow **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
