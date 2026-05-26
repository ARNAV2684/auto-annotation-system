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

## Project roadmap — Sprints 1–7

The project is delivered in seven sprints. Each sprint is a GitHub milestone; work
lands on `feature/*` branches via Pull Requests into `develop`, and each sprint
closes with a tagged `develop → main` release (`v0.1` … `v0.7`).

| Sprint | Theme | What gets built | Outcome |
|:------:|-------|-----------------|---------|
| **1** | **Foundation** | Repo scaffold, GPU environment (CUDA PyTorch), SAM 2 inference core (point / box / automatic modes), image preprocessing, CLI smoke test | SAM 2 produces masks on a real drone image, on the GPU |
| **2** | **Detection + Grounded-SAM** | GroundingDINO text-prompted detector, mask post-processing (filter, NMS, contours, polygon simplify), the "text → boxes → SAM masks" chain | Type a word ("tree") → clean masks appear automatically |
| **3** | **Annotation export** | YOLO / COCO / Pascal VOC converters, class-name mapping, export validation + packaging, stats summary, ≥5 unit tests per converter | Masks export to all three industry formats, validated |
| **4** | **Backend API** | FastAPI app, REST routes (upload, infer, auto-detect, annotations, export, stats), server-side model + embedding cache, batch processing, API tests | The whole engine is callable over HTTP (Swagger / curl) |
| **5** | **Frontend** | React + Vite review UI: image upload, canvas mask overlay, click = point prompt, drag = box prompt, auto-detect, approve/reject + class labels, export & download | End-to-end human-in-the-loop tool: upload → review → export |
| **6** | **Custom training** | Label drone data with the tool, fine-tune SAM 2 (and/or train a detector) on drone vegetation, before/after evaluation, feed the trained model back in | A fine-tuned model specialised for our drone imagery |
| **7** | **Dockerization & shipping** | Backend + frontend Dockerfiles (GPU-enabled), `docker-compose.yml`, health checks, `CHANGELOG.md`, "Run with Docker" docs | One command (`docker compose up`) launches the full system anywhere |

**Workflow rules (all sprints):** Conventional Commits (`feat:`, `fix:`, `docs:`,
`test:`, `refactor:`); PR per feature; `ruff` lint + Google-style docstrings + type
hints; never commit weights, `.env`, or `.venv`; target ≥70% test coverage.

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
