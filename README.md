# One-Click Auto Annotation System

AI-assisted image annotation for **drone imagery**. A human clicks (or runs an
auto-detector); **SAM 2** suggests precise object masks; a human approves; approved
annotations export to **YOLO / COCO / Pascal VOC** — and feed a **custom model**
that is trained, evaluated, and deployed from inside the app.

Built as Project 1 of the AI & Computer Vision internship at GarudaUAV.

> **Status:** custom training shipped — the loop below is closed end-to-end.
> Remaining: Dockerization.

## What it does

```
upload → annotate (auto-detect | click | box | smart select | polygon)
       → human approve/label → export YOLO/COCO/VOC
       → TRAIN → evaluate on a frozen test split → deploy only if better
       → better auto-detect → annotate faster → repeat
```

- **SAM 2** turns clicks/boxes into pixel-precise masks (class-agnostic).
- **GroundingDINO** auto-suggests boxes from a text prompt ("tree"), which feed
  SAM 2 — the "Grounded-SAM" pattern. Used by projects without a trained model.
- **Fine-tuned YOLOv8-seg** powers auto-detect for the Road Defects project
  (cracks / potholes / ravelling) — one model call yields boxes, labels and
  masks, no text prompts needed. See [`training/README.md`](training/README.md).
- **Human-in-the-loop** review gate; approved masks become training data.
- **TRAIN button**: reviewed work joins the dataset, a background job
  fine-tunes and scores the candidate, and it is **deployed only if it beats
  the live model** on the held-out test split. Every run is recorded as an
  immutable version with one-click rollback.

**Did custom training pay off?** On the held-out road-defects test split the
generic Grounded-SAM pipeline scores **F1 0.02**; the fine-tuned model scores
**F1 0.53** (mask mAP50 0.603). Full numbers + methodology:
[`docs/before_after_road_defects.md`](docs/before_after_road_defects.md).

## Architecture

| Layer | Tech | Location |
|-------|------|----------|
| AI engine | SAM 2, GroundingDINO, YOLOv8-seg, OpenCV, PyTorch | `core/` |
| Custom training | Ultralytics fine-tuning, eval gate, model registry | `training/` |
| Backend API | FastAPI (REST) | `app/` |
| Frontend | React + Vite | `frontend/` |
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

## Run with Docker (one command)

The whole system — backend with GPU models + frontend — runs from a single
compose file. Requirements: **Docker Desktop** (WSL2 backend on Windows) and a
recent **NVIDIA driver**. Verify GPU passthrough once with
`docker run --rm --gpus all ubuntu nvidia-smi` — it should print your GPU.

```powershell
# 1. First time only: download the SAM checkpoint into the mounted models/
#    folder (runs inside the image — no local Python needed)
docker compose build backend
docker compose run --rm backend python -m scripts.fetch_models

# 2. Launch everything
docker compose up --build -d

# 3. Watch until the backend reports healthy (first boot also downloads
#    GroundingDINO, ~700 MB, into a persistent cache volume)
docker compose ps

# 4. Open the app
#    http://localhost:8080        — the annotation UI
#    http://localhost:8000/docs   — backend Swagger

# Stop (data survives — models/, outputs/ and data/ are host folders)
docker compose down
```

Sessions, annotations, model weights, trained versions and datasets all live
in mounted host folders, so containers are disposable. The TRAIN button works
inside Docker too — the training job runs in the backend container on the
same GPU. No GPU? Delete the `deploy:` block in `docker-compose.yml` and
everything runs on CPU (slow, but fine for a demo).

## Repository layout

```
app/      FastAPI backend (routes, session/project stores, training service)
core/     AI engine: preprocessor, sam_inference, detector, trained_detector,
          mask_processor, annotation_converter
training/ offline fine-tuning: data prep, train, evaluate, run_job (TRAIN
          button backend), model registry, before/after comparison
utils/    config loader, visualizer, file/IO helpers, validators
models/   checkpoints + trained model versions (gitignored)
data/     sample images, training datasets, app-annotation snapshots (gitignored)
tests/    pytest suite (GPU-free in CI)
docs/     API contract, before/after evaluation report
frontend/ React + Vite SPA
```

## Running the backend (Sprint 4+)

The FastAPI server loads the SAM 2 + GroundingDINO models once at startup and
exposes the routes documented in [`docs/api_contract.md`](docs/api_contract.md).

```powershell
# from the repo root, with the venv activated
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- <http://localhost:8000/docs> — interactive **Swagger UI**: every route with a
  click-to-try button. Use this to upload an image, run auto-detect, and download
  a YOLO/COCO/VOC export without writing any JavaScript.
- <http://localhost:8000/api/health> — health probe.

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Bind port |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS-allowed origin for the React UI |
| `LOAD_MODELS_ON_STARTUP` | `true` | Set to `false` to skip model load (useful for fast iteration on non-model routes) |

Uploaded images, masks, and annotations live under
`outputs/sessions/default/` (gitignored). When the project is later run inside
Docker (Sprint 7), this folder will be mounted to a host directory so all data
persists across container restarts.

## Running the frontend (Sprint 5+)

The React + Vite + Tailwind UI lives in `frontend/`. It talks to the backend
via Vite's dev-server proxy — every `/api/*` request is forwarded to the
backend on port 8000, so the frontend code never hardcodes the host.

```powershell
# one-time, from frontend/
npm install

# dev server with hot reload
npm run dev
# opens http://localhost:5173

# production build (TypeScript check + Vite bundle)
npm run build
```

You need **both servers running** for the app to work:

| Terminal | Command | Port |
|---|---|---|
| 1 (backend) | `uvicorn app.main:app --reload --port 8000` | 8000 |
| 2 (frontend) | `cd frontend && npm run dev` | 5173 |

Open <http://localhost:5173> in your browser.

## Development

```powershell
ruff check .          # backend lint (PEP 8, line length 100)
pytest                # backend tests
pytest --cov          # coverage (target ≥ 70%)

cd frontend
npm run lint          # frontend lint (ESLint)
npm run format        # auto-format with Prettier
npm run build         # production build (also catches TypeScript errors)
```

Custom training (offline scripts behind the TRAIN button — see
[`training/README.md`](training/README.md)):

```powershell
pip install -r requirements-train.txt        # ultralytics (not needed in CI)
python -m training.data_prep road_defects    # build the train/val/test split
python -m training.train_road road_defects   # fine-tune YOLOv8-seg
python -m training.evaluate road_defects     # score on the frozen test split
python -m training.run_job road_defects      # the full TRAIN-button pipeline
python -m training.compare_baseline road_defects  # before/after report data
```

Branching: `main` (protected) ← `develop` ← `feature/*`; merges via PR.
Commits follow **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
