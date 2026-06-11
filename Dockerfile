# ──────────────────────────────────────────────────────────────────────────
# Backend image: FastAPI + SAM 2 + GroundingDINO + YOLOv8 on the GPU.
#
# No CUDA base image needed: PyTorch's cu128 wheels bundle their own CUDA
# runtime; the container only needs the host's NVIDIA driver, which Docker
# injects at run time (`--gpus all` / compose device reservations).
#
# Model weights are NOT baked in — they live in volumes mounted at run time
# (see docker-compose.yml), so images stay lean and trained model versions
# survive rebuilds. Layers are ordered slowest-changing first so code edits
# rebuild in seconds, not minutes.
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# git: SAM 2 installs from source. libgl1 + libglib2.0-0: OpenCV runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8 build (Blackwell / sm_120 support) — the biggest layer.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SAM 2 from source, pinned for reproducible builds. The CUDA extension is
# skipped (matches the dev environment; it only affects optional
# post-processing).
ENV SAM2_BUILD_CUDA=0
RUN pip install --no-cache-dir \
        "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

# Application code (changes most often -> last layers).
COPY config.yaml ./
COPY app/ app/
COPY core/ core/
COPY utils/ utils/
COPY training/ training/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
