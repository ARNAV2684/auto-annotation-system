"""Offline custom-training pipeline (Sprint 7).

YOLOv8-seg fine-tuning on labelled drone data. These scripts are run from the
command line on the GPU machine and are intentionally kept out of the runtime
and the GPU-free CI path (their deps live in ``requirements-train.txt``).
"""
