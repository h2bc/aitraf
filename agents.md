# Agents Overview

This repo currently focuses on data preparation for aggressive inline trick recognition: exporting labeled clips from Label Studio, validating canonical schema, and emitting train/val/test manifests plus vocab metadata for downstream modeling notebooks.

## Project Flow
1. `src/aitraf/scripts/pull_ls.py` downloads the latest Label Studio export using `LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, and `LABEL_STUDIO_PROJECT_ID`, drops rows missing the video path, reports label sparsity, and writes `data/labeled.parquet`.
2. `src/aitraf/scripts/build_manifests.py` reads `data/labeled.parquet`, checks the schema defined in `src/aitraf/dataset_schema.py`, performs stratified splits (train/val/test), writes JSONL manifests under `data/manifests/`, and emits `labels.json` containing label vocabularies and id mappings.
3. Downstream notebooks under `notebooks/` (e.g., pose experiments, EDA) consume the manifests and the YOLO pose weights in `models/`.

## Key Components
- `src/aitraf/dataset_schema.py` — central definitions for columns (`video`, `trick`, `key_foot`, `person`) and categorical features used across scripts.
- `src/aitraf/paths.py` — canonical directories (`DATA_DIR`, `NOTEBOOKS_DIR`, `MODELS_DIR`) imported by scripts/configs to avoid hard-coded paths.
- `src/aitraf/scripts/` — automation entry points for data ingestion (`pull_ls.py`) and manifest generation (`build_manifests.py`); run via `python -m aitraf.scripts.<name>`.
- `pyproject.toml` — declares the research stack (PyTorch, Ultralytics, AV/FFmpeg helpers, Label Studio SDK, datasets tooling, boto3) plus dev tools (`ruff`, `pytest`).

## Assets & Artifacts
- `data/` — includes `labeled.parquet`, generated manifests (`train/val/test.jsonl`, `labels.json`), and sample media (`vileika_example_1.MOV`).
- `models/` — currently holds `yolo11n-pose.pt`, likely used within the notebooks.
- `runs/pose/` — experiment outputs (predict, predict2, …) from prior pose runs.
- `notebooks/` — exploratory and modeling notebooks (`1-testing_open_pose.ipynb`, `2-simple_clips_eda.ipynb`).

## Operational Notes
- Both scripts refuse to overwrite outputs unless `--force` is passed, preventing accidental data loss.
- The README documents environment setup with `uv`, Ruff usage, and Git LFS requirements but still needs a high-level project description.
