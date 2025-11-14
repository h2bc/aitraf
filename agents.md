# Agents Overview

This repo houses the small data-engineering pipeline that powers aggressive inline trick recognition. The flow is now Hydra-managed: we pull exports from Label Studio, validate them against a shared schema, and generate train/val/test manifests plus vocab metadata consumed by notebooks and future model training runs.

## Project Flow
1. `make data` (or `uv run python scripts/data_pipeline.py`) launches the Hydra entrypoint in `scripts/data_pipeline.py`. `configs/config.yaml` toggles `tasks.pull_ls` / `tasks.build_manifests` and wires the `label_studio`/`manifests` config groups that fill in concrete paths, split ratios, and `force` behavior.
2. When `tasks.pull_ls` is enabled, `aitraf.data.pull_ls.pull_label_studio` loads `.env` or shell `LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, and `LABEL_STUDIO_PROJECT_ID`, downloads the export with `label-studio-sdk`, enforces the schema from `aitraf.dataset_schema`, and writes `data/raw/labelstudio.parquet`.
3. When `tasks.build_manifests` is enabled, `aitraf.data.build_manifests.build_manifests` reads that parquet, drops incomplete rows, stratifies train/val/test splits (skating trick label as the stratification target), and emits JSONL manifests plus `labels.json` vocab metadata under `data/manifests/`.
4. Research notebooks in `notebooks/` and pose weights in `models/` rely on those manifests, so regenerating them keeps downstream experiments in sync.

## Key Components
- `scripts/data_pipeline.py` — Hydra entrypoint that orchestrates the individual tasks while keeping command-lines short.
- `configs/` — Hydra defaults and overrides: `config.yaml` defines repo-wide paths/toggles; `label_studio/*.yaml` handles export destinations and force flags; `manifests/*.yaml` captures split ratios, seeds, and output directories.
- `src/aitraf/dataset_schema.py` — single source of truth for expected columns (`video`, `trick`, `key_foot`, `person`) and categorical vocab lists, reducing drift between tasks.
- `src/aitraf/data/pull_ls.py` — Label Studio ingestion helper built on `label-studio-sdk` and `python-dotenv`; refuses to overwrite unless `force` is set.
- `src/aitraf/data/build_manifests.py` — manifest/vocab builder that validates ratios, catches undersized datasets, and writes JSONL + `labels.json`.
- `pyproject.toml` — managed via `uv`; declares the DL/tooling stack (torch, Ultralytics, transformers, Hydra, pandas/sklearn, datasets, FFmpeg helpers, boto3, etc.) plus dev extras (`ruff`, `pytest`, `ipykernel`, `ipywidgets`).

## Assets & Artifacts
- `data/raw/labelstudio.parquet` — canonical export written by the pull step.
- `data/manifests/{train,val,test}.jsonl` + `data/manifests/labels.json` — downstream-ready splits and vocab metadata.
- `data/hydra/*` — Hydra run directories/logs produced by `make data`.
- `models/yolo11n-pose.pt` — YOLO pose weights referenced in notebooks.
- `runs/pose/*` — historical Ultralytics pose experiments (predict outputs, etc.).
- `notebooks/*.ipynb` — exploratory notebooks (pose testing, clip EDA) that consume the manifests and models.

## Operational Notes
- Provide `LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, and `LABEL_STUDIO_PROJECT_ID` via environment or `.env` before running `make data`; the pull task errors early if any are missing.
- Config defaults currently set `force: true` for both the pull and manifest steps, so reruns overwrite artifacts; flip them to `false` when you want to preserve previous outputs.
- Hydra run folders land under `data/hydra/<timestamp>`; override `hydra.run.dir` when you need deterministic paths.
- Quality-of-life targets live in the `Makefile`: `make lint`/`make format` (Ruff) and `make jupyter` to launch notebooks inside the `uv`-managed environment.
- README still needs a fuller explainer; point teammates here for now when they ask how the data flow works.
