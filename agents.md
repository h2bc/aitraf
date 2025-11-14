# Agents Overview

This repo houses the small data-engineering pipeline that powers aggressive inline trick recognition. The flow is now Hydra-managed: we pull exports from Label Studio, validate them against a shared schema, and generate train/val/test manifests plus vocab metadata consumed by notebooks and future model training runs.

## Project Flow
1. `make data` (or `uv run python scripts/data_pipeline.py`) launches the Hydra entrypoint in `scripts/data_pipeline.py`. `configs/config.yaml` (general defaults) simply includes `configs/data/default.yaml`, so all pipeline toggles live under `cfg.data.tasks` and the per-stage knobs (`cfg.data.download_labels`, `cfg.data.create_manifests`) sit in that one data config file.
2. When `cfg.data.tasks.download_labels` is true, `aitraf.data.download_labels.download_labels` loads `.env` or shell `LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, and `LABEL_STUDIO_PROJECT_ID`, downloads the export with `label-studio-sdk`, enforces the schema from `aitraf.data.schema`, and writes `data/labels.jsonl`.
3. When `cfg.data.tasks.download_clips` is true, `aitraf.data.download_clips.download_clips` walks the labels file right after the download step, deduplicates S3 clip paths, and pulls each media file down to `data/clips/` via `boto3`; reruns skip files unless `force` is set.
4. When `cfg.data.tasks.create_manifests` is true, `aitraf.data.create_manifests.create_manifests` reads the JSONL export (and optionally the freshly downloaded clips), drops incomplete rows, stratifies train/val/test splits (skating trick label as the stratification target), and emits JSONL manifests plus `labels.json` vocab metadata under `data/manifests/`; each manifest row includes `video_id`, `s3_path`, `trick`, `key_foot`, and `person`.
5. Research notebooks in `notebooks/` and pose weights in `models/` rely on those manifests/clips, so regenerating/downloading them keeps downstream experiments in sync.

## Key Components
- `scripts/data_pipeline.py` — Hydra entrypoint that orchestrates the individual tasks while keeping command-lines short.
- `configs/config.yaml` + `configs/data/default.yaml` — two-file Hydra setup: the general config pulls in the `data` group (default), which owns every download/manifests knob and task toggle.
- `src/aitraf/data/schema.py` — single source of truth for expected columns (`video`, `trick`, `key_foot`, `person`) and categorical vocab lists, reducing drift between tasks.
- `src/aitraf/data/download_labels.py` — Label Studio ingestion helper built on `label-studio-sdk` and `python-dotenv`; refuses to overwrite unless `force` is set.
- `src/aitraf/data/create_manifests.py` — manifest/vocab builder that validates ratios, catches undersized datasets, and writes JSONL + `labels.json`.
- `src/aitraf/data/download_clips.py` — S3 clip downloader driven by the labels file; mirrors referenced media under `data/clips/`.
- `pyproject.toml` — managed via `uv`; declares the DL/tooling stack (torch, Ultralytics, transformers, Hydra, pandas/sklearn, datasets, FFmpeg helpers, boto3, etc.) plus dev extras (`ruff`, `pytest`, `ipykernel`, `ipywidgets`).

## Assets & Artifacts
- `data/labels.jsonl` — canonical export written by the download step.
- `data/manifests/{train,val,test}.jsonl` + `data/manifests/labels.json` — downstream-ready splits and vocab metadata.
- `data/clips/` — optional working set of downloaded S3 clips (ignored by git).
- `data/hydra/*` — Hydra run directories/logs produced by `make data`.
- `models/yolo11n-pose.pt` — YOLO pose weights referenced in notebooks.
- `runs/pose/*` — historical Ultralytics pose experiments (predict outputs, etc.).
- `notebooks/*.ipynb` — exploratory notebooks (pose testing, clip EDA) that consume the manifests and models.

## Operational Notes
- Provide `LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, and `LABEL_STUDIO_PROJECT_ID` via environment or `.env` before running `make data`; the pull task errors early if any are missing.
- `download_clips` relies on standard AWS credentials (env vars, default profile, etc.); keep the task disabled unless you actually need the local media since it pulls every referenced clip.
- Config defaults currently set `force: true` for both the pull and manifest steps, so reruns overwrite artifacts; flip them to `false` when you want to preserve previous outputs.
- Hydra run folders land under `data/hydra/<timestamp>`; override `hydra.run.dir` when you need deterministic paths.
- Quality-of-life targets live in the `Makefile`: `make lint`/`make format` (Ruff) and `make jupyter` to launch notebooks inside the `uv`-managed environment.
- README still needs a fuller explainer; point teammates here for now when they ask how the data flow works.
