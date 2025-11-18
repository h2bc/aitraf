# Agents Overview

This repository contains two parallel tracks for aggressive inline trick recognition:

1. A Hydra-managed data engineering pipeline that mirrors Label Studio exports locally, validates schemas, builds manifests/vocab metadata, and optionally downloads every referenced clip from S3.
2. Early VideoMAE experimentation code that consumes those manifests/clips to stand up PyTorch + Transformers training loops.

Everything runs through `uv` and simple `make` targets so agents can reproduce results quickly.

## Pipelines & Entrypoints
- `make data` → `scripts/data_pipeline.py` drives the Label Studio → manifests flow. The Hydra config is defined by `configs/data.yaml`, which in turn includes `configs/base.yaml` (paths/hydra dirs) and `configs/data/default.yaml` (per-stage options).
- `make train-video-mae` → `scripts/video_mae/train.py` launches a Hydra-configured VideoMAE run (`configs/video_mae.yaml`). The current implementation focuses on loading batches, running them through the processor/decoder stack, and logging clip metadata—it is scaffolding for real training.
- `make lint` / `make format` run Ruff, and `make jupyter` exposes notebooks inside the uv-managed virtual environment.

## Data Pipeline Details
1. **Download labels** (`cfg.data.tasks.download_labels`): `aitraf.data.download_labels.download_labels` loads Label Studio credentials from `.env` or the shell (`LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, `LABEL_STUDIO_PROJECT_ID`), downloads the export via `label-studio-sdk`, validates required columns from `aitraf.data.schema`, and writes `data/labels.jsonl`. Existing files are skipped unless `force` is true.
2. **Download clips** (`cfg.data.tasks.download_clips`): `aitraf.data.download_clips.download_clips` scans the JSONL’s `video` column for `s3://` URIs, deduplicates them, and streams the media into `data/clips/` using `boto3`. Requires `AWS_ENDPOINT_URL` + `AWS_DEFAULT_REGION` (and whichever AWS creds the SDK picks up). Existing files are skipped unless `force` is set.
3. **Create manifests** (`cfg.data.tasks.create_manifests`): `aitraf.data.create_manifests.create_manifests` drops incomplete rows, enforces minimum dataset sizes, performs stratified `train/val/test` splits, and writes JSONL manifests plus `labels.json` vocab metadata under `data/manifests/`. Each row includes `video_id`, `s3_path`, `trick`, `key_foot`, and `person`.

Hydra run outputs land under `data/hydra/<timestamp>`, and overrides can be applied the usual Hydra way (e.g., `uv run python scripts/data_pipeline.py data.tasks.download_clips=false`).

## VideoMAE Experimentation
- **Entrypoint**: `scripts/video_mae/train.py` reads `configs/video_mae/default.yaml` (default backbone `MCG-NJU/videomae-base`, manifest path `data/manifests/train.jsonl`, GPU device, etc.) and prepares a `VideoMAETrainingConfig`.
- **Processing helpers**: `src/aitraf/video_mae/processing.py` exposes `load_clip`, which resolves a manifest row to a local clip, decodes frames with `torchcodec`, samples evenly spaced frames, runs them through a `VideoMAEImageProcessor`, and attaches label metadata.
- **Training stub**: `src/aitraf/video_mae/training.py` wires the processor/model from Hugging Face Transformers, creates a PyTorch `DataLoader`, logs batches, and is the starting point for full fine-tuning. It respects `batch_size`, `num_workers`, `num_frames`, `device`, `output_dir`, and currently limits iterations via `max_batches`.

These modules depend on local clips existing under `data/clips/` (matching the downloaded filenames) and the manifest produced by the data pipeline.

## Configuration & Modules
- `configs/base.yaml` centralizes project/manifests directories and Hydra run dirs. Both `configs/data.yaml` and `configs/video_mae.yaml` use it so relative paths stay consistent.
- `src/aitraf/data/schema.py` defines canonical column names and vocab columns shared across scripts, preventing drift between download/manifests/training.
- `src/aitraf/logging.py` wraps Loguru with helper functions (`setup_logging`, `heading`, `spacer`) so CLI entrypoints get consistent colored output while muting noisy dependencies (`httpx`, `boto3`, etc.).
- `pyproject.toml` (consumed by `uv`) pins the Python stack: torch/torchvision/torchcodec from the CUDA 12.6 index, Transformers/Hydra/pandas/scikit-learn, AWS + Label Studio clients, and dev extras (`ruff`, `pytest`, `ipykernel`, `ipywidgets`).

## Assets & Directories
- `data/labels.jsonl` — canonical export from Label Studio; `data/clips/` mirrors referenced S3 files; `data/manifests/{train,val,test}.jsonl` + `labels.json` feed downstream work; `data/hydra/*` stores Hydra run metadata.
- `models/` ships reference weights (e.g., `yolo11n-pose.pt`) used by notebooks.
- `runs/*` hosts experiment outputs (`runs/video_mae` for VideoMAE scaffolding, `runs/pose/*` for historical Ultralytics tests).
- `notebooks/` contains exploratory notebooks that expect manifests/clips to exist locally.

## Operational Notes
- Always provide `LABEL_STUDIO_URL`, `LABEL_STUDIO_TOKEN`, and `LABEL_STUDIO_PROJECT_ID` before running the data pipeline; the job fails early if any are missing.
- Clip downloads require valid AWS credentials plus `AWS_ENDPOINT_URL`/`AWS_DEFAULT_REGION`. Disable `data.tasks.download_clips` if you only need manifests—the step pulls every referenced clip.
- Defaults currently set `force: true` for labels/manifests and `force: false` for clips. Adjust per run when you need to preserve previous artifacts.
- Use `uv run <command>` (see README) to ensure dependencies resolve via the project lockfile; the Makefile already wraps `uv run` for common targets.
