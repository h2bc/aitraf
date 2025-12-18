# AITRAF: Aggressive Inline Trick Recognition and Feedback

Model training and evaluation stack for inline skating trick recognition.

## Prerequisites

- Nvidia GPU with drivers for CUDA 12.6+
- Docker (used by the dev container)
- Access to project infra: S3 bucket `aitraf`, Label Studio, and MLflow (see links below)

## Environment Setup

1. Start the dev container (VS Code `Reopen in Container` or `devcontainer up`).
2. Copy `.env.example` to `.env` and fill in the Label Studio + AWS credentials.
3. Install task runner dependencies once with `uv sync`

## Tasks

Run commands via [Task](https://taskfile.dev); everything executes inside the uv-managed virtualenv.

| Command        | Description |
|----------------|-------------|
| Command | Description |
|---------|-------------|
| `task lint` | Runs Ruff checks across the repo. |
| `task format` | Applies Ruff formatting fixes. |
| `task data [overrides]` | Executes the Hydra-managed data pipeline (see below). |
| `task train -- model=… task=… [args]` | Runs the Hydra-managed training entrypoint (`scripts/train.py`). |
| `task eval -- model=… task=… model_id=… [args]` | Runs the unified evaluation entrypoint (`scripts/eval.py`). |
| `task train_eval -- model=… task=… [args]` | Runs the combined train+eval workflow (`scripts/train_eval.py`). |

### Data pipeline script

`task data` runs `scripts/data_pipeline.py`, which orchestrates the entire data prep flow via Hydra:

1. Download the latest Label Studio annotations as `data/labels.jsonl`.
2. Optionally sync referenced clips into `data/clips/`.
3. Optionally perform pose + bounding-box extraction with the configured Ultralytics weights.
4. Build train/val/test manifests plus a shared categorical vocabulary under `data/manifests/`.

Toggle each stage in `configs/data_ops.yaml` (e.g., disable clip downloads when cache is warm).

## Project Integrations

- **S3 storage**: https://storage.h2bcweb.com (bucket `aitraf`)
- **Label Studio**: https://label.h2bcweb.com/
- **MLflow**: https://mlops.h2bcweb.com/
