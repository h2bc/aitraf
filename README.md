# AITRAF: Aggressive Inline Trick Recognition and Feedback

Model training and evaluation stack for inline skating trick recognition & performance feedback


## Prerequisites

- Nvidia GPU with drivers for CUDA 12.6+
- Docker (used by the dev container)
- Access to project infra: S3 bucket `aitraf`, Label Studio, and MLflow (see links below)

## Environment Setup

1. Start the dev container (VS Code `Reopen in Container` or `devcontainer up`).
2. Copy `.env.example` to `.env` and fill in the Label Studio + AWS credentials.
3. Install task runner dependencies once with `uv sync`


## Dev Commands

- `task lint` — run Ruff lint checks.
- `task format` — apply Ruff formatting fixes.

## Dataset

The dataset is customly filmed from a single fixed angle with multiple people. It currently covers 7 tricks, uses a consistent obstacle, and is captured under similar lighting conditions.

## Tasks

- **trick_classification**  
  Predicts the discrete trick label for each clip. Train/val/test splits are stratified by the target to preserve class balance. Available labels: `ao-soul`, `bs-royale`, `fs-royale`, `fs-savanah`, `mizou`, `soul`, `top-soul`.
- **score_prediction**  
  Predicts the execution score for each clip. Scores are collected as 1–4 ★★★★ and converted to a 0–1 percentage for training. We plan to extend this to pairwise ranking with separated goals (component scores vs a single overall score).


## Models

- **pose_tcn** (`configs/model/pose_tcn.yaml`)  
  Temporal convolutional network over pose keypoints sampled from pose sequences. Configurable depth/hidden size, Gaussian frame sampling, and Lightning-based training on GPU.
- **video_mae** (`configs/model/video_mae.yaml`)  
  Hugging Face VideoMAE backbone fine-tuned on sampled clips. Supports freezing the backbone, cacheable weights, and MLflow logging via the Hugging Face Trainer stack.



## Pipelines

Run commands via [Task](https://taskfile.dev)

| Command | Description |
|---------|-------------|
| `task data -- [overrides]` | Executes the Hydra-managed data pipeline (`scripts/data_pipeline.py`). Pass Hydra overrides after `--`. |
| `task train -- task=<task> model=<model> [overrides]` | Runs the Hydra-managed training entrypoint (`scripts/train.py`). |
| `task eval -- task=<task> model=<model> model_id=<model_id> [overrides]` | Runs the unified evaluation entrypoint (`scripts/eval.py`). |
| `task train_eval -- task=<task> model=<model> [overrides]` | Runs the combined train+eval workflow (`scripts/train_eval.py`). |

### Data ops script

`task data` runs `scripts/data_pipeline.py`, a Hydra-driven workflow composed of four stages (enable/disable each and set `force` flags in `configs/data_ops.yaml` or via cmd args):

#### Download Labels

- Pulls the latest Label Studio export (`data/labels.jsonl`) to refresh annotations used downstream.

#### Download Clips

- Resolves referenced video IDs and syncs the corresponding MP4 clips into `data/clips/`.

#### Pose + Bounding-Box Extraction

- Applies the Ultralytics pose + detection model to cached clips, writing keypoints to `data/poses/` and detection boxes to `data/boxes/`.
- Parameters cover device selection, image size, confidence thresholds, batch size, and optional clip limits.

#### Manifest Creation

- Builds train/val/test JSONL manifests under `data/manifests/<task>/`, stratifying by the target column when configured.
- Emits a shared `vocab.json` capturing label/id mappings per task for downstream consumers.


## Project Integrations

- **S3 storage**: https://storage.h2bcweb.com (bucket `aitraf`)
- **Label Studio**: https://label.h2bcweb.com/
- **MLflow**: https://mlops.h2bcweb.com/
