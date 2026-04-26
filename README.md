# AITRAF: Aggressive Inline Trick Recognition and Feedback

Model training and evaluation stack for inline skating trick recognition & performance feedback


## Prerequisites

- Nvidia GPU with drivers for CUDA 12.6+
- Docker (used by the dev container)
- Access to project infra: S3 bucket `aitraf` and MLflow (see links below)

## Environment Setup

1. Start the dev container (VS Code `Reopen in Container` or `devcontainer up`).
2. Copy `.env.example` to `.env` and fill in the AWS + MLflow credentials.
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
  Predicts the execution score for each clip. Scores are collected as 1–4 ★★★★ and converted to a 0–1 percentage for training.
- **score_prediction_binary**
  Predicts a binary quality label for each clip using only 1-star (`bad`) and 4-star (`good`) examples.
- **score_prediction_pairwise**
  Predicts the preferred clip from same-trick comparison pairs.


## Models

- **pose_tcn** (`configs/model/pose_tcn.yaml`)  
  Temporal convolutional network over pose keypoints sampled from pose sequences. Configurable depth/hidden size, Gaussian frame sampling, and Lightning-based training on GPU.
- **video_mae** (`configs/model/video_mae.yaml`)  
  Hugging Face VideoMAE backbone fine-tuned on sampled clips. Supports freezing the backbone, cacheable weights, and MLflow logging via the Hugging Face Trainer stack.



## Pipelines

Run commands via [Task](https://taskfile.dev)

| Command | Description |
|---------|-------------|
| `task label_ops -- [overrides]` | Executes the label ops pipeline (`scripts/label_ops_pipeline.py`). |
| `task data_ops -- [overrides]` | Executes the data ops pipeline (`scripts/data_ops_pipeline.py`). |
| `task prepare -- task=<task> [overrides]` | Executes the task preparation pipeline (`scripts/prepare.py`). |
| `task train -- task=<task> model=<model> [overrides]` | Runs the training pipeline (`scripts/train.py`). |
| `task eval -- task=<task> model=<model> model_id=<model_id> [overrides]` | Runs the evaluation pipeline (`scripts/eval.py`). |
| `task train_eval -- task=<task> model=<model> [overrides]` | Runs the combined train+eval pipeline (`scripts/train_eval.py`). |

### Label ops script

`task label_ops` runs `scripts/label_ops_pipeline.py`, a workflow composed of three stages (enable/disable each and set `force` flags in `configs/label_ops.yaml` or via cmd args):

#### Download Labels

- Downloads annotation files from a configurable S3 prefix and merges them into `data/labels.jsonl`.

#### Create pairwise comparison tasks

- Builds same-trick comparison pairs, writing one JSON file per pair with the `{"data": {"trick": ..., "left": ..., "right": ...}}` format.

#### Upload Pairs

- Uploads the generated pair files to S3 under a configurable prefix

### Data ops script

`task data_ops` runs `scripts/data_ops_pipeline.py`, a workflow composed of four shared-data stages (enable/disable each and set `force` flags in `configs/data_ops.yaml` or via cmd args):

#### Download Labels

- Downloads annotation files from a configurable S3 prefix and merges them into `data/labels.jsonl`.

#### Download Clips

- Resolves referenced video IDs and syncs the corresponding MP4 clips into `data/clips/`.

#### Pose + Bounding-Box Extraction

- Applies the Ultralytics pose + detection model to cached clips, writing keypoints to `data/poses/` and detection boxes to `data/boxes/`.
- Parameters cover device selection, image size, confidence thresholds, batch size, and optional clip limits.

#### Download Pairwise Labels

- Downloads annotation files from a configurable S3 prefix and merges them into one JSONL file

### Prepare script

`task prepare` runs `scripts/prepare.py`, which dispatches to the selected task's own preparation module.

- Builds train/val/test JSONL manifests under `data/manifests/<task>/`.
- Emits a task-local `vocab.json` under `data/manifests/<task>/` when the task defines one.
- Use `task prepare -- task=score_prediction` to prepare one task.


## Project Integrations

- **S3 storage**: https://storage.h2bcweb.com (bucket `aitraf`)
- **MLflow**: https://mlops.h2bcweb.com/
