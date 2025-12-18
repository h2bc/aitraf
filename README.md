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

## Pipelines

Run commands via [Task](https://taskfile.dev)

| Command | Description |
|---------|-------------|
| `task lint` | Runs Ruff checks across the repo. |
| `task format` | Applies Ruff formatting fixes. |
| `task data -- …` | Executes the Hydra-managed data pipeline (`scripts/data_pipeline.py`). Pass Hydra overrides after `--`. |
| `task train -- task=… model=… …` | Runs the Hydra-managed training entrypoint (`scripts/train.py`). |
| `task eval -- task=… model=… model_id=… …` | Runs the unified evaluation entrypoint (`scripts/eval.py`). |
| `task train_eval -- task=… model=… …` | Runs the combined train+eval workflow (`scripts/train_eval.py`). |

### Data pipeline script

`task data` runs `scripts/data_pipeline.py`, a Hydra-driven workflow composed of four stages (toggle them in `configs/data_ops.yaml`):

#### Download Labels

- Pulls the latest Label Studio export (`data/labels.jsonl`), ensuring we have a fresh canonical annotation snapshot.
- Respects the `force` flag to skip work when cached annotations are acceptable.

#### Download Clips

- Reads the manifest of referenced video IDs and syncs the corresponding MP4 clips into `data/clips/`.
- Supports incremental downloads and a `force` switch to re-pull corrupted/missing files.

#### Pose + Bounding-Box Extraction

- Runs the Ultralytics pose + detection model on the cached clips, writing keypoints to `data/poses/` and detection boxes to `data/boxes/`.
- Accepts parameters for device selection, image size, confidence thresholds, batch size, and an optional `limit` for quick smoke tests.

#### Manifest Creation

- Assembles train/val/test JSONL manifests under `data/manifests/<task>/`, stratifying by the target column when configured.
- Emits a shared `vocab.json` capturing label/id mappings per task, used later by all training/eval pipelines.

## Tasks

- **trick_classification**  
  Classification task predicting the discrete trick label. Uses manifests under `data/manifests/trick_classification/` with the target column `trick`. Training/validation/test splits are stratified by the target to preserve class balance.

## Models

- **pose_tcn** (`configs/model/pose_tcn.yaml`)  
  Temporal convolutional network over pose keypoints sampled from pose sequences. Configurable depth/hidden size, Gaussian frame sampling, and Lightning-based training on GPU.
- **video_mae** (`configs/model/video_mae.yaml`)  
  Hugging Face VideoMAE backbone fine-tuned on sampled clips. Supports freezing the backbone, cacheable weights, and MLflow logging via the Hugging Face Trainer stack.

## Pre-processing

### Pose TCN

- **Pose sampling**: we subsample a fixed number of frames per clip via `sample_frame_indices`, supporting `gaussian_stochastic` (biased toward the center) and `uniform` distributions. Missing detections are skipped before sampling.
- **Tensor layout**: sampled keypoints and detection confidences are stacked into `(F, J, 3)` tensors (frames × joints × coords) and flattened to feed the TCN, while raw pose, score, and frame tensors are retained for analysis.

### VideoMAE

- **Clip decoding**: manifests reference MP4 clips stored under `data/clips`. Each row specifies a clip ID that is decoded on-the-fly with `torchcodec`.
- **Frame sampling**: the same `sample_frame_indices` helper selects `sample_frames` frames per clip according to the configured distribution.
- **Rotation correction**: if a clip carries an EXIF/video rotation flag, frames are rotated back to upright orientation with Kornia before feeding the VideoMAE processor.

## Project Integrations

- **S3 storage**: https://storage.h2bcweb.com (bucket `aitraf`)
- **Label Studio**: https://label.h2bcweb.com/
- **MLflow**: https://mlops.h2bcweb.com/
