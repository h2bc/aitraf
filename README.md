# AITRAF: Aggressive Inline Trick Recognition and Feedback

Model training and evaluation stack for inline skating trick recognition & performance feedback


## Prerequisites

- Nvidia GPU with drivers for CUDA 12.8+
- Docker (used by the dev container)
- Access to project infra: S3 bucket `aitraf` and MLflow (see links below)

## Environment Setup

1. Start the dev container (VS Code `Reopen in Container` or `devcontainer up`).
2. Copy `.env.example` to `.env` and fill in the AWS + MLflow credentials.
3. Install task runner dependencies once with `uv sync`


## Dev Commands

- `task lint` — run Ruff lint checks.
- `task format` — apply Ruff formatting fixes.

## Tasks

- **trick_classification**  
  Predicts the discrete trick label for each clip. Train/val/test splits are stratified by the target to preserve class balance. Available labels: `ao-soul`, `bs-royale`, `fs-royale`, `fs-savanah`, `mizou`, `soul`, `top-soul`.
- **score_prediction_ordinal**  
  Predicts the execution score for each clip as an ordinal 1–3 star rating.
- **score_prediction_pairwise**
  Predicts the preferred clip from same-trick comparison pairs.


## Models

- **pose_tcn** (`configs/model/pose_tcn.yaml`)  
  Temporal convolutional network over pose keypoints sampled from pose sequences. Configurable depth/hidden size, Gaussian frame sampling, and Lightning-based training on GPU.
- **video_mae** (`configs/model/video_mae.yaml`)  
  Hugging Face VideoMAE backbone fine-tuned on sampled clips. Supports freezing the backbone, cacheable weights, and MLflow logging via the Hugging Face Trainer stack.
- **video_mae_temporal_fusion** (`configs/model/video_mae_temporal_fusion.yaml`)  
  Temporal fusion model over cached VideoMAE features from multiple clips per sample. Configurable clip count, sampling strategy, fusion layers/heads/queries, dropout, and training hyperparameters.


## Data And Storage

`data/` contains lightweight repo-local experiment inputs and outputs:

- `data/labels.jsonl` and `data/pairwise_labels.jsonl` are merged annotation files.
- `data/manifests/<task>/` contains train/val/test manifests and task vocab files created by `task prepare`.

`storage/` contains larger generated assets, caches, and run outputs. By default it is created at `./storage`, but it can be moved by setting `AITRAF_STORAGE_PATH`.

- `storage/data/clips/`, `storage/data/poses/`, and `storage/data/boxes/` contain downloaded clips and extracted pose/detection outputs.
- `storage/data/video_mae_features/` contains cached VideoMAE features used by temporal fusion models.
- `storage/models/` contains downloaded model weights and caches.
- `storage/runs/` contains training/evaluation outputs and Hydra run directories.


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

`task label_ops` runs `scripts/label_ops_pipeline.py`, a workflow composed of three stages:

#### Download Labels

- Downloads annotation files from a configurable S3 prefix and merges them into `data/labels.jsonl`.

#### Create pairwise comparison tasks

- Builds same-trick comparison pairs, writing one JSON file per pair with the `{"data": {"trick": ..., "left": ..., "right": ...}}` format.

#### Upload Pairs

- Uploads the generated pair files to S3 under a configurable prefix

### Data ops script

`task data_ops` runs `scripts/data_ops_pipeline.py`, a workflow composed of shared-data stages:

#### Download Labels

- Downloads annotation files from a configurable S3 prefix and merges them into `data/labels.jsonl`.

#### Download Clips

- Resolves referenced video IDs and syncs the corresponding MP4 clips into `data/clips/`.

#### Pose + Bounding-Box Extraction

- Applies the Ultralytics pose + detection model to cached clips, writing keypoints to `data/poses/` and detection boxes to `data/boxes/`.
- Parameters cover device selection, image size, confidence thresholds, batch size, and optional clip limits.

#### Download Pairwise Labels

- Downloads annotation files from a configurable S3 prefix and merges them into one JSONL file

#### VideoMAE Feature Extraction

- Extracts and caches VideoMAE features from downloaded clips for temporal fusion experiments.

### Prepare script

`task prepare` runs `scripts/prepare.py`, a one-step pipeline that dispatches manifest creation to the selected task's own preparation module.

- Builds train/val/test JSONL manifests under `data/manifests/<task>/`.
- Emits a task-local `vocab.json` under `data/manifests/<task>/` when the task defines one.
- Use `task prepare -- task=score_prediction_ordinal` to prepare one task.

### Train script

`task train` runs `scripts/train.py`, dispatching to the training implementation for the selected task/model pair.

- Reads prepared manifests from `data/manifests/<task>/`.
- Writes run outputs under `storage/runs/`.
- Logs training runs and model artifacts to MLflow.

### Eval script

`task eval` runs `scripts/eval.py`, evaluating an existing MLflow model for the selected task/model pair.

- Requires `model_id=<model-name-or-version>` so the script can load `models:/<model_id>`.
- Reads the selected task's prepared manifests.
- Writes evaluation outputs under `storage/runs/`.

### Train + Eval script

`task train_eval` runs `scripts/train_eval.py`, training and then evaluating the model produced by that training run.

- Uses the same task/model dispatch as `task train` and `task eval`.
- Passes the trained model URI directly into evaluation.
- Useful for single experiment runs and Hydra multiruns.


## Experiment Configuration

Experiment defaults live in `configs/` and are composed with Hydra when a pipeline runs. Top-level files such as `train.yaml`, `eval.yaml`, `train_eval.yaml`, `prepare.yaml`, `data_ops.yaml`, and `label_ops.yaml` define pipeline defaults. Task configs live in `configs/task/`, model configs live in `configs/model/`, and shared paths live in `configs/base.yaml`.

```text
configs/
|-- base.yaml
|-- train.yaml
|-- eval.yaml
|-- train_eval.yaml
|-- prepare.yaml
|-- data_ops.yaml
|-- label_ops.yaml
|-- task/
|   |-- trick_classification.yaml
|   |-- score_prediction_ordinal.yaml
|   `-- score_prediction_pairwise.yaml
`-- model/
    |-- pose_tcn.yaml
    |-- video_mae.yaml
    `-- video_mae_temporal_fusion.yaml
```

You can change experiment settings in two ways:

- Edit the YAML config when changing the default behavior for everyone.
- Pass command-line overrides after `--` when running one experiment.

```bash
task train_eval -- task=score_prediction_ordinal model=video_mae_temporal_fusion
task train -- task=trick_classification model=pose_tcn model.learning_rate=5e-4
task prepare -- task=score_prediction_ordinal create_manifests.force=false
```

Use Hydra multirun with `-m` to run several experiment variants from one command. Comma-separated values define the sweep:

```bash
task train_eval -- -m task=score_prediction_ordinal model=video_mae,pose_tcn
task train_eval -- -m task=trick_classification,score_prediction_ordinal model=video_mae_temporal_fusion
```


## MLOps

Training and evaluation runs are tracked in MLflow for experiment comparison and reproducibility. Runs log the key config parameters, metrics, datasets, and trained model artifacts needed to compare local, remote, and multirun experiments.


## DevOps

The repository includes a production Docker image for running experiments outside the local dev container. GitHub Actions publishes the image to GHCR, giving remote or hosted training machines the same runtime used locally. Pull the image, provide credentials/storage, and run the same `task ... -- [overrides]` commands.


## Project Integrations

- **S3 storage**: https://storage.h2bcweb.com (bucket `aitraf`)
- **MLflow**: https://mlops.h2bcweb.com/
