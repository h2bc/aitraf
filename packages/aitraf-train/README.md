# aitraf-train

`aitraf-train` owns the offline AITRAF workflows: data preparation, data ops,
label ops, training, evaluation, metrics, tracking, Hydra configs, and operator
scripts.

Shared runtime processing is imported from `aitraf_core`; it is not duplicated in
train modules. For example, train-side feature extraction uses
`aitraf_core.processing.video` and `aitraf_core.processing.models`.

## Tasks

- **trick_classification**: predicts the discrete trick label for each clip.
  Train/val/test splits are stratified by the target to preserve class balance.
  Available labels: `ao-soul`, `bs-royale`, `fs-royale`, `fs-savanah`, `mizou`,
  `soul`, `top-soul`.
- **score_prediction_ordinal**: predicts the execution score for each clip as an
  ordinal 1-3 star rating.
- **score_prediction_pairwise**: predicts the preferred clip from same-trick
  comparison pairs.

## Models

- **pose_tcn** (`configs/model/pose_tcn.yaml`): temporal convolutional network
  over pose keypoints sampled from pose sequences. Configurable depth/hidden
  size, Gaussian frame sampling, and Lightning-based training on GPU.
- **video_mae** (`configs/model/video_mae.yaml`): Hugging Face VideoMAE backbone
  fine-tuned on sampled clips. Supports freezing the backbone, cacheable weights,
  and MLflow logging via the Hugging Face Trainer stack.
- **video_mae_temporal_fusion** (`configs/model/video_mae_temporal_fusion.yaml`):
  temporal fusion model over cached VideoMAE features from multiple clips per
  sample. Configurable clip count, sampling strategy, fusion
  layers/heads/queries, dropout, and training hyperparameters.

## Data And Storage

`data/` contains lightweight repo-local experiment inputs and outputs:

- `data/labels.jsonl` and `data/pairwise_labels.jsonl` are merged annotation
  files.
- `data/manifests/<task>/` contains train/val/test manifests and task vocab files
  created by `task train:prepare`.

`storage/` contains larger generated assets, caches, and run outputs. By default
it is created at `./storage`, but it can be moved by setting
`AITRAF_STORAGE_PATH`.

- `storage/data/clips/`, `storage/data/poses/`, and `storage/data/boxes/`
  contain downloaded clips and extracted pose/detection outputs.
- `storage/data/video_mae_features/` contains cached VideoMAE features used by
  temporal fusion models.
- `storage/models/` contains downloaded model weights and caches.
- `storage/runs/` contains training/evaluation outputs and Hydra run
  directories.

## Commands

Run train commands from the repo root:

| Command | Description |
|---------|-------------|
| `task train:label_ops -- [overrides]` | Executes the label ops pipeline. |
| `task train:data_ops -- [overrides]` | Executes the data ops pipeline. |
| `task train:prepare -- task=<task> [overrides]` | Executes task preparation. |
| `task train:train -- task=<task> model=<model> [overrides]` | Runs training. |
| `task train:eval -- task=<task> model=<model> model_id=<model_id> [overrides]` | Runs evaluation. |
| `task train:train_eval -- task=<task> model=<model> [overrides]` | Runs training followed by evaluation. |

When working inside `packages/aitraf-train`, the same commands can be run without
the `train:` namespace:

```bash
task prepare -- task=trick_classification
task data_ops -- pose_and_bbox_extraction.limit=1 video_mae_feature_extraction.limit=1
task train -- task=trick_classification model=video_mae max_samples=8
task eval -- task=score_prediction_ordinal model=video_mae model_id=<model-id>
task train_eval -- task=score_prediction_ordinal model=video_mae_temporal_fusion
```

## Pipelines

### Label Ops

`label_ops` is composed of:

- Download labels from a configurable S3 prefix and merge them into
  `data/labels.jsonl`.
- Create same-trick pairwise comparison tasks.
- Upload generated pair files to S3 under a configurable prefix.

### Data Ops

`data_ops` is composed of:

- Download labels from a configurable S3 prefix and merge them into
  `data/labels.jsonl`.
- Download referenced clips into `storage/data/clips/`.
- Extract pose keypoints and detection boxes into `storage/data/poses/` and
  `storage/data/boxes/`.
- Download pairwise labels into one JSONL file.
- Extract and cache VideoMAE features into `storage/data/video_mae_features/`.

### Prepare

`prepare` dispatches manifest creation to the selected task preparation module.
It builds train/val/test JSONL manifests under `data/manifests/<task>/` and
emits task-local `vocab.json` files when needed.

### Train

`train` dispatches to the training implementation for the selected task/model
pair. It reads prepared manifests, writes run outputs under `storage/runs/`, and
logs training runs and model artifacts to MLflow.

### Eval

`eval` evaluates an existing MLflow model for the selected task/model pair. It
requires `model_id=<model-name-or-version>` so the script can load
`models:/<model_id>`.

### Train + Eval

`train_eval` trains a model and evaluates the model produced by that training
run. It uses the same task/model dispatch as `train` and `eval`.

## Experiment Configuration

Experiment defaults live in `configs/` and are composed with Hydra when a
pipeline runs. Top-level files such as `train.yaml`, `eval.yaml`,
`train_eval.yaml`, `prepare.yaml`, `data_ops.yaml`, and `label_ops.yaml` define
pipeline defaults. Task configs live in `configs/task/`, model configs live in
`configs/model/`, and shared paths live in `configs/base.yaml`.

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

Change experiment settings by editing YAML defaults or by passing command-line
overrides after `--`:

```bash
task train:train_eval -- task=score_prediction_ordinal model=video_mae_temporal_fusion
task train:train -- task=trick_classification model=pose_tcn model.learning_rate=5e-4
task train:prepare -- task=score_prediction_ordinal create_manifests.force=false
```

Use Hydra multirun with `-m` to run several experiment variants from one command:

```bash
task train:train_eval -- -m task=score_prediction_ordinal model=video_mae,pose_tcn
task train:train_eval -- -m task=trick_classification,score_prediction_ordinal model=video_mae_temporal_fusion
```

## Validation

```bash
uv run python -m compileall -q packages/aitraf-core/src packages/aitraf-train/src packages/aitraf-api/src packages/aitraf-train/scripts
uv run python -c "import aitraf_core, aitraf_train, aitraf_api"
task --list
```

## Integrations

- **S3 storage**: https://storage.h2bcweb.com (bucket `aitraf`)
- **MLflow**: https://mlops.h2bcweb.com/
