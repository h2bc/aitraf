# aitraf-train

`aitraf-train` owns the current offline AITRAF workflows: data preparation,
data ops, label ops, training, evaluation, metrics, tracking, Hydra configs, and
operator scripts.

Owned code and assets:

- `configs/`: Hydra config tree for prepare, data ops, label ops, train, eval,
  and train+eval.
- `scripts/`: executable offline entrypoints used by the root `Taskfile.yml`.
- `aitraf_train.data_ops`: workflow orchestration for downloads, pose/bbox
  extraction, and VideoMAE feature cache creation.
- `aitraf_train.tasks`, `datasets`, `models`, `metrics`, and `tracking`: task
  implementations, train-only support code, reports, and MLflow/run behavior.

Shared runtime processing is imported from `aitraf_core`; it is not duplicated in
train modules. For example, train-side feature extraction uses
`aitraf_core.processing.video` and `aitraf_core.processing.models`.

Run commands from the repo root:

```bash
task train:prepare -- task=trick_classification
task train:data_ops -- pose_and_bbox_extraction.limit=1 video_mae_feature_extraction.limit=1
task train:train -- task=trick_classification model=video_mae max_samples=8
task train:eval -- task=score_prediction_ordinal model=video_mae model_id=<model-id>
task train:train_eval -- task=score_prediction_ordinal model=video_mae_temporal_fusion
```

`data/`, `storage/`, and `notebooks/` remain repo-level assets. They are not
moved under this package because they are workspace inputs/outputs rather than
Python package code.

Lightweight validation:

```bash
uv run python -m compileall -q packages/aitraf-core/src packages/aitraf-train/src packages/aitraf-api/src packages/aitraf-train/scripts
uv run python -c "import aitraf_core, aitraf_train, aitraf_api"
uv run python -c "import importlib.util; assert importlib.util.find_spec('aitraf') is None"
task --list
```
