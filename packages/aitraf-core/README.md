# aitraf-core

`aitraf-core` owns reusable runtime processing that is not specific to Hydra,
MLflow, or an offline training command.

Owned code:

- `aitraf_core.processing`: video frame loading/sampling, label transforms,
  collate helpers, and model-input preparation.
- `aitraf_core.processing.models`: reusable processors/wrappers for Pose TCN,
  VideoMAE, and VideoMAE temporal-fusion inputs.
- `aitraf_core.utils.huggingface` and `aitraf_core.utils.video_utils`: shared
  helper code used by both train-side workflows and future inference.

Not owned here:

- `data_ops`, `label_ops`, Hydra configs, scripts, MLflow tracking, metrics, and
  task dispatch. Those belong to `aitraf-train`.
- API routes, schemas, adapters, or service runtime. `aitraf-api` is reserved
  for that future work.

Runtime outputs such as sampled frames, pose outputs, and feature tensors may be
used by multiple surfaces, but registry/run ownership stays in `aitraf-train`.
Core code must not import from `aitraf_train` or `aitraf_api`.
