# aitraf-core

`aitraf-core` owns reusable runtime processing and model-loading helpers shared
by training and serving surfaces.

Owned code:

- `aitraf_core.pre_processing`: video-to-model-input helpers shared by data
  pipelines and serving, including VideoMAE temporal-fusion feature extraction
  utilities.
- `aitraf_core.processing`: video frame loading/sampling, label transforms,
  collate helpers, and model-input preparation.
- `aitraf_core.processing.models`: reusable processors/wrappers for Pose TCN,
  VideoMAE, and VideoMAE temporal-fusion inputs.
- `aitraf_core.inference`: shared prediction helpers for decoding logits and
  running VideoMAE inference.
- `aitraf_core.loading`: model loaders grouped by artifact source, including
  MLflow-trained artifacts and HuggingFace base components.
- `aitraf_core.utils.huggingface` and `aitraf_core.utils.jsonl`: shared helper
  code used by train-side workflows and the API.

Runtime outputs such as sampled frames, pose outputs, and feature tensors may be
used by training and serving surfaces.
