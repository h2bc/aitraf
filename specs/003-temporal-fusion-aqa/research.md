# Research: Temporal Fusion Trick AQA

## Decision: Use `aitraf_core.pre_processing` for video-to-model-input work

**Decision**: Introduce `aitraf_core.pre_processing` as the shared layer for
video decoding, frame sampling, optional sampled-frame cache access, VideoMAE
processor application, VideoMAE feature extraction, and VideoMAE feature cache
reads/writes.

**Rationale**: The current train data op
`aitraf_train.data_ops.video_mae_feature_extraction` owns most VideoMAE feature
extraction orchestration, while API inference must do equivalent work for
temporal-fusion predictions. Moving this shared behavior into core prevents
drift between data pipelines and serving. The module is named `pre_processing`
instead of `pre-processing` because Python imports cannot use hyphenated package
names.

**Alternatives considered**:

- Keep feature extraction in `aitraf-train` and call it from the API. Rejected
  because API serving should not import offline pipeline package code.
- Duplicate extraction in `aitraf-api`. Rejected because it violates the reuse
  requirement and risks train/serve preprocessing skew.
- Keep everything under `aitraf_core.processing`. Rejected because the user
  explicitly wants to distinguish video-to-input preparation from model-input
  processing.

## Decision: Keep `aitraf_core.processing` for model-input processing

**Decision**: Retain `aitraf_core.processing` as the shared home for operations
that transform already-prepared model inputs into train/eval/predict batches:
label transforms, class weights, collate functions, temporal-fusion feature
sample shaping, and validation of model-input tensors.

**Rationale**: Current temporal-fusion training and evaluation already import
`process_temporal_fusion_feature_sample`, `build_collate`, and label helpers from
`aitraf-core`. Keeping these in `processing` preserves the existing mental model
while removing video decoding and feature extraction from this layer.

**Alternatives considered**:

- Move all temporal-fusion code into `pre_processing`. Rejected because training
  and evaluation also need model-input transforms that are not video decoding or
  feature extraction.
- Move label and collate helpers into `aitraf-train`. Rejected because shared
  processing helpers are already used by multiple model/task surfaces.

## Decision: Put temporal-fusion predictions under `aitraf_core.inference`

**Decision**: Add temporal-fusion prediction helpers under
`aitraf_core.inference.models.video_mae_temporal_fusion`, including logits,
decoded label/value output, and any confidence derivation needed by API
responses.

**Rationale**: Existing `aitraf_core.inference.models.video_mae` owns VideoMAE
prediction and decoding for non-fusion models. Temporal fusion should follow that
shape so `aitraf-api` can call core prediction helpers and avoid task-local model
logic.

**Alternatives considered**:

- Decode temporal-fusion output in `aitraf-api`. Rejected because decoding logic
  belongs with reusable model inference behavior and should also be available to
  eval/predict smoke paths.
- Decode output inside `aitraf-train` evaluation only. Rejected because serving
  needs the same contract.

## Decision: Cache models in API runtime and cache frames/features by explicit contracts

**Decision**: Keep loaded model instance caching behind the API/core model loader
for repeated requests. Add file-backed sampled-frame and VideoMAE feature cache
entries with explicit contract metadata. Reuse cached frames/features only when
video identity, sampling contract, backbone/processor identity, model or artifact
version, and preprocessing contract match.

**Rationale**: Model loading is expensive and already cached in
`aitraf_api.prediction.load_registered_model`. Frame and feature cache entries
need stronger contract checks because stale cached tensors can silently produce
wrong predictions if sampling, processor, or model settings change.

**Alternatives considered**:

- Use path existence alone as the feature cache contract. Rejected because the
  current feature extraction path encodes some settings but not enough metadata
  for serving safety.
- Always recompute frames and features in API inference. Rejected because
  repeated demo-video predictions would be unnecessarily expensive and the user
  explicitly requested caching.
- Cache final predictions. Rejected because the request asks to cache model,
  frame sampling, and VideoMAE features, not prediction results; prediction
  caching could hide model changes.

## Decision: Keep orchestration in `aitraf-train` but use core helpers from data ops and task modules

**Decision**: `aitraf_train.data_ops.video_mae_feature_extraction` becomes a
Hydra/logging/DataLoader orchestration wrapper over `aitraf_core.pre_processing`.
Temporal-fusion train/eval modules continue to live under
`aitraf_train.tasks.score_prediction_ordinal.video_mae_temporal_fusion`, but
their sample processing and prediction-compatible transforms come from
`aitraf-core`.

**Rationale**: This respects the existing train package structure:
`scripts/data_ops_pipeline.py` dispatches data ops, `scripts/train.py` dispatches
training targets, `scripts/eval.py` dispatches evaluation targets, and each
task/model pair owns its own `training.py` and `evaluation.py`. The refactor
should not move Hydra command ownership into core.

**Alternatives considered**:

- Move Hydra data ops into `aitraf-core`. Rejected because core should be reusable
  runtime processing, not offline command orchestration.
- Replace the task/model package structure. Rejected because it is already used
  consistently across score prediction, classification, PoseTCN, VideoMAE, and
  temporal fusion.

## Decision: API service shape is explicit `pre_processing`, `processing`, `predict`

**Decision**: Trick AQA serving should orchestrate three named stages:

1. `pre_processing(cache_frames=?, cache_video_features=?)`: resolve the video,
   sample or load frames, and produce or load VideoMAE features.
2. `processing()`: validate/shape temporal-fusion model inputs.
3. `predict()`: run the cached temporal-fusion model and decode the result.

**Rationale**: This matches the user's requested mental model while keeping the
API route/service thin. It also gives tests stable seams for verifying cache
reuse and explicit failures without loading real model weights.

**Alternatives considered**:

- One monolithic `predict_manifest_row` function for all models. Rejected because
  temporal fusion needs cache-aware pre-processing and should not overload the
  non-fusion VideoMAE path.
- Separate API-only temporal-fusion service internals. Rejected because it would
  duplicate core behavior needed by data and eval pipelines.
