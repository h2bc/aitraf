# Core Pipeline Contract: Temporal Fusion Trick AQA

This contract defines the shared core behavior used by `aitraf-train` data,
train, and eval pipelines and by `aitraf-api` trick AQA serving.

Python package paths use `pre_processing` rather than `pre-processing` because
hyphenated names are not importable modules.

## Stage 1: `pre_processing(cache_frames, cache_video_features)`

**Owning package**: `packages/aitraf-core`

**Primary module surface**:

- `aitraf_core.pre_processing.video`
- `aitraf_core.pre_processing.cache`
- `aitraf_core.pre_processing.models.video_mae`

**Purpose**: Convert a selected video into temporal-fusion model inputs or
cache-compatible intermediate artifacts.

**Inputs**:

- `video_id`: Relative video id/path from the manifest.
- `clips_dir`: Root directory containing clips.
- `backbone`: VideoMAE backbone identity.
- `processor`: VideoMAE processor or loaded processor reference.
- `feature_model`: VideoMAE backbone model for feature extraction.
- `num_clips`: Number of temporal segments.
- `sample_frames`: Frames per segment.
- `sampling_dist`: Supported sampling distribution.
- `cache_frames`: Whether sampled-frame cache entries may be reused/written.
- `cache_video_features`: Whether VideoMAE feature cache entries may be
  reused/written.
- `frame_cache_dir`: Required when `cache_frames` is enabled.
- `features_dir`: Required when `cache_video_features` is enabled.
- `force_refresh`: Explicitly recompute and overwrite compatible cache entries.

**Outputs**:

- `SampledFrameSet` when callers need frame-level output.
- `VideoMaeFeatureSet` when callers need temporal-fusion feature output.
- Cache metadata describing the output contract.

**Required behavior**:

- Reuse sampled frames only when the sampled-frame cache contract matches.
- Reuse VideoMAE features only when the feature cache contract matches.
- Recompute frames/features only when cache is disabled, cache is missing, or
  explicit refresh is configured.
- Raise explicit errors for missing source videos, unsupported sampling
  distributions, or missing required cached features.

**Primary consumers**:

- `aitraf_train.data_ops.video_mae_feature_extraction`
- `aitraf_api.features.trick_assessment.service`

## Stage 2: `processing()`

**Owning package**: `packages/aitraf-core`

**Primary module surface**:

- `aitraf_core.processing.models.video_mae_temporal_fusion`
- `aitraf_core.processing.utils`

**Purpose**: Convert cached or freshly extracted VideoMAE features into
model-ready temporal-fusion inputs for training, evaluation, and API prediction.

**Inputs**:

- `manifest_row`: Current manifest row for train/eval or selected API video.
- `features`: `VideoMaeFeatureSet` or compatible feature tensor.
- `label_key`: Required for train/eval.
- `label_transform`: Required for train/eval when labels are present.
- `expected_num_clips`: Expected clip count.
- `expected_hidden_size`: Optional hidden size from model metadata.

**Outputs**:

- `TemporalFusionProcessedInput` with `features`, optional `labels`, and
  metadata.

**Required behavior**:

- Do not decode videos or run VideoMAE feature extraction in this stage.
- Require labels for training/evaluation samples.
- Allow label-free processing for API prediction.
- Keep collate behavior reusable by train/eval and testable with simple sample
  dictionaries.

**Primary consumers**:

- `aitraf_train.tasks.score_prediction_ordinal.video_mae_temporal_fusion.training`
- `aitraf_train.tasks.score_prediction_ordinal.video_mae_temporal_fusion.evaluation`
- `aitraf_api.features.trick_assessment.service`

## Stage 3: `predict()`

**Owning package**: `packages/aitraf-core`

**Primary module surface**:

- `aitraf_core.inference.models.video_mae_temporal_fusion`
- `aitraf_core.inference.classification`

**Purpose**: Run the temporal-fusion model on processed inputs and decode
user-facing trick AQA outputs.

**Inputs**:

- `loaded_model`: Cached temporal-fusion model instance.
- `processed_input`: `TemporalFusionProcessedInput`.
- `id2label` or equivalent decoder metadata.
- `model_kind`: Compact display model kind.

**Outputs**:

- Decoded label/value and confidence.

**Required behavior**:

- Put the model in eval/inference mode for serving prediction.
- Decode outputs with artifact-provided metadata, not API-local constants.
- Return confidence normalized from 0 to 1.
- Raise explicit errors for missing logits or unsupported output shape.

**Primary consumers**:

- `aitraf_api.prediction`
- Temporal-fusion evaluation smoke paths if prediction helpers are reused outside
  the Hugging Face Trainer flow.

## API Orchestration Contract

The trick AQA service should follow this sequence:

```text
row = resolve current trick AQA manifest row by video id
model_ref = resolve configured temporal-fusion model reference
loaded_model = load cached model(model_ref)
features = pre_processing(
    video_id=row.video_id,
    cache_frames=settings/cache flag,
    cache_video_features=settings/cache flag,
    contract=model_ref.preprocessing_contract,
)
processed = processing(row=row, features=features, label_required=False)
prediction = predict(loaded_model=loaded_model, processed_input=processed)
return demo-facing result
```

The API must not call `aitraf-train` data ops directly.

## Train/Data Pipeline Contract

`aitraf-train` keeps command ownership:

- `scripts/data_ops_pipeline.py` builds Hydra config and calls the data op.
- `data_ops/video_mae_feature_extraction.py` orchestrates clip listing,
  batching, logging, and calls `aitraf_core.pre_processing`.
- `tasks/score_prediction_ordinal/video_mae_temporal_fusion/training.py` and
  `evaluation.py` keep Trainer/MLflow/metrics ownership and call
  `aitraf_core.processing` for sample processing.

Train/eval code must not duplicate frame sampling or VideoMAE feature extraction.
