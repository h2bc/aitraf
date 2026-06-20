# Data Model: Temporal Fusion Trick AQA

## TrickAQADemoVideo

Represents one current trick AQA manifest row selectable by the API.

**Fields**:

- `id`: Stable selection id returned by the demo-video API, currently the
  manifest `video_id`.
- `video_id`: Relative video path from the trick AQA manifest.
- `execution_score`: Ground-truth assessment label or value.
- `trick`: Optional display metadata.
- `person`: Optional display metadata.
- `s3_path`: Optional source metadata.

**Relationships**:

- Selected by `TemporalFusionPredictionRequest`.
- Points to a video under configured clip storage.
- Can produce one `SampledFrameSet` and one `VideoMaeFeatureSet` per compatible
  preprocessing contract.

**Validation Rules**:

- `video_id` is required and must resolve in the current trick AQA manifest.
- The resolved video must exist under the configured clip storage before
  prediction.
- Missing rows or missing clips are explicit errors.

## TemporalFusionModelReference

Represents the configured registered temporal-fusion model used by trick AQA.

**Fields**:

- `model_uri`: Registered model reference from environment/configuration.
- `model_kind`: Compact display value, expected to identify temporal fusion.
- `task_name`: Expected task identity, `score_prediction_ordinal` for trick AQA.
- `backbone`: VideoMAE backbone identity used by feature extraction.
- `num_clips`: Number of temporal clips expected by the model.
- `sample_frames`: Number of sampled frames per clip.
- `sampling_dist`: Sampling distribution used for serving.
- `decoder_metadata`: Label/value mapping required to decode predictions.

**Relationships**:

- Loaded by the model cache.
- Supplies contract fields used by `SampledFrameCacheContract` and
  `VideoMaeFeatureCacheContract`.
- Produces `TemporalFusionPredictionResult`.

**Validation Rules**:

- Missing `model_uri`, task metadata, preprocessing metadata, or decoder metadata
  is an explicit service error.
- The configured model kind must not be substituted with ordinal, baseline,
  classification, dummy, or non-fusion model kinds.
- The artifact task/output contract must match trick AQA expectations.

## TemporalFusionPredictionRequest

Represents one approved API request to predict trick AQA for a selected demo
video.

**Fields**:

- `video_id`: Selected trick AQA video id.
- `cache_frames`: Boolean controlling sampled-frame cache reuse.
- `cache_video_features`: Boolean controlling VideoMAE feature cache reuse.
- `auth_token`: Approved app token supplied by the caller.

**Relationships**:

- Resolves one `TrickAQADemoVideo`.
- Uses one `TemporalFusionModelReference`.
- Produces one `TemporalFusionPredictionResult`.

**Validation Rules**:

- `video_id` must resolve in the current trick AQA manifest.
- Auth must be valid before protected prediction work runs.
- Cache flags control frame/feature cache reuse only; they do not permit stale
  data or alternate model fallback.

## SampledFrameCacheContract

Describes the compatibility identity for sampled frames.

**Fields**:

- `video_id`: Video identity.
- `clip_path`: Concrete clip path used to read frames.
- `num_clips`: Number of temporal segments.
- `sample_frames`: Frames per segment.
- `sampling_dist`: Sampling distribution.
- `source_mtime` or `source_digest`: Video source identity used to detect stale
  entries.

**Relationships**:

- Identifies compatibility for `SampledFrameSet`.
- Feeds VideoMAE feature extraction.

**Validation Rules**:

- Cache reuse is valid only when all contract fields match.
- Missing or incompatible entries are not silently accepted.
- If `cache_frames` is disabled, frame sampling recomputes from the source video.

## SampledFrameSet

Represents decoded and sampled frames for one video/contract.

**Fields**:

- `contract`: `SampledFrameCacheContract`.
- `frames`: Segmented frame tensors or arrays.
- `frame_indices`: Frame indices selected from each segment.

**Relationships**:

- Can be loaded from or written to the sampled-frame cache.
- Used to produce `VideoMaeFeatureSet`.

**Validation Rules**:

- `frames` and `frame_indices` must match `num_clips` and `sample_frames`.
- Shape mismatches are explicit errors.

## VideoMaeFeatureCacheContract

Describes the compatibility identity for cached VideoMAE features.

**Fields**:

- `video_id`: Video identity.
- `backbone`: VideoMAE backbone identity.
- `processor_id`: Processor identity or equivalent artifact reference.
- `model_version`: Backbone or artifact version used for feature extraction.
- `num_clips`: Number of temporal clips.
- `sample_frames`: Frames per clip.
- `sampling_dist`: Sampling distribution.
- `preprocessing_version`: Version or hash of preprocessing behavior.
- `frame_contract`: Source `SampledFrameCacheContract` fields or digest.

**Relationships**:

- Identifies compatibility for `VideoMaeFeatureSet`.
- Derived from `TemporalFusionModelReference` and `SampledFrameSet`.

**Validation Rules**:

- Cache reuse is valid only when the feature contract matches the requested
  prediction contract.
- Incompatible cached features must be rejected or recomputed explicitly.
- Path existence alone is not sufficient to prove compatibility.

## VideoMaeFeatureSet

Represents temporal clip embeddings used as temporal-fusion model inputs.

**Fields**:

- `contract`: `VideoMaeFeatureCacheContract`.
- `features`: Tensor shaped for temporal-fusion input.
- `frame_indices`: Frame indices used to derive features.

**Relationships**:

- Produced by `aitraf_core.pre_processing`.
- Consumed by `TemporalFusionProcessedInput`.
- Reused by data pipelines, train/eval processing, and API prediction when
  contract-compatible.

**Validation Rules**:

- Feature tensor shape must match the temporal-fusion model contract.
- Missing or malformed features are explicit errors.

## TemporalFusionProcessedInput

Represents model-ready inputs after processing.

**Fields**:

- `features`: Float tensor batch accepted by the temporal-fusion model.
- `labels`: Optional tensor used by training/evaluation, absent for API
  prediction.
- `metadata`: Video id and contract metadata carried for tracing.

**Relationships**:

- Built from `VideoMaeFeatureSet`.
- Consumed by `aitraf_core.inference` prediction helpers and `aitraf-train`
  train/eval data collators.

**Validation Rules**:

- API prediction must not require labels.
- Training/evaluation must fail explicitly if required labels are missing.

## TemporalFusionPredictionResult

Represents the demo-facing trick AQA result returned by the API.

**Fields**:

- `video_id`: Selected video id.
- `prediction.label`: Decoded assessment label or value.
- `prediction.confidence`: Normalized confidence from 0 to 1.
- `ground_truth.label`: Decoded ground-truth assessment from the manifest.
- `model.kind`: Compact temporal-fusion model identity.

**Relationships**:

- Produced from `TemporalFusionProcessedInput` and
  `TemporalFusionModelReference`.
- Returned by the trick AQA inference endpoint.

**Validation Rules**:

- Result must identify temporal fusion as the model kind.
- Prediction output must be decoded from model output metadata.
- Missing decoding metadata is an explicit service error.
