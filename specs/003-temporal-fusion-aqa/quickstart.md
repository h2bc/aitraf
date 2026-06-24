# Quickstart: Temporal Fusion Trick AQA

## Prerequisites

- Workspace dependencies installed with `uv sync`.
- Current trick AQA manifest present:
  - `data/manifests/score_prediction_ordinal/test.jsonl`
- Local clips available under `$AITRAF_STORAGE_PATH/data/clips`.
- Temporal-fusion model config available:
  - `packages/aitraf-train/configs/model/video_mae_temporal_fusion.yaml`
- API environment/configuration includes:
  - `AITRAF_API_TOKEN`
  - `AITRAF_DATA_PATH`
  - `AITRAF_STORAGE_PATH`
  - `MLFLOW_TRACKING_URI`
  - `AITRAF_AQA_MODEL_URI` pointing at the temporal-fusion trick AQA model
- VideoMAE feature cache and optional sampled-frame cache locations configured
  through Hydra/API settings.
- The temporal-fusion registered model artifact includes preprocessing contract
  metadata and output decoder metadata.

## Validation Scope

This feature validates orchestration and reuse, not model quality:

- `aitraf-core` exposes shared pre-processing, processing, and inference helpers.
- `aitraf-train` data ops reuse core pre-processing for VideoMAE feature
  extraction.
- `aitraf-train` temporal-fusion train/eval reuse core processing helpers.
- `aitraf-api` trick AQA prediction orchestrates
  `pre_processing(cache_frames=?, cache_video_features=?)`, `processing()`, and
  `predict()`.
- Model cache, frame cache, and VideoMAE feature cache behavior is covered by
  focused tests or smoke checks.

Do not use this feature's API tests to validate model performance, metric
quality, or production latency.

## Run Focused Tests

```bash
task api:test
```

Expected outcome: API tests pass, including trick AQA temporal-fusion response
shape, invalid selection failures, model cache reuse, and no fallback to the old
ordinal model kind.

Run core/train tests directly if package task aliases are not available:

```bash
uv run pytest packages/aitraf-core packages/aitraf-train packages/aitraf-api
```

Expected outcome: core cache tests, train reuse tests, and API trick AQA tests
pass without downloading real model weights in unit paths.

Latest local validation: `uv run pytest packages/aitraf-api packages/aitraf-core/tests packages/aitraf-train/tests` passed with 16 tests.

## Generate Or Refresh VideoMAE Feature Cache

```bash
uv run python packages/aitraf-train/scripts/data_ops_pipeline.py \
  video_mae_feature_extraction.enabled=true \
  video_mae_feature_extraction.force=false \
  video_mae_feature_extraction.limit=2
```

Expected outcome: the data op lists clips, uses `aitraf_core.pre_processing` for
frame sampling and VideoMAE feature extraction, skips compatible cached features,
and writes contract metadata for new feature entries.

To force refresh compatible feature entries:

```bash
uv run python packages/aitraf-train/scripts/data_ops_pipeline.py \
  video_mae_feature_extraction.enabled=true \
  video_mae_feature_extraction.force=true \
  video_mae_feature_extraction.limit=2
```

Expected outcome: existing compatible VideoMAE features are recomputed
explicitly.

## Train/Eval Smoke

Train temporal fusion for the ordinal score task with a small sample limit:

```bash
uv run python packages/aitraf-train/scripts/train.py \
  task=score_prediction_ordinal \
  model=video_mae_temporal_fusion \
  max_samples=4
```

Expected outcome: training uses cached VideoMAE features through
`aitraf_core.processing` sample processing and logs a temporal-fusion model URI.

Evaluate a registered temporal-fusion model:

```bash
uv run python packages/aitraf-train/scripts/eval.py \
  task=score_prediction_ordinal \
  model=video_mae_temporal_fusion \
  model_id="replace-with-model-name-or-version"
```

Expected outcome: evaluation uses the temporal-fusion task structure, reads
cached features through core processing, logs metrics, and fails loudly if cache
entries or model metadata are missing.

## Run API Locally

```bash
export AITRAF_API_TOKEN="replace-with-local-token"
export MLFLOW_TRACKING_URI="replace-with-mlflow-uri"
export AITRAF_AQA_MODEL_URI="models:/aitraf-trick-aqa-temporal-fusion@infant"
export AITRAF_CLASSIFICATION_MODEL_URI="models:/aitraf-trick-classification@infant"
export AITRAF_DATA_PATH="data"
export AITRAF_STORAGE_PATH="storage"
task api:run
```

Expected outcome: FastAPI starts and exposes docs at
`http://localhost:8000/docs`.

## Smoke Requests

Fetch demo videos:

```bash
curl -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  http://localhost:8000/demo-videos
```

Run trick AQA with configured cache defaults:

```bash
curl -X POST -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  http://localhost:8000/inference/trick-aqa/{id}
```

Run trick AQA with explicit cache flags:

```bash
curl -X POST -H "Authorization: Bearer $AITRAF_API_TOKEN" \
  "http://localhost:8000/inference/trick-aqa/{id}?cache_frames=true&cache_video_features=true"
```

Expected outcome: valid ids return the selected video id, decoded trick AQA
prediction, ground truth, and temporal-fusion model identity. Missing model
metadata, missing cached features, missing videos, and invalid ids return explicit
errors rather than fallback predictions.

## Related Artifacts

- Implementation plan: `specs/003-temporal-fusion-aqa/plan.md`
- Data model: `specs/003-temporal-fusion-aqa/data-model.md`
- Core pipeline contract: `specs/003-temporal-fusion-aqa/contracts/core-pipeline.md`
- API contract: `specs/003-temporal-fusion-aqa/contracts/openapi.yaml`
