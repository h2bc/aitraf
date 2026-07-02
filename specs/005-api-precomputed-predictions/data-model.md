# Data Model: AITRAF API Precomputed Predictions

## Artifact Source

Configured MLflow/MLOps source for one prediction artifact.

**Fields**:
- `run_id`: MLflow run ID from environment config.
- `task`: `trick_classification` or `trick_aqa`.

**Rules**:
- `run_id` is required in runtime config.
- The artifact path is the fixed API code constant `test_predictions.json`, not
  an environment variable or source-field value.
- The API fails startup if the artifact cannot be downloaded or read.

## Prediction Row

One row from a downloaded prediction artifact.

**Fields**:
- `video_id`: video identifier used for filtering.
- `label`: predicted label or score display value.
- `confidence`: optional confidence value when the artifact provides it.
- `s3_path`: video path used by the demo when present.
- `person`: display metadata when present.
- `trick`: ground-truth trick label when present.
- `execution_score`: ground-truth AQA score when present.
- `metadata`: optional extra fields already present in the artifact.

**Rules**:
- `video_id` is required.
- `label` is required.
- The API does not run model logic to repair or fill missing predictions.

## Video Metadata

Video metadata embedded in each full prediction artifact row.

**Fields**:
- `video_id`
- `s3_path`
- `person`
- `trick`
- `execution_score`

**Rules**:
- The API does not need a separate local manifest when the prediction artifacts
  include these metadata fields.
- A record is returned only when matching classification and AQA prediction rows
  exist.

## Demo Predictions Response

Response from `GET /demo-predictions`.

**Fields**:
- Root JSON value is a list of demo prediction records.
- Each record includes video metadata plus:
  - `predictions.trick_classification.label`
  - `predictions.trick_classification.confidence`
  - `predictions.trick_aqa.label`
  - `predictions.trick_aqa.confidence`

**Rules**:
- Response is built at startup by matching the two downloaded prediction row
  lists by `video_id`.
- Task prediction objects do not repeat video metadata or raw artifact metadata;
  shared display fields live on the demo prediction record.
- No live inference fields or model runtime state are exposed.
