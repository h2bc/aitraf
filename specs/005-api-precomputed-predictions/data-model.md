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
- `s3_path`: artifact source path converted by the API into an internal S3
  object key for URL signing.
- `thumbnail_s3_path`: API-prepared thumbnail source path converted into a presigned URL.
- `person`: display metadata when present.
- `key_foot`: skater stance metadata when present.
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
- `thumbnail_s3_path`
- `person`
- `key_foot`
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
  - `video_url`: 900-second presigned browser-playable HTTP(S) URL for the demo
    video.
  - `thumbnail_url`: 900-second presigned HTTP(S) URL for the video thumbnail.
  - `person` and `key_foot`: shared display metadata for the clip.
  - `predictions.trick_classification.label`
  - `predictions.trick_classification.confidence`
  - `predictions.trick_aqa.label`
  - `predictions.trick_aqa.confidence`

**Rules**:
- Internal records are built at startup by matching the two downloaded
  prediction row lists by `video_id`; `video_url` is generated while serving
  `GET /demo-predictions`.
- Raw artifact `s3_path` and `thumbnail_s3_path` values are not returned publicly.
- Task prediction objects do not repeat video metadata or raw artifact metadata;
  shared display fields live on the demo prediction record.
- No live inference fields or model runtime state are exposed.
