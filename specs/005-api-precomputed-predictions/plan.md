# Implementation Plan: AITRAF API Precomputed Predictions

**Branch**: `[005-api-precomputed-predictions]` | **Date**: 2026-07-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-api-precomputed-predictions/spec.md`

## Summary

Simplify `aitraf-api` so the demo API stops doing live inference.

The new runtime behavior is intentionally small:

1. All `aitraf-train` evaluation scripts support logging full test prediction
   artifacts to MLflow, including video metadata and prediction values.
2. The classification and AQA prediction run IDs are supplied externally after
   evaluation runs exist in MLOps with full `test_predictions.json` artifacts.
3. On startup, the API downloads the classification full test prediction
   artifact from the configured MLflow run ID using the fixed artifact path in
   API code.
4. On startup, the API downloads the AQA full test prediction artifact from the
   configured MLflow run ID using the fixed artifact path in API code.
5. The API matches the two downloaded prediction artifacts by `video_id`.
6. The API stores the final demo predictions response in memory.
7. On `GET /demo-predictions`, the API returns the in-memory demo predictions
   response.

No model loading, no video decoding, no feature extraction, no Torch, no
Transformers, no CUDA, and no `aitraf-core` in the API image. `aitraf-core`
stays in the repo unchanged for reusable ML runtime code used by train/eval and
future offline workflows.

## Technical Context

**Language/Version**: Python 3.10+ (`>=3.10,<3.14`)

**Primary Dependencies**: FastAPI, Pydantic, python-dotenv, uvicorn, MLflow
artifact download client.

**Dependencies To Remove From `aitraf-api` Runtime**: `aitraf-core`, `torch`,
`transformers`, CUDA image dependencies, ffmpeg, model cache behavior, and
serving-time preprocessing dependencies.

**Storage**: Full test prediction artifacts are generated outputs stored in
MLOps/MLflow by `aitraf-train` evaluation/export. They are downloaded at API
startup by run ID plus fixed code-level artifact paths and are not committed to
the repo.

**Inspected Existing Eval Run IDs**:
- Classification predictions run: `0a232293f2d44976abc5830953b06a76`
- AQA predictions run: `e50e861919cc4187bc4f58c5ab9dc717`

These existing runs were inspected during planning and do not contain full test
prediction artifacts; they only contain metrics, plots, params, and
`misses_summary.json`. Implementation adds full prediction artifact logging;
usable validation run IDs are supplied externally after evaluation is run.

**External Evaluation Input**: Running evaluation for the registered demo models
is outside the Codex implementation plan. The API implementation expects two
MLflow run IDs that already contain full `test_predictions.json` artifacts.

**Supplied Validation Prediction Run IDs**:
- Classification predictions run: `2b2208e417e34e2198bb108e4f683cf9`
- AQA predictions run: `da6a8082c5e646448c7a79cd124b8e09`
- Both runs were verified to contain `test_predictions.json` with 100 rows and
  the required `video_id`, `s3_path`, `person`, `key_foot`, `trick`,
  `execution_score`, and `label` fields.

**Registered Models To Re-Evaluate**:
- Trick classification registered model currently configured as
  `models:/aitraf-trick-classification@infant`
- AQA registered model currently configured as `models:/aitraf-trick-aqa@infant`
- These are reference model URIs for the external evaluation runs; the API
  itself consumes only prediction run IDs.

**Eval Scripts To Update**:
- `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/video_mae/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/video_mae_temporal_fusion/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/trick_classifier/pose_tcn/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction/video_mae/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction/pose_tcn/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/video_mae/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/video_mae_temporal_fusion/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_ordinal/pose_tcn/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_binary/video_mae/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_binary/video_mae_temporal_fusion/evaluation.py`
- `packages/aitraf-train/src/aitraf_train/tasks/score_prediction_pairwise/video_mae/evaluation.py`

**Testing**: A small endpoint test suite covers `GET /demo-predictions` success
and authentication. Local smoke validation starts the API with the supplied
MLflow run IDs and confirms startup downloads both `test_predictions.json`
artifacts. Real Docker smoke validation must run the API container with those
run IDs, let the container download the artifacts at startup, then call
`/health` plus authenticated `/demo-predictions`. Docker build validation does
not require Codex to rerun model evaluation.

**Target Platform**: CPU-only demo API container.

**Project Type**: FastAPI service in a Python monorepo.

**Performance Goals**: No live inference work on startup or request. Startup
does one simple matching/filtering pass over downloaded prediction rows.
Request time returns the already-prepared in-memory response.

**Scope**: Demo-only selected test videos. Supported tasks are trick
classification and trick AQA/assessment. Uploaded/arbitrary video inference is
out of scope and removed from the API surface.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. Missing prediction run ID config, failed
  download, or unreadable files fail explicitly. The API does not fall back to
  live inference.
- **Package By Feature**: PASS. API serving changes stay in
  `packages/aitraf-api`; prediction artifact generation belongs in
  `packages/aitraf-train`. `aitraf-core` remains the reusable ML runtime package
  but leaves the API runtime.
- **Function Decomposition**: PASS. Keep helpers small: download artifact, read
  rows, match task rows by `video_id`, build response.
- **Functional Programming And State**: PASS. The prepared demo response is
  stored at the FastAPI boundary; matching and response mapping are pure
  transformations.
- **Reproducibility**: PASS. Runtime config points at explicit MLflow run IDs,
  while artifact paths are fixed constants in API code. Generated prediction
  files are not source files.
- **No Legacy Compatibility**: PASS. Old live inference routes/config/tests/docs
  are removed instead of kept as shims.
- **Required Types Over Defensive Normalization**: PASS. Startup expects the
  configured MLflow run IDs and fixed `test_predictions.json` schema directly;
  artifact readers reject records that do not match that schema instead of
  accepting alternate shapes and coercing them.

## Project Structure

```text
packages/aitraf-api/
├── src/aitraf_api/
│   ├── app.py                  # download/read artifacts at startup
│   ├── auth.py                 # keep
│   ├── config.py               # API token + prediction run ID config
│   ├── schemas.py              # demo-predictions response schemas
│   └── features/
│       ├── health/             # keep
│       └── demo_predictions/   # artifact loading + prepared response route
├── tests/
├── Dockerfile
├── README.md
└── pyproject.toml
```

```text
packages/aitraf-train/
└── src/aitraf_train/
    └── tasks/
        ├── trick_classifier/.../evaluation.py
        └── score_prediction_ordinal/.../evaluation.py
```

Remove from `packages/aitraf-api`:

- `features/trick_classification`
- `features/trick_assessment`
- API live model loading imports
- API model/preprocessing dependencies
- API config for live model URIs, model cache paths, and prediction artifact
  path environment variables
- API Docker CUDA/ffmpeg/core copy behavior

Keep outside the API runtime:

- `packages/aitraf-core` as reusable ML runtime code
- `packages/aitraf-train` as training/eval/MLOps artifact producer

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design And Contracts

See [data-model.md](./data-model.md), [contracts/openapi.yaml](./contracts/openapi.yaml),
and [quickstart.md](./quickstart.md).

## Implementation Direction

1. Add shared `aitraf-train` full prediction export helpers and wire them into
   all evaluation scripts so each eval can log a full `test_predictions.json`
   artifact with one row per test example. Rows must include `video_id`, display
   metadata (`s3_path`, `person`, `key_foot`, `trick`, `execution_score` when
   available),
   predicted label, and optional confidence/metadata.
2. Add config for two prediction run IDs:
   - `AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID`
   - `AITRAF_AQA_PREDICTIONS_RUN_ID`
   The artifact paths are fixed constants in API code, not `.env` values.
3. Add a tiny API-owned helper that downloads an MLflow artifact and reads rows
   from the artifact file.
4. At startup, download/read both artifacts.
5. Keep only basic startup checks:
   - prediction run ID config exists
   - artifact download succeeds
   - artifact file can be parsed
   - rows have `video_id`
   - rows have the prediction fields needed by the response
6. Add `GET /demo-predictions`:
   - startup matches classification and AQA rows by `video_id`
   - startup builds the final demo predictions response and stores it in memory
   - request time returns that stored response
7. Remove old live inference endpoints and route registration.
8. Remove API dependencies on `aitraf-core`, `torch`, and `transformers`.
9. Simplify the API Docker image to CPU-only Python runtime.
10. Refactor root `.env.example` for the simplified API:
   - remove `AITRAF_CLASSIFICATION_MODEL_URI`
   - remove `AITRAF_AQA_MODEL_URI`
   - remove unused `AITRAF_DATA_PATH`
   - keep `AITRAF_STORAGE_PATH` for train image and offline workflows
   - remove API code dependence on `AITRAF_STORAGE_PATH`
   - add `AITRAF_CLASSIFICATION_PREDICTIONS_RUN_ID`
   - add `AITRAF_AQA_PREDICTIONS_RUN_ID`
   - do not add prediction artifact path env vars
11. Rewrite API tests and README for the simplified precomputed-prediction API.
12. Build the Docker image and smoke test the API path with externally supplied
    prediction run IDs:
    - verify `/health`
    - verify authenticated `GET /demo-predictions`
    - verify the container downloads `test_predictions.json` from MLflow at
      startup
    - verify the response is served from cached precomputed predictions, not
      live inference

## Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. The API either has the configured run IDs
  and fixed artifact paths or fails; it never runs inference as backup.
- **Package By Feature**: PASS. All serving behavior remains in
  `packages/aitraf-api`; core/train boundaries are unchanged.
- **Function Decomposition**: PASS. Implementation is small helpers plus route
  response mapping.
- **Functional Programming And State**: PASS. Startup behavior is list matching
  and response construction; request behavior returns prepared data.
- **Reproducibility**: PASS. `aitraf-train` produces full prediction artifacts,
  prediction run ID config identifies the MLOps inputs, and artifact paths are
  fixed in versioned API code.
- **No Legacy Compatibility**: PASS. Unsupported inference API behavior is
  deleted.
- **Required Types Over Defensive Normalization**: PASS. The API serves one
  prepared demo prediction schema from one required artifact shape; it does not
  support fallback artifact formats, scalar-or-list variants, or legacy response
  adapters.

## Complexity Tracking

No constitution violations or complexity exceptions.
