# Implementation Plan: API Inference Surface

**Branch**: `not set` | **Date**: 2026-06-19 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-api-inference/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Implement the reserved `packages/aitraf-api` package as a FastAPI inference
service for the Next.js demo frontend. The API exposes no-parameter health and
demo-video endpoints plus two inference endpoints that accept only a selected
video id. Demo videos are a flat display list built from matching rows in the
current `trick_classification` and `score_prediction_ordinal` test manifests.
Inference loads the selected video, uses shared `aitraf-core` video-to-frame
processing, loads the configured registered MLflow model, uses the processor and
preprocessing metadata packaged with that registered model, and returns decoded
predictions. Protected endpoints use token-based app authentication.

## Technical Context

**Language/Version**: Python 3.10+ (repo currently supports `>=3.10,<3.14`)

**Primary Dependencies**: FastAPI, Uvicorn, PyTorch, Transformers, pandas or JSONL parsing utilities, `aitraf-core` shared video processing, pytest

**Storage**: Repo-local manifests derived from `$AITRAF_DATA_PATH`: `manifests/trick_classification/test.jsonl` and `manifests/score_prediction_ordinal/test.jsonl`; registered MLflow model artifacts; video files derived from `$AITRAF_STORAGE_PATH/data/clips`; `.env`/environment for `AITRAF_DATA_PATH`, `AITRAF_STORAGE_PATH`, `MLFLOW_TRACKING_URI`, model URIs, API token, and any MLflow credentials

**Testing**: Focused `pytest` API tests using FastAPI's test client and Arrange/Act/Assert structure through fixtures, action calls, and assertions. Cover basic functionality only: health, missing/invalid-token auth rejection, flat demo-video listing, and valid inference requests through a stubbed predictor. Do not test model quality, model performance, metric values, or exhaustive edge cases.

**Target Platform**: Linux dev container / Docker runtime for local API development; GPU-capable runtime where configured models require it; CPU-compatible test path through dependency injection/stubs

**Project Type**: Monorepo Python ML pipeline with `packages/aitraf-api` as a first-class serving package, composing shared runtime helpers from `packages/aitraf-core`

**Performance Goals**: No model-performance targets in this feature. API smoke validation should prove request routing and orchestration work; latency/throughput tuning is out of scope.

**Constraints**: Use FastAPI; endpoints have fixed request shapes: health has no parameters, demo videos has no parameters, each inference endpoint takes only the video id. Demo-video response data is for display/selection and is not grouped by task. `.env`/environment is the source of truth for `AITRAF_DATA_PATH`, `AITRAF_STORAGE_PATH`, `MLFLOW_TRACKING_URI`, API auth token, MLflow credentials, and registered model URIs. API config is the source of truth only for endpoint-to-model mapping, compact demo-facing model kind, and derived manifest/clip path suffixes. The registered MLflow model/core inference path is the source of truth for trained model artifact, processor, preprocessing metadata, and output decoding metadata. Do not accept arbitrary uploaded videos. Do not add silent fallback predictions. Do not test model performance or metric quality here. Keep secrets out of committed files.

**Scale/Scope**: Four API capabilities: health, flat demo-video listing, trick classification inference by id, and trick AQA inference by id. Trick AQA maps to existing `score_prediction_ordinal` test-set and model surfaces.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. Missing auth token, missing MLflow tracking
  env, missing manifest rows, missing videos, missing endpoint model mapping,
  missing registered model, missing packaged processor/preprocessing metadata, or
  unsupported endpoint/model mappings must return explicit errors. The API must
  not return dummy or guessed predictions when runtime artifacts are unavailable.
- **Architecture Fit**: PASS WITH JUSTIFICATION. The previous surface split
  created `packages/aitraf-api` as the reserved serving surface. Implementing the
  API there extends that architecture rather than creating a parallel serving
  tree. Shared video and model-input processing remains in `aitraf-core`.
- **Function Decomposition**: PASS. Route handlers remain thin over separate
  helpers for settings/env loading, token validation, manifest loading, video id
  selection, registered model loading, packaged preprocessing execution,
  prediction decoding, ground truth decoding, and compact response shaping.
- **State And Mutation**: PASS. FastAPI app state and lazy registered-model
  handles are localized to API startup/runtime boundaries; manifest row selection
  and response shaping stay transformation-oriented.
- **Reproducibility**: PASS. Current manifests, selected video ids, `.env`
  MLflow tracking/model URI configuration, registered model package metadata,
  and documented test/run commands are sufficient for another developer to
  validate behavior.

### Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. Contracts require explicit 401/403/404/422/503
  style failures for unauthorized requests, invalid ids, missing manifests,
  missing videos, missing MLflow env, unavailable registered models, or registered
  models missing processor/preprocessing/decoding metadata.
- **Architecture Fit**: PASS. API code is owned by `packages/aitraf-api`; reusable
  video frame loading and model-input processing remain in `aitraf-core`; training
  and evaluation modules are not imported as serving code.
- **Function Decomposition**: PASS. Design artifacts separate entities and
  contracts so implementation can use narrow helpers and dependency injection for
  tests.
- **State And Mutation**: PASS. Runtime state is limited to env/settings, cached
  registered-model objects, and request-scoped prediction work.
- **Reproducibility**: PASS. Quickstart requires manifests, local clips or
  equivalent storage, `.env` MLflow/API configuration, and registered model URIs
  while keeping model quality validation out of the API test scope.

## Project Structure

### Documentation (this feature)

```text
specs/002-api-inference/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── openapi.yaml
└── tasks.md
```

### Source Code (repository root)

```text
packages/
├── aitraf-api/
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/aitraf_api/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── auth.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── demo_videos/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── route.py
│   │   │   │   └── service.py
│   │   │   ├── health/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── route.py
│   │   │   │   └── service.py
│   │   │   ├── trick_assessment/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── route.py
│   │   │   │   └── service.py
│   │   │   └── trick_classification/
│   │   │       ├── __init__.py
│   │   │       ├── route.py
│   │   │       └── service.py
│   │   ├── manifests.py
│   │   ├── prediction.py
│   │   ├── schemas.py
│   └── tests/
│       ├── conftest.py
│       ├── test_auth.py
│       └── features/
│           ├── demo_videos/test_demo_videos.py
│           ├── health/test_health.py
│           ├── trick_assessment/test_trick_assessment.py
│           └── trick_classification/test_trick_classification.py
├── aitraf-core/
│   └── src/aitraf_core/
│       └── processing/
│           ├── video.py
│           └── models/video_mae.py
└── aitraf-train/
    └── configs/
        ├── task/trick_classification.yaml
        ├── task/score_prediction_ordinal.yaml
        └── model/video_mae.yaml

data/
├── manifests/trick_classification/test.jsonl
└── manifests/score_prediction_ordinal/test.jsonl

storage/
└── data/clips/
```

**Structure Decision**: Implement API runtime code only in `packages/aitraf-api`.
Reuse `aitraf-core` for loading videos, sampling frames, and preparing model
inputs. The API must load the processor, preprocessing contract, output decoding
metadata, and compact model display metadata from the registered MLflow model
package. If existing model registration does not package those serving metadata,
implementation must fix the registration/logging path instead of adding
duplicated preprocessing constants to the API. Tests live in the API package and
should stub model prediction rather than loading real model weights or asserting
model quality. Inference responses should be demo-facing: return the selected
`video_id`, decoded prediction label/value with confidence, decoded ground truth
label/value, and compact model metadata such as `ordinal`, `video_mae`, or
`fusion`.

**Runtime Config Decision**: `.env`/environment provides `AITRAF_DATA_PATH`,
`AITRAF_STORAGE_PATH`, `MLFLOW_TRACKING_URI`, the API auth token, any MLflow
credentials, and the two registered MLflow model URIs. `aitraf-api`
configuration provides the endpoint mapping, compact demo-facing model kind, and
derived path suffixes for current manifests and clips. Do not define processor,
frame count, sampling strategy, or label decoding as API constants; those belong
with the registered model/core inference path.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
