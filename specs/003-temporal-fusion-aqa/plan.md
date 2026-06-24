# Implementation Plan: Temporal Fusion Trick AQA

**Branch**: `not set` | **Date**: 2026-06-20 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-temporal-fusion-aqa/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add temporal-fusion trick AQA inference to the API by refactoring shared runtime
behavior into `aitraf-core` layers that can be reused by both offline pipelines
and serving:

- `aitraf_core.pre_processing`: video-to-model-input preparation, including frame
  sampling, optional sampled-frame cache, VideoMAE feature extraction, and
  VideoMAE feature cache contract checks.
- `aitraf_core.processing`: model-input processing used by train/eval/API,
  including temporal-fusion feature sample shaping, label transforms, collate
  helpers, and model-input validation.
- `aitraf_core.inference`: loaded-model prediction and output decoding, including
  temporal-fusion trick AQA prediction.

`aitraf-train` should keep its current `tasks/<task>/<model>/training.py` and
`evaluation.py` structure, but move duplicated VideoMAE temporal-fusion
pre-processing into `aitraf-core`. Data ops reuse `pre_processing(...)`;
train/eval reuse `processing()` and `predict()` where relevant; the API service
orchestrates `pre_processing(cache_frames=?, cache_video_features=?)`,
`processing()`, and `predict()` for trick AQA.

## Technical Context

**Language/Version**: Python 3.10+ (repo currently supports `>=3.10,<3.14`)

**Primary Dependencies**: FastAPI, PyTorch, Transformers, MLflow, Hydra,
datasets, pandas/JSONL utilities, pytest

**Storage**: Repo-local `data/` manifests, `storage/data/clips`, generated
sampled-frame cache entries, generated `storage/` VideoMAE feature artifacts,
registered MLflow temporal-fusion model artifacts, `.env`/environment for data
paths, storage paths, MLflow, API auth token, and registered model URIs

**Testing**: Focused pytest coverage in `packages/aitraf-core`,
`packages/aitraf-train`, and `packages/aitraf-api`, plus command-level smoke
validation for data ops, eval, and API inference. API tests use stubs for model
loading and prediction; core cache tests use small tensors/files.

**Target Platform**: Linux dev container / Docker runtime for local API and test
work; CUDA-capable hosts for real VideoMAE feature extraction and temporal-fusion
training/evaluation; CPU-compatible unit-test paths through dependency injection
and small fixtures

**Project Type**: Monorepo Python ML pipeline with `aitraf-core` shared runtime
helpers, `aitraf-train` offline data/training/evaluation package, and
`aitraf-api` FastAPI serving package

**Performance Goals**: Repeated API trick AQA requests for the same model/video
must reuse the loaded model instance. When enabled and contract-compatible,
sampled-frame and VideoMAE feature caches must avoid repeated video decoding and
feature extraction for the same video/sampling/model contract. No model-quality
or latency target is introduced in this planning phase.

**Constraints**: Preserve package-by-feature ownership. Use importable Python
module names, so the requested pre-processing layer is implemented as
`aitraf_core.pre_processing`. Do not duplicate temporal-fusion frame sampling or
VideoMAE feature extraction in `aitraf-train` and `aitraf-api`. Do not add hidden
fallbacks: missing manifests, videos, model refs, processors, cache contract
mismatches, decoding metadata, or unsupported task/model combinations must fail
explicitly or recompute only when recomputation is configured and possible.
Serving must not accept arbitrary uploaded videos.

**Scale/Scope**: One API capability update for trick AQA, plus shared core
refactor used by the existing temporal-fusion data op, `score_prediction_ordinal`
train/eval flow, and API inference. Existing trick classification and non-fusion
model flows should continue to pass.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. The plan requires explicit errors for missing
  configuration, artifacts, manifests, videos, unsupported model/task mappings,
  and incompatible cache entries. Cache reuse is allowed only with matching
  contract metadata; otherwise recomputation must be explicit and configured.
- **Package By Feature**: PASS. Serving changes stay in `packages/aitraf-api`;
  shared frame/feature/model-input/prediction helpers move to
  `packages/aitraf-core`; offline data/training/evaluation orchestration stays in
  `packages/aitraf-train`.
- **Function Decomposition**: PASS. The requested service shape separates
  `pre_processing(...)`, `processing()`, and `predict()` into small helpers with
  explicit inputs, outputs, and failure modes.
- **Functional Programming And State**: PASS. Core helpers are
  transformation-oriented. Mutable state is limited to controlled runtime caches:
  API model cache and optional file-backed frame/feature caches.
- **Reproducibility**: PASS. Cache contracts, Hydra config, manifests, model
  references, MLflow model metadata, and smoke commands are documented as
  reproducibility inputs.

### Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. Contracts require explicit failures for
  invalid selections, missing model metadata, unavailable videos, unsupported
  mappings, and incompatible cache contracts. The only non-error cache miss path
  is explicit recomputation when cache use is disabled or refresh is configured.
- **Package By Feature**: PASS. The design introduces `aitraf_core.pre_processing`
  because both `aitraf-train` and `aitraf-api` need the behavior. Train package
  structure remains `data_ops/`, `tasks/<task>/<model>/training.py`, and
  `tasks/<task>/<model>/evaluation.py`.
- **Function Decomposition**: PASS. Data model and contracts split cached frame
  preparation, feature extraction, feature-sample processing, model loading,
  prediction, and response shaping.
- **Functional Programming And State**: PASS. Cache readers/writers receive
  explicit contract objects. API model caching remains localized behind a loader.
- **Reproducibility**: PASS. Quickstart covers data ops cache generation, core/API
  tests, train/eval smoke checks, API smoke requests, and required env/config
  values.

## Project Structure

### Documentation (this feature)

```text
specs/003-temporal-fusion-aqa/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── core-pipeline.md
│   └── openapi.yaml
└── tasks.md
```

### Source Code (repository root)

```text
packages/
├── aitraf-api/
│   ├── src/aitraf_api/
│   │   ├── config.py
│   │   ├── prediction.py
│   │   ├── schemas.py
│   │   └── features/trick_assessment/
│   │       ├── route.py
│   │       └── service.py
│   └── tests/
│       ├── conftest.py
│       └── features/trick_assessment/
├── aitraf-core/
│   └── src/aitraf_core/
│       ├── pre_processing/
│       │   ├── __init__.py
│       │   ├── cache.py
│       │   ├── video.py
│       │   └── models/
│       │       ├── __init__.py
│       │       └── video_mae.py
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── labels.py
│       │   ├── utils.py
│       │   └── models/
│       │       ├── video_mae.py
│       │       └── video_mae_temporal_fusion.py
│       └── inference/
│           ├── __init__.py
│           ├── classification.py
│           └── models/
│               ├── video_mae.py
│               └── video_mae_temporal_fusion.py
└── aitraf-train/
    ├── configs/
    │   ├── data_ops.yaml
    │   └── model/video_mae_temporal_fusion.yaml
    ├── scripts/
    │   ├── data_ops_pipeline.py
    │   ├── train.py
    │   └── eval.py
    └── src/aitraf_train/
        ├── data_ops/video_mae_feature_extraction.py
        └── tasks/score_prediction_ordinal/video_mae_temporal_fusion/
            ├── training.py
            └── evaluation.py
```

**Structure Decision**: Keep `aitraf-train` as the owner of Hydra command
orchestration, task configs, metrics, tracking, and task-specific training/eval
entrypoints. Move reusable temporal-fusion preparation and prediction primitives
to `aitraf-core` only where both train/data ops and API need the behavior. Keep
API route/service code thin over settings, manifest row selection, clip
validation, cached pre-processing, processing, prediction, and response shaping.
Do not create a new top-level serving or training subsystem.

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
