# Implementation Plan: Demo Clip Download

**Branch**: `not set` | **Date**: 2026-07-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-demo-clip-download/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add an explicit runtime path for the API to download only the selected demo clips
when the service starts, without bundling clips into the API image and without
depending on `aitraf-train` orchestration from serving code.

The reusable S3 and clip-download behavior should move to `aitraf-core` because
both the API runtime and train data pipeline need it. `aitraf-api` remains the
owner of demo selection and startup hydration policy. `aitraf-train` remains the
owner of Hydra/data-ops orchestration and should reuse the core downloader rather
than maintaining separate S3 download logic.

The existing API image path should build without local clips. Runtime demo clip
download is opt-in through explicit configuration and fails loudly when required
manifests, clip source fields, credentials, or writable storage are unavailable.

## Technical Context

**Language/Version**: Python 3.10+ (repo currently supports `>=3.10,<3.14`)

**Primary Dependencies**: FastAPI, Uvicorn, `aitraf-core`, `aitraf-api`,
`aitraf-train`, boto3/botocore, python-dotenv, pandas for train label ingestion,
Docker Buildx/GHCR for image publishing

**Storage**: Repo `data/` manifests identify selected demo rows and their
`s3_path` clip sources. Runtime `AITRAF_STORAGE_PATH` provides
`data/clips/` for downloaded clips, `data/video_mae_features/` for feature
caches, and `models/` for model caches. S3-compatible object storage provides
source clips through AWS-compatible environment settings.

**Testing**: Unit tests for shared S3 parsing/download request behavior, API demo
download item selection, and train data-ops reuse. API package tests via
`uv run --package aitraf-api --extra dev pytest packages/aitraf-api/tests`.
Core tests via `uv run --package aitraf-core pytest packages/aitraf-core/tests`.
Train-focused tests where practical for label-to-download conversion. Smoke
validation with a temporary storage directory and mocked or real S3-compatible
credentials.

**Target Platform**: Linux Docker runtime for API serving; local/dev Linux
environment for data preparation; GitHub-hosted Ubuntu runner for API image
build/publish without local clips.

**Project Type**: Monorepo Python ML/API project with `aitraf-api` serving,
`aitraf-core` reusable runtime processing, and `aitraf-train` offline data,
training, evaluation, metrics, tracking, Hydra configs, and scripts.

**Performance Goals**: Runtime startup should download only currently selected
demo clips, skip existing clips by default, and avoid downloading the full clip
corpus. API image build should not depend on local clips and should remain small
relative to a clip-bundled image.

**Constraints**: Runtime download must be explicitly enabled; no hidden default
network downloads. Missing manifests, missing selected clip source fields,
missing object-storage credentials, inaccessible source objects, and unwritable
clip storage must fail explicitly. API code must not import `aitraf-train`.
Shared download behavior must not duplicate the current train S3 implementation.
No secrets or generated clips are committed.

**Scale/Scope**: One shared core S3/clip-download surface, one API demo clip
hydration path at startup, refactoring the existing train clip download data-op
to reuse the shared core surface, docs/validation updates, and cleanup of API
Docker/workflow clip bundling assumptions. No model behavior, label schema,
training algorithm, metrics, or hosting infrastructure changes are introduced.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. Runtime demo download is opt-in and missing
  manifests, sources, credentials, objects, or writable storage fail explicitly.
  Existing clips may be skipped only as declared cache behavior, not as a hidden
  repair path.
- **Package By Feature**: PASS. API startup/demo policy stays in
  `packages/aitraf-api`; reusable S3 and clip download primitives move to
  `packages/aitraf-core`; train Hydra/data-ops orchestration stays in
  `packages/aitraf-train`.
- **Function Decomposition**: PASS. The work decomposes into small helpers:
  parse/load S3 settings, represent clip download requests, download one clip,
  download many clips, select API demo clip requests, and adapt train labels to
  shared requests.
- **Functional Programming And State**: PASS. Selection and request construction
  can be pure transformations. Stateful clients and filesystem writes remain at
  explicit runtime boundaries.
- **Reproducibility**: PASS. Behavior is driven by versioned manifests, explicit
  environment settings, documented commands, and testable package surfaces.

### Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. Contracts require explicit failure for
  missing settings, malformed S3 URIs, unavailable objects, and storage write
  failures.
- **Package By Feature**: PASS. Data model and contracts preserve the API/core/
  train ownership split and avoid API imports from train.
- **Function Decomposition**: PASS. Contracts define separate core downloader,
  API hydration, and train adapter responsibilities.
- **Functional Programming And State**: PASS. Demo selection and download request
  construction are explicit transformations; network and filesystem effects are
  localized to downloader execution.
- **Reproducibility**: PASS. Quickstart documents image build without clips,
  package tests, and runtime smoke inputs needed to rerun validation.

## Project Structure

### Documentation (this feature)

```text
specs/005-demo-clip-download/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── api-runtime-demo-download.md
│   ├── core-clip-download.md
│   └── train-data-ops-reuse.md
└── tasks.md
```

### Source Code (repository root)

```text
packages/
├── aitraf-api/
│   ├── Dockerfile
│   ├── README.md
│   ├── Taskfile.yml
│   ├── src/aitraf_api/
│   │   ├── app.py
│   │   ├── config.py
│   │   └── features/demo_videos/
│   └── tests/
├── aitraf-core/
│   ├── pyproject.toml
│   ├── src/aitraf_core/
│   │   └── storage/
│   └── tests/
└── aitraf-train/
    ├── configs/
    ├── scripts/
    ├── src/aitraf_train/preparation/data_ops/
    └── tests/

data/
└── manifests/

storage/
└── data/clips/
```

**Structure Decision**: Add reusable S3 and clip download behavior under
`packages/aitraf-core/src/aitraf_core/storage/`. Keep API-specific demo
selection and startup hydration in `packages/aitraf-api/src/aitraf_api/features/
demo_videos/` or a neighboring API-owned service module. Keep train label/Hydra
orchestration in `packages/aitraf-train/src/aitraf_train/preparation/data_ops/`,
rewired to construct shared core download requests. Remove API Docker/workflow
requirements that expect local clips during image build.

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
