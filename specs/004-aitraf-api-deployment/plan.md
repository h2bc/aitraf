# Implementation Plan: AITRAF API Deployment

**Branch**: `not set` | **Date**: 2026-06-25 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-aitraf-api-deployment/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add a production Docker image path for `aitraf-api` and extend the existing GHCR
publishing workflow so both train and API images are built on `master` updates.

The API Dockerfile should reuse the existing train image conventions where they
fit: CUDA runtime base, `uv`, frozen workspace install, root build context, and
GHCR metadata tags. It must diverge where serving scope differs: install only
`aitraf-api` and its `aitraf-core` dependency, copy the small repo `data/`
directory into the image for manifests/vocabularies, keep `storage/` and model
artifacts external, and start the API server rather than a long-running shell
placeholder.

The existing image workflow should become the shared Docker-image publishing
workflow. Train publishing remains independent. API publishing must run API tests
first and publish to `ghcr.io/<owner>/aitraf-api` only when those tests pass. If
API tests fail, the API image path fails without blocking the train image job
from continuing.

## Technical Context

**Language/Version**: Python 3.10+ (repo currently supports `>=3.10,<3.14`)

**Primary Dependencies**: FastAPI, Uvicorn, PyTorch, Transformers, MLflow,
`aitraf-core`, `uv`, Docker Buildx, GitHub Actions, GHCR

**Storage**: Copy repo `data/` into the API image for manifests and vocabularies
(`data/` is small, about 1.8 MB locally). Do not copy repo `storage/` into the
image (`storage/` is external runtime state and very large locally). Runtime
`AITRAF_STORAGE_PATH` continues to point at mounted or externally available
clips, feature caches, model caches, and MLflow/model inputs.

**Testing**: Run `task api:test` or equivalent `uv run pytest
packages/aitraf-api/tests` before API publishing. Validate the Dockerfile with a
local image build. Validate real container startup/health only with required
runtime model/data/storage/credential inputs available.

**Target Platform**: Linux Docker runtime for API serving; GitHub-hosted Ubuntu
runner for image build/test/publish; CUDA-compatible runtime base matching the
existing train image convention because `aitraf-core` resolves CUDA PyTorch
wheels through the workspace configuration.

**Project Type**: Monorepo Python ML/API project with `aitraf-api` serving,
`aitraf-core` reusable runtime processing, and `aitraf-train` offline train/eval
surfaces.

**Performance Goals**: Keep the API image scoped to serving needs by excluding
`aitraf-train`, notebooks, generated runs, and `storage/`; API image build should
complete using GitHub Actions cache and fail before publish if API tests fail.
No serving latency or model-throughput target is introduced by this deployment
planning feature.

**Constraints**: Preserve package-by-feature ownership. Reuse existing train
Dockerfile/workflow conventions where applicable. API image installs only
`aitraf-api` plus `aitraf-core` transitively, not `aitraf-train`. Copy `data/`
but not `storage/`. Do not commit secrets. Missing runtime env vars, model refs,
storage paths, model artifacts, or workflow permissions must fail explicitly.

**Scale/Scope**: One new `packages/aitraf-api/Dockerfile`, updates to the
existing `.github/workflows/publish-train-image.yml` image publishing workflow,
and API deployment documentation/validation notes. No hosting infrastructure,
rollout policy, model registry mutation, data generation, or API behavior change
is planned.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. The plan requires explicit failures for
  missing runtime env vars, missing model/storage inputs, failed API tests,
  Docker build failures, and GHCR permission/publish failures.
- **Package By Feature**: PASS. API deployment packaging stays under
  `packages/aitraf-api`; shared runtime dependency remains `packages/aitraf-core`;
  train image behavior remains in the existing train workflow path.
- **Function Decomposition**: PASS. No new business logic functions are expected.
  Workflow jobs are separated by responsibility: train publish, API test, and
  API publish.
- **Functional Programming And State**: PASS. The feature is deployment
  packaging, not new mutable business state. Runtime state remains environment
  and mounted storage supplied explicitly at container launch.
- **Reproducibility**: PASS. Dockerfile, workflow, pinned lockfile install,
  source-revision tags, image labels, copied `data/`, and documented validation
  commands provide reproducible build and verification inputs.

### Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. Contracts require API tests to gate API image
  publishing and runtime smoke checks to fail explicitly when env/model/storage
  inputs are missing.
- **Package By Feature**: PASS. The design adds API image packaging to
  `packages/aitraf-api` and extends the existing repository image workflow
  rather than adding a separate release subsystem.
- **Function Decomposition**: PASS. The workflow design keeps train and API jobs
  independent, with API test and API publish as distinct responsibilities.
- **Functional Programming And State**: PASS. No new hidden state or business
  mutation is introduced; deployment configuration remains explicit environment
  input.
- **Reproducibility**: PASS. Data model, contracts, and quickstart document image
  contents, external runtime inputs, tags, tests, build commands, and smoke
  validation requirements.

## Project Structure

### Documentation (this feature)

```text
specs/004-aitraf-api-deployment/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── docker-image.md
│   └── github-workflow.md
└── tasks.md
```

### Source Code (repository root)

```text
.github/
└── workflows/
    └── publish-train-image.yml

packages/
├── aitraf-api/
│   ├── Dockerfile
│   ├── README.md
│   ├── Taskfile.yml
│   ├── pyproject.toml
│   ├── src/aitraf_api/
│   └── tests/
├── aitraf-core/
│   ├── pyproject.toml
│   └── src/aitraf_core/
└── aitraf-train/
    └── Dockerfile

data/
├── labels.jsonl
└── manifests/
```

**Structure Decision**: Add the API Dockerfile inside `packages/aitraf-api`
because the deployment artifact belongs to the API surface. Keep the existing
workflow file and extend it to publish both images, preserving its GHCR
conventions. Copy only the package code needed for API serving (`aitraf-api` and
`aitraf-core`) plus root workspace metadata and `data/`; do not copy
`aitraf-train`, notebooks, runs, models, or `storage/`.

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
