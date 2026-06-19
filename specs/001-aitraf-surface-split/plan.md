# Implementation Plan: AITRAF Surface Split

**Branch**: `not set` | **Date**: 2026-06-19 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-aitraf-surface-split/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Refactor the repository into a single-repo, multi-package design under
`packages/`: `aitraf-core` for reusable runtime processing and model-input
logic, `aitraf-train` for Hydra-driven data preparation, training, evaluation,
and tracking workflows, and `aitraf-api` as an empty reserved package for the
future inference surface. Preserve the existing repository structure as much as
possible by moving current reusable
`processing/` and selected shared `utils/` code into core, keeping Hydra configs
and offline scripts in train, keeping MLflow- and run-oriented behavior in
train, keeping `data/`, `storage/`, and `notebooks/` as repo-level shared
assets, and preventing API runtime behavior from being added until live
inference is implemented.

## Technical Context

**Language/Version**: Python 3.10+ (repo currently supports `>=3.10,<3.14`)

**Primary Dependencies**: PyTorch, Transformers, Lightning, Hydra, MLflow, pandas, scikit-learn

**Storage**: Repo-local `data/` manifests, generated `storage/` artifacts, MLflow tracking, S3-backed inputs/artifacts as applicable

**Testing**: Current baseline is command-level smoke validation because the repo does not retain an established test suite for this refactor

**Target Platform**: Linux GPU environment, dev container / Docker runtime, CUDA-capable training hosts

**Project Type**: Monorepo Python ML pipeline with shared runtime and placeholder API surfaces

**Performance Goals**: No material regression in representative prepare/train/eval smoke runs; shared clip processing for frames, poses, and features remains usable on existing CUDA-capable dev hosts; API remains an empty placeholder with no runtime execution path in this phase

**Constraints**: Use one dev container/workspace for all packages; move Hydra `configs/` and offline `scripts/` under `aitraf-train`; keep `data/`, `storage/`, and `notebooks/` at repo root; avoid excessive fallbacks; keep MLflow/run ownership in train; keep shared runtime outputs and registry-managed artifacts explicitly separated

**Scale/Scope**: Re-home current `src/aitraf/` responsibilities into `packages/aitraf-core`, `packages/aitraf-train`, and `packages/aitraf-api` with minimal structural churn, add per-package `pyproject.toml` and `README.md` files, update operator docs and dependency ownership, and preserve current trick classification plus ordinal score workflows as the first validated consumers

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. Missing package wiring and missing model
  assets will raise explicit errors rather than silently rerouting to legacy
  code. API runtime execution is not provided in this phase.
- **Architecture Fit**: PASS WITH JUSTIFICATION. The feature introduces three
  explicit package surfaces under `packages/` because the product requirement is
  a repo-visible split with separate package ownership. This is broader than a
  pure in-place `src/` refactor, but it still preserves one repo, one dev
  container, and shared repo-level assets instead of fragmenting workflows into
  multiple repositories.
- **Function Decomposition**: PASS. The design centers on small shared helpers for
  frame loading, pose extraction orchestration, feature generation, model asset
  resolution, and operation-specific adapters instead of task-specific copies.
- **State And Mutation**: PASS. Stateful concerns remain localized to training
  orchestration, external model clients, and future API request handling; shared
  runtime helpers stay transformation-oriented.
- **Reproducibility**: PASS. Existing manifests, storage paths, and MLflow
  outputs remain repo-managed, while the plan adds explicit ownership rules for
  reusable runtime artifacts versus registry-managed artifacts and makes package
  dependency ownership reviewable in package manifests.

### Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. The API package remains empty until runtime
  inference is delivered, so there is no placeholder runtime fallback path.
- **Architecture Fit**: PASS. `packages/aitraf-train` owns Hydra configs and
  offline scripts, while the repo root remains a shared workspace with optional
  convenience entrypoints only. New packages represent real ownership, not a
  second repo or disconnected workflow layer.
- **Function Decomposition**: PASS. Shared capabilities are grouped by domain in
  core, while train and api remain thin orchestrators over those helpers.
- **State And Mutation**: PASS. Mutable pipeline state stays in training runs,
  request-scoped API handlers, and storage/tracking boundaries only.
- **Reproducibility**: PASS. The quickstart and contracts require smoke coverage
  for command continuity, shared-core reuse, explicit placeholder behavior, and
  reviewable package dependency ownership.

## Project Structure

### Documentation (this feature)

```text
specs/001-aitraf-surface-split/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── package-surfaces.md
└── tasks.md
```

### Source Code (repository root)

```text
packages/
├── aitraf-core/
│   ├── README.md
│   ├── pyproject.toml
│   └── src/aitraf_core/
│       ├── processing/
│       │   ├── models/
│       │   ├── utils.py
│       │   └── video.py
│       └── utils/
│           ├── huggingface.py
│           └── video_utils.py
├── aitraf-train/
│   ├── README.md
│   ├── pyproject.toml
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── data_ops.yaml
│   │   ├── eval.yaml
│   │   ├── label_ops.yaml
│   │   ├── model/
│   │   ├── prepare.yaml
│   │   ├── task/
│   │   ├── train.yaml
│   │   └── train_eval.yaml
│   ├── scripts/
│   │   ├── data_ops_pipeline.py
│   │   ├── eval.py
│   │   ├── label_ops_pipeline.py
│   │   ├── prepare.py
│   │   ├── train.py
│   │   └── train_eval.py
│   └── src/aitraf_train/
│       ├── data_ops/
│       ├── cli/
│       ├── datasets/
│       ├── label_ops/
│       ├── metrics/
│       ├── models/
│       ├── tasks/
│       ├── tracking/
│       ├── logging.py
│       └── prepare.py
└── aitraf-api/
    ├── README.md
    ├── pyproject.toml
    └── src/aitraf_api/
        └── __init__.py

Taskfile.yml
README.md
pyproject.toml
data/
storage/
notebooks/
```

**Structure Decision**: Use a real multi-package monorepo under `packages/`,
but preserve the current internal structure wherever possible. `aitraf-core`
should look like the existing reusable surface: primarily `processing/` and the
small subset of `utils/` that is genuinely shared by both training support and
future API inference. Concretely, that means reusing the existing
`processing/video.py`, `processing/utils.py`, `processing/models/*`, and the
small set of shared utility modules instead of inventing new top-level folders.
`aitraf-train` owns the current workflow-oriented modules: `data_ops`,
`label_ops`, `datasets`, `models`, `tasks`, `metrics`, `tracking`, Hydra
configs, and offline scripts. `aitraf-api` owns the future-serving seam, but it
is intentionally empty in this implementation.
`data/`, `storage/`, and `notebooks/` stay at repo root because they are shared
workspace assets rather than package-owned code. The root may keep thin
convenience wrappers such as `Taskfile.yml`, but package ownership must be
explicit and package-local READMEs must explain each boundary. Temporary
compatibility wrappers, if used during migration, must stay thin, explicit, and
removable before the feature is considered complete.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| New first-class packages under `packages/` | The feature requires explicit install/dependency and documentation boundaries for core/train/api inside one repo | Keeping a single `src/` tree with multiple namespaces would improve code separation but would not make package ownership, dependency manifests, and per-surface documentation explicit enough |
