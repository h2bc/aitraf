# Implementation Plan: Split Core And ML Core

**Branch**: `[006-split-core-ml-core]` | **Date**: 2026-07-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/006-split-core-ml-core/spec.md`

## Summary

Refactor the current `aitraf-core` distribution so lightweight consumers no
longer install the ML stack. Keep generic file-cache, JSON/JSONL, and shared S3
helpers in `aitraf-core`; create `aitraf-ml-core` with the `aitraf_ml_core` import namespace
for reusable tensor, model-loading, inference, preprocessing, sampling, and
video behavior; and move clip-download orchestration into `aitraf-train`.
Refactor API thumbnail generation and media presigning to use core S3 helpers.
Update all callers directly and delete old ML and clip
paths from `aitraf_core` without compatibility aliases.

## Technical Context

**Language/Version**: Python `>=3.10,<3.14`

**Primary Dependencies**: `aitraf-core` uses the Python standard library and Boto3.
`aitraf-ml-core` depends on `aitraf-core`, AV, Hugging Face Hub, Kornia, MLflow,
NumPy, Torch 2.9.1, TorchCodec 0.8.1, and Transformers. `aitraf-train` retains its
existing training stack and directly owns clip-download orchestration.

**Storage**: Generic JSON/JSONL file reads, file-cache control, and shared S3
client/URI/existence/presigning helpers remain in `aitraf-core`. Clip downloads
move to `aitraf-train`. Existing cache paths and external artifact locations do
not change.

**Testing**: Pytest package tests, clean isolated installation/import checks,
static source/import boundary checks, dependency-tree inspection, compileall,
and a representative existing ML workflow smoke test.

**Target Platform**: Linux CPU/GPU development, training, and offline workflow
environments; lightweight core must also install and import in a plain supported
Python environment without the ML runtime.

**Project Type**: Python monorepo with independently installable packages.

**Performance Goals**: No model-performance or throughput change. Installing
`aitraf-core` resolves zero ML/video/tracking dependencies beyond Boto3;
representative
ML workflow output remains identical for fixed inputs and configuration.

**Constraints**: Preserve required typed contracts and current cache/artifact
locations; enforce `aitraf-core ← aitraf-ml-core ← aitraf-train` dependency
direction where applicable; allow train to depend directly on core; avoid import
side effects, fallbacks, compatibility paths, and duplicate implementations.

**Scale/Scope**: Classify all 29 current source files under `aitraf-core`, update
all repository imports and package/container/workspace metadata, relocate three
existing core test modules to their owning packages, and add boundary tests.

## Constitution Check

*GATE: Passed before Phase 0 and re-checked after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. Missing dependencies, configuration,
  artifacts, files, and invalid input shapes keep explicit failures. No optional
  import fallback or alternate implementation is introduced.
- **Package By Feature**: PASS with justified new package. Generic helpers remain
  in core; reusable heavy ML runtime receives an explicit ML package; shared S3
  behavior stays in core for API and train, while clip orchestration moves to
  train. The new package is
  required to make dependency weight independently selectable.
- **Function Decomposition**: PASS. Mixed internal imports are rewritten to the
  new owner, while generic file helpers and train storage remain small,
  single-purpose modules.
- **Functional Programming And State**: PASS. Pure transformations retain
  explicit arguments and results. Filesystem, model, network, and framework
  state remains localized to cache, storage, loading, and processing boundaries.
- **Reproducibility**: PASS. Workspace manifests, lockfile, Docker inputs, test
  commands, fixed workflow configuration, and before/after output comparison
  make the migration reproducible.
- **No Legacy Compatibility**: PASS. All in-repository callers change in one
  migration; old ML/storage modules, aliases, and exports under `aitraf_core`
  are deleted.
- **Required Types Over Defensive Normalization**: PASS. Existing public
  contracts remain singular. Moving modules does not add alternate argument
  forms or coercion paths.

## Project Structure

### Documentation (this feature)

```text
specs/006-split-core-ml-core/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── package-boundaries.md
└── tasks.md
```

### Source Code (repository root)

```text
packages/
├── aitraf-core/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/aitraf_core/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── jsonl.py
│   └── tests/
│       ├── test_cache.py
│       ├── test_dependency_boundary.py
│       └── test_jsonl.py
├── aitraf-ml-core/
│   ├── pyproject.toml
│   ├── README.md
│   ├── src/aitraf_ml_core/
│   │   ├── __init__.py
│   │   ├── inference/
│   │   ├── loading/
│   │   ├── pre_processing/
│   │   ├── processing/
│   │   └── utils/
│   │       └── huggingface.py
│   └── tests/
│       └── test_mlflow_loading.py
└── aitraf-train/
    ├── pyproject.toml
    ├── Dockerfile
    ├── README.md
    ├── src/aitraf_train/
    │   ├── storage/
    │   │   ├── __init__.py
    │   │   ├── clips.py
    │   │   └── s3.py
    │   └── ... existing feature modules updated to new imports
    └── tests/
        ├── test_storage_clips.py
        └── test_storage_s3.py
```

**Structure Decision**: Use a distinct distribution and import namespace for
the ML runtime: `aitraf-ml-core` / `aitraf_ml_core`. This prevents namespace
package ambiguity and makes prohibited reverse imports searchable. Retain only
generic and shared S3 helpers in `aitraf-core`. Move only clip orchestration to
train and make API thumbnail/media code consume the core S3 surface.
Task-specific training, evaluation, metrics, tracking, configs, and scripts stay
in `aitraf-train`; API behavior remains unchanged.

### Existing Module Classification

| Current surface | Destination | Decision |
|-----------------|-------------|----------|
| `aitraf_core.cache` | `aitraf_core.cache` | Retain generic file-cache helper |
| `aitraf_core.utils.jsonl` | `aitraf_core.utils.jsonl` | Retain generic strict JSON readers |
| `aitraf_core.utils.huggingface` | `aitraf_ml_core.utils.huggingface` | Move model-hub naming helper with its sole ML consumer |
| `aitraf_core.storage.s3` | `aitraf_core.storage.s3` | Retain shared S3 primitives used by API and train |
| `aitraf_core.storage.clips` | `aitraf_train.storage.clips` | Move clip-download orchestration to its feature owner |
| `aitraf_core.inference` | `aitraf_ml_core.inference` | Move all inference modules and initializers |
| `aitraf_core.loading` | `aitraf_ml_core.loading` | Move all model loaders and initializers |
| `aitraf_core.pre_processing` | `aitraf_ml_core.pre_processing` | Move model preprocessing and cache-path behavior |
| `aitraf_core.processing` | `aitraf_ml_core.processing` | Move sampling, video, and model processing modules |

All package initializers within those trees follow their classified owner. Empty
task/model initializers move with their parent tree; no forwarding initializer
remains in core.

## Phase 0: Research

See [research.md](./research.md). Decisions cover package/namespace naming,
module classification, storage ownership, dependency enforcement, and validation
without relying on optional extras.

## Phase 1: Design And Contracts

See [data-model.md](./data-model.md),
[contracts/package-boundaries.md](./contracts/package-boundaries.md), and
[quickstart.md](./quickstart.md).

## Implementation Direction

1. Capture a complete source/import inventory and a deterministic representative
   workflow output before moving code.
2. Add the new workspace member and `aitraf-ml-core` package metadata with an
   explicit dependency on `aitraf-core`.
3. Reduce `aitraf-core` dependencies to Boto3 only and add isolated boundary tests that
   scan both declared and imported dependencies.
4. Move all classified ML modules into `aitraf_ml_core`, rewriting internal
   imports to use `aitraf_ml_core` while continuing to import generic cache
   behavior from `aitraf_core`.
5. Retain S3 primitives and tests in core, move clip orchestration into train,
   and update API thumbnail/media plus train consumers to use the shared core S3 path.
6. Update all train ML imports to `aitraf_ml_core`, add `aitraf-ml-core` to train
   dependencies, and retain direct `aitraf-core` only for genuine generic helper
   use.
7. Update root workspace metadata, lockfile, train Docker build inputs,
   compile/import commands, and documentation.
8. Delete the old core ML/storage trees and prove there are no stale source
   imports, aliases, or compatibility modules.
9. Run isolated package tests, dependency and import boundary checks, train tests,
   compile/import validation, and the representative before/after workflow
   comparison.

## Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. The contracts require direct failures and
  prohibit conditional legacy imports.
- **Package By Feature**: PASS. The data model gives every existing module one
  destination; storage follows its actual feature consumer.
- **Function Decomposition**: PASS. Mixed ownership is handled by explicit module
  boundaries rather than wrappers or copied functions.
- **Functional Programming And State**: PASS. No new shared mutable state is
  introduced by the package move.
- **Reproducibility**: PASS. The quickstart records clean-environment,
  dependency-tree, test, and workflow-equivalence validation.
- **No Legacy Compatibility**: PASS. The package contract explicitly makes old ML
  and storage import paths invalid.
- **Required Types Over Defensive Normalization**: PASS. Public surface movement
  preserves one required contract per function and forbids compatibility input
  variants.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| New top-level `aitraf-ml-core` package | Reusable heavy ML behavior must be installable independently from generic helpers | Keeping all code in core preserves the dependency problem; optional extras do not enforce imports or ownership; moving everything to train conflicts with the explicitly chosen reusable ML-core product boundary |
