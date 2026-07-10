# Tasks: Split Core And ML Core

**Input**: Design documents from `/specs/006-split-core-ml-core/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/package-boundaries.md, quickstart.md

**Validation**: Automated package and boundary tests plus a representative command-level ML workflow comparison are required by the specification.

**Organization**: Tasks are grouped by user story so lightweight core isolation, ML runtime migration, and durable boundary enforcement can be validated as explicit increments.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it changes different files and has no dependency on another incomplete task in the phase
- **[Story]**: Maps the task to User Story 1, 2, or 3
- Every task includes an exact file or directory path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Capture migration evidence and scaffold the new independently installable package.

- [X] T001 Record every current `packages/aitraf-core/src/aitraf_core/**/*.py` module, its in-repository importers, and its retain/move destination in `specs/006-split-core-ml-core/module-inventory.md`
- [X] T002 Capture the pre-refactor command, versioned fixture, configuration, model reference, seed if applicable, and output for one deterministic representative ML workflow in `specs/006-split-core-ml-core/validation/baseline.md`
- [X] T003 Create the `packages/aitraf-ml-core/src/aitraf_ml_core/`, `packages/aitraf-ml-core/tests/`, and package initializer scaffold described by `specs/006-split-core-ml-core/plan.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Wire the package graph and establish the direct migration boundary before moving behavior.

**⚠️ CRITICAL**: Complete this phase before user-story implementation.

- [X] T004 Add `aitraf-ml-core` as a root project dependency, workspace source, and workspace member in `pyproject.toml`
- [X] T005 Define the `aitraf-ml-core` distribution, `aitraf_ml_core` package discovery, supported Python range, explicit ML dependencies, and dependency on `aitraf-core` in `packages/aitraf-ml-core/pyproject.toml`
- [X] T006 Update `packages/aitraf-train/pyproject.toml` to depend explicitly on `aitraf-ml-core` while retaining direct `aitraf-core` only for generic helper imports
- [X] T007 Update `packages/aitraf-train/Dockerfile` to copy `packages/aitraf-ml-core` before the frozen `aitraf-train` installation
- [X] T008 Regenerate `uv.lock` from the changed workspace manifests and verify that all three distributions resolve from workspace sources

**Checkpoint**: The new distribution resolves and package moves can begin without temporary aliases.

---

## Phase 3: User Story 1 - Use Lightweight Shared Helpers Independently (Priority: P1) 🎯 MVP

**Goal**: Make `aitraf-core` a lightweight generic and shared-S3 helper package that installs and runs without the ML stack.

**Independent Test**: Install only `packages/aitraf-core` in a clean environment, import every public core module, run its tests, and verify that the resolved dependency set contains no third-party runtime package.

### Validation for User Story 1

- [X] T009 [P] [US1] Add strict cache behavior tests, including invalid path and callback behavior, in `packages/aitraf-core/tests/test_cache.py`
- [X] T010 [P] [US1] Add valid and invalid JSON/JSONL object-shape tests in `packages/aitraf-core/tests/test_jsonl.py`
- [X] T011 [P] [US1] Add manifest, prohibited-source-import, public-module-import, and clean dependency boundary tests that permit only Boto3 beyond the standard library in `packages/aitraf-core/tests/test_dependency_boundary.py`

### Implementation for User Story 1

- [X] T012 [US1] Remove all ML/video/tracking runtime dependencies from `packages/aitraf-core/pyproject.toml` while retaining Boto3 for shared S3 helpers
- [X] T013 [P] [US1] Keep only the generic public cache contract and exports in `packages/aitraf-core/src/aitraf_core/cache.py` and `packages/aitraf-core/src/aitraf_core/__init__.py`
- [X] T014 [P] [US1] Keep strict JSON/JSONL object readers as the sole utility exports in `packages/aitraf-core/src/aitraf_core/utils/jsonl.py` and `packages/aitraf-core/src/aitraf_core/utils/__init__.py`
- [X] T015 [US1] Rewrite `packages/aitraf-core/README.md` with the lightweight ownership rules, public helpers, explicit failure behavior, and isolated installation/test commands
- [X] T016 [US1] Run and record the isolated core install, Boto3-only direct dependency report, imports, and `packages/aitraf-core/tests` results in `specs/006-split-core-ml-core/validation/core-isolation.md`

**Checkpoint**: User Story 1 is complete when core works independently and contains no third-party runtime dependency or prohibited import.

---

## Phase 4: User Story 2 - Reuse ML Runtime Capabilities Explicitly (Priority: P2)

**Goal**: Move reusable heavy ML behavior to `aitraf-ml-core`, retain shared S3 primitives in core, move clip orchestration to train, and update all consumers directly.

**Independent Test**: Run ML-core tests and affected train tests through their new namespaces, confirm removed core paths are unavailable, and reproduce the baseline workflow output with the same inputs and configuration.

### Validation for User Story 2

- [X] T017 [P] [US2] Move and update MLflow loader tests from `packages/aitraf-core/tests/test_mlflow_loading.py` to `packages/aitraf-ml-core/tests/test_mlflow_loading.py` using the `aitraf_ml_core` namespace
- [X] T018 [P] [US2] Extend shared S3 client, URI, existence, and presigning tests for API and train use cases in `packages/aitraf-core/tests/test_storage_s3.py`
- [X] T019 [P] [US2] Move and update clip storage tests from `packages/aitraf-core/tests/test_storage_clips.py` to `packages/aitraf-train/tests/test_storage_clips.py` using `aitraf_train.storage`
- [X] T020 [P] [US2] Add explicit invalid-runtime failure coverage for migrated ML loaders in `packages/aitraf-ml-core/tests/test_mlflow_loading.py`

### Implementation for User Story 2

- [X] T021 [P] [US2] Move `packages/aitraf-core/src/aitraf_core/inference/` to `packages/aitraf-ml-core/src/aitraf_ml_core/inference/` and rewrite all internal imports to the new namespace
- [X] T022 [P] [US2] Move `packages/aitraf-core/src/aitraf_core/loading/` to `packages/aitraf-ml-core/src/aitraf_ml_core/loading/` and preserve its explicit model/device/artifact contracts
- [X] T023 [US2] Move `packages/aitraf-core/src/aitraf_core/pre_processing/` and `packages/aitraf-core/src/aitraf_core/utils/huggingface.py` to `packages/aitraf-ml-core/src/aitraf_ml_core/pre_processing/` and `packages/aitraf-ml-core/src/aitraf_ml_core/utils/huggingface.py`, importing cache behavior from `aitraf_core.cache`
- [X] T024 [US2] Move `packages/aitraf-core/src/aitraf_core/processing/` to `packages/aitraf-ml-core/src/aitraf_ml_core/processing/` and rewrite its inference and preprocessing imports to `aitraf_ml_core`
- [X] T025 [P] [US2] Retain shared S3 primitives in `packages/aitraf-core/src/aitraf_core/storage/s3.py` and move only `packages/aitraf-core/src/aitraf_core/storage/clips.py` to `packages/aitraf-train/src/aitraf_train/storage/clips.py`
- [X] T026 [US2] Update every ML runtime import under `packages/aitraf-train/src/aitraf_train/` to `aitraf_ml_core`, every clip import to `aitraf_train.storage.clips`, and every shared S3 import to `aitraf_core.storage.s3`
- [X] T027 [US2] Add `aitraf-core` to `packages/aitraf-api/pyproject.toml` and refactor `packages/aitraf-api/src/aitraf_api/features/demo_predictions/thumbnails.py` and `packages/aitraf-api/src/aitraf_api/features/demo_predictions/videos.py` to use core S3 client, URI, object-existence, and presigning helpers, updating `packages/aitraf-api/tests/features/demo_predictions/test_thumbnails.py` and related media tests
- [X] T028 [US2] Delete the moved ML, clip, and Hugging Face helper source paths and obsolete tests from `packages/aitraf-core/` without removing `aitraf_core.storage.s3` or leaving forwarding modules
- [X] T029 [P] [US2] Document ML runtime ownership, public module groups, dependencies, and validation in `packages/aitraf-ml-core/README.md`
- [X] T030 [P] [US2] Update package ownership, ML imports, storage ownership, and compile/import commands in `packages/aitraf-train/README.md`
- [X] T031 [US2] Run ML-core and train tests, reproduce the T002 workflow with identical inputs, compare outputs, and record commands and results in `specs/006-split-core-ml-core/validation/ml-equivalence.md`

**Checkpoint**: User Story 2 is complete when all heavy runtime callers use ML core, API thumbnails use core S3 helpers, clips are train-owned, old paths fail to import, and the representative output is equivalent.

---

## Phase 5: User Story 3 - Maintain Clear Dependency Direction (Priority: P3)

**Goal**: Make the new ownership discoverable and prevent future reverse imports or dependency creep automatically.

**Independent Test**: Run architecture checks that pass on the repository and fail against controlled prohibited dependency/import examples, then verify a maintainer can locate the ownership rule in repository documentation.

### Validation for User Story 3

- [X] T032 [P] [US3] Add ML-core prohibited reverse-import checks in `packages/aitraf-ml-core/tests/test_ml_core_dependency_boundary.py`
- [X] T033 [P] [US3] Add repository migration assertions for removed `aitraf_core` paths and stale consumer imports in `packages/aitraf-ml-core/tests/test_removed_core_paths.py`

### Implementation for User Story 3

- [X] T034 [US3] Enforce prohibited imports and dependencies in `packages/aitraf-core/tests/test_dependency_boundary.py` and `packages/aitraf-ml-core/tests/test_ml_core_dependency_boundary.py`
- [X] T035 [P] [US3] Update the package overview and ownership decision guide in `README.md` with `aitraf-core`, `aitraf-ml-core`, `aitraf-train`, and `aitraf-api` responsibilities
- [X] T036 [P] [US3] Update the durable package-by-feature boundary in `AGENTS.md` so reusable lightweight helpers and reusable ML runtime processing have distinct owners
- [X] T037 [US3] Reconcile every entry in `specs/006-split-core-ml-core/module-inventory.md` against the final source tree and mark its destination and stale-import validation complete

**Checkpoint**: User Story 3 is complete when automated checks enforce both dependency directions and the final inventory has no unresolved module.

---

## Phase 6: Polish & Cross-Cutting Validation

**Purpose**: Validate the complete direct migration and update shared repository surfaces.

- [X] T038 Update root package links, environment instructions, and package descriptions for the four-package layout in `README.md`
- [X] T039 Run the stale import search, isolated package suites, full train suite, compileall, and four-namespace import commands from `specs/006-split-core-ml-core/quickstart.md` and record results in `specs/006-split-core-ml-core/validation/final.md`
- [X] T040 Inspect `pyproject.toml`, `packages/aitraf-core/pyproject.toml`, `packages/aitraf-ml-core/pyproject.toml`, `packages/aitraf-train/pyproject.toml`, and `uv.lock` to confirm the final dependency direction and absence of obsolete workspace entries
- [X] T041 Search `packages/`, `README.md`, `AGENTS.md`, and active `specs/006-split-core-ml-core/` artifacts for duplicate implementations, compatibility aliases, stale runtime imports, alternate input-shape normalization, and obsolete commands, then remove any findings in their owning files

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)** starts immediately; T001 and T002 establish evidence before source moves, and T003 scaffolds the destination.
- **Foundational (Phase 2)** depends on Setup and blocks all user stories because workspace resolution must recognize the destination package.
- **User Story 1 (Phase 3)** depends on Foundational and delivers the independently installable lightweight-core MVP.
- **User Story 2 (Phase 4)** depends on Foundational and uses the retained US1 cache helper; execute it after US1 for a single-worker migration.
- **User Story 3 (Phase 5)** depends on the final paths created by US1 and US2.
- **Polish (Phase 6)** depends on all selected user stories.

### User Story Dependencies

```text
Setup → Foundational → US1 (lightweight core)
                   └→ US2 (ML core migration; consumes retained core cache)
US1 + US2 → US3 (durable enforcement) → Polish
```

- **US1** is independently testable after Foundational.
- **US2** is independently testable after Foundational when the retained core cache contract exists; in normal execution, complete US1 first.
- **US3** intentionally validates the completed package locations and therefore follows US1 and US2.

### Parallel Opportunities

- T001 and T002 can run in parallel; T003 can proceed once the intended destination is confirmed.
- US1 test tasks T009–T011 can run in parallel, followed by parallel implementation tasks T013–T014.
- US2 validation tasks T017–T020 can be prepared in parallel.
- US2 moves T021, T022, and T025 touch separate trees and can run in parallel; T023 and T024 then reconcile cross-tree imports.
- Documentation tasks T029 and T030 can run in parallel after destination APIs stabilize.
- US3 validation tasks T032–T033 and documentation tasks T035–T036 are parallel file groups.

## Parallel Examples

### User Story 1

```text
Task T009: Add cache tests in packages/aitraf-core/tests/test_cache.py
Task T010: Add JSON tests in packages/aitraf-core/tests/test_jsonl.py
Task T011: Add boundary tests in packages/aitraf-core/tests/test_dependency_boundary.py
```

### User Story 2

```text
Task T021: Move inference into packages/aitraf-ml-core/src/aitraf_ml_core/inference/
Task T022: Move loading into packages/aitraf-ml-core/src/aitraf_ml_core/loading/
Task T025: Retain core S3 primitives and move clips into packages/aitraf-train/src/aitraf_train/storage/
```

### User Story 3

```text
Task T032: Add ML-core dependency checks in packages/aitraf-ml-core/tests/test_dependency_boundary.py
Task T033: Add removed-path checks in packages/aitraf-ml-core/tests/test_removed_core_paths.py
Task T035: Update the ownership guide in README.md
Task T036: Update durable ownership rules in AGENTS.md
```

## Implementation Strategy

### MVP First

1. Complete Setup and Foundational phases.
2. Complete User Story 1.
3. Stop and prove that core installs, imports, and tests without third-party runtime dependencies.
4. Continue to the ML migration only after the lightweight boundary is demonstrated.

### Incremental Delivery

1. Establish manifests and destination package.
2. Deliver lightweight core isolation.
3. Move ML and clip trees atomically, retain shared S3 primitives, and migrate API thumbnail/media consumers.
4. Add durable repository-wide enforcement and documentation.
5. Run the complete quickstart and baseline-equivalence gate.

## Notes

- Do not introduce `try/except ImportError`, forwarding modules, re-exports, or optional extras for old paths.
- Preserve unrelated worktree changes, especially the existing API feature edits.
- Use filesystem moves carefully and review every resulting internal import; the implementation workflow must still use patch-based edits for authored changes.
- A task is complete only when its exact owning files and associated validation evidence are updated.
