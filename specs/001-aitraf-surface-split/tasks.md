# Tasks: AITRAF Surface Split

**Input**: Design documents from `/specs/001-aitraf-surface-split/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Validation**: Validation tasks are REQUIRED. This repo currently has no retained automated test suite for this refactor; use command-level smoke validation for pipeline behavior and package boundaries.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Workspace packages**: `packages/aitraf-core/`, `packages/aitraf-train/`, `packages/aitraf-api/`
- **Repository entrypoints**: `Taskfile.yml`, `README.md`, root `pyproject.toml`
- **Validation**: command-level smoke checks documented in `README.md` and package READMEs
- **Analysis only**: `notebooks/` (do not leave production-only behavior here)

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the monorepo package scaffolding and workspace install surface

- [X] T001 Create the package directory skeleton and package entry files in `packages/aitraf-core/`, `packages/aitraf-train/`, and `packages/aitraf-api/`
- [X] T002 Add package-local manifests in `packages/aitraf-core/pyproject.toml`, `packages/aitraf-train/pyproject.toml`, and `packages/aitraf-api/pyproject.toml`
- [X] T003 [P] Add package-local ownership docs in `packages/aitraf-core/README.md`, `packages/aitraf-train/README.md`, and `packages/aitraf-api/README.md`
- [X] T004 Update workspace package installation wiring in `pyproject.toml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared package boundaries and package wiring before story work

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create core package scaffolding for existing reusable modules in `packages/aitraf-core/src/aitraf_core/__init__.py`, `packages/aitraf-core/src/aitraf_core/processing/__init__.py`, and `packages/aitraf-core/src/aitraf_core/utils/__init__.py`
- [X] T006 [P] Create train package scaffolding for workflow modules in `packages/aitraf-train/src/aitraf_train/__init__.py`, `packages/aitraf-train/src/aitraf_train/logging.py`, and `packages/aitraf-train/src/aitraf_train/prepare.py`
- [X] T007 [P] Create empty API package scaffolding in `packages/aitraf-api/src/aitraf_api/__init__.py`
- [X] T008 Move the Hydra config tree into `packages/aitraf-train/configs/` while preserving the current file layout from `configs/`
- [X] T009 Move the offline command entrypoints into `packages/aitraf-train/scripts/` while preserving the current file layout from `scripts/`
- [X] T010 Update root command wrappers in `Taskfile.yml` and root `pyproject.toml` to execute the train package entrypoints
- [X] T011 Add explicit package-boundary metadata in `packages/aitraf-core/src/aitraf_core/__init__.py`, `packages/aitraf-train/src/aitraf_train/__init__.py`, and `packages/aitraf-api/src/aitraf_api/__init__.py`
- [X] T012 Document workspace-level package rules and reproducibility expectations in `README.md`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Separate Shared And Training Responsibilities (Priority: P1) 🎯 MVP

**Goal**: Make the repo boundaries explicit so shared processing has one owner and training orchestration has another

**Independent Test**: A maintainer can inspect the package layout and command surface, then verify that `aitraf-core`, `aitraf-train`, and `aitraf-api` each have one documented responsibility and that train commands still resolve through the train package

### Validation for User Story 1 ⚠️

- [X] T013 [P] [US1] Add package-ownership smoke validation commands to `README.md`
- [X] T014 [P] [US1] Add command-continuity smoke validation commands to `README.md`

### Implementation for User Story 1

- [X] T015 [P] [US1] Move workflow-oriented modules into `packages/aitraf-train/src/aitraf_train/data_ops/`, `packages/aitraf-train/src/aitraf_train/label_ops/`, and `packages/aitraf-train/src/aitraf_train/datasets/`
- [X] T016 [P] [US1] Move training support modules into `packages/aitraf-train/src/aitraf_train/models/`, `packages/aitraf-train/src/aitraf_train/metrics/`, and `packages/aitraf-train/src/aitraf_train/tracking/`
- [X] T017 [US1] Move task implementations into `packages/aitraf-train/src/aitraf_train/tasks/` and update train-side imports there
- [X] T018 [US1] Update the relocated scripts in `packages/aitraf-train/scripts/` and root `Taskfile.yml` to import only from `aitraf_train`
- [X] T019 [US1] Write boundary and ownership documentation in `README.md`, `packages/aitraf-core/README.md`, and `packages/aitraf-train/README.md`
- [X] T020 [US1] Keep root package, config, and script ownership out of the workspace root

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Reuse Shared Processing Across Offline And Online Flows (Priority: P2)

**Goal**: Reuse one shared runtime-processing surface for train-side workflows now, while keeping the core surface suitable for a future API consumer

**Independent Test**: A model engineer can inspect and smoke-test train-side consumers, then confirm they import shared runtime-processing logic from `aitraf-core` instead of carrying separate implementations

### Validation for User Story 2 ⚠️

- [X] T021 [P] [US2] Add shared-core import validation commands to `README.md`
- [X] T022 [P] [US2] Add shared-runtime reuse smoke validation guidance for train-side consumers to `README.md`

### Implementation for User Story 2

- [X] T023 [P] [US2] Move reusable processing modules into `packages/aitraf-core/src/aitraf_core/processing/video.py` and `packages/aitraf-core/src/aitraf_core/processing/utils.py`
- [X] T024 [P] [US2] Move model-specific processing modules into `packages/aitraf-core/src/aitraf_core/processing/models/pose_tcn.py`, `packages/aitraf-core/src/aitraf_core/processing/models/video_mae.py`, and `packages/aitraf-core/src/aitraf_core/processing/models/video_mae_temporal_fusion.py`
- [X] T025 [P] [US2] Move shared utility modules into `packages/aitraf-core/src/aitraf_core/utils/huggingface.py` and `packages/aitraf-core/src/aitraf_core/utils/video_utils.py`
- [X] T026 [US2] Refactor the train-side extraction consumers in `packages/aitraf-train/src/aitraf_train/data_ops/pose_and_bbox_extraction.py` and `packages/aitraf-train/src/aitraf_train/data_ops/video_mae_feature_extraction.py` to compose `aitraf_core` helpers instead of owning reusable processing logic
- [X] T027 [US2] Refactor train task and dataset imports to use `aitraf_core` from `packages/aitraf-train/src/aitraf_train/tasks/` and `packages/aitraf-train/src/aitraf_train/datasets/pose_tcn.py`
- [X] T028 [US2] Document shared runtime artifact ownership and reproducibility rules in `packages/aitraf-core/README.md`, `packages/aitraf-train/README.md`, and `README.md`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Reserve A Future Inference Package (Priority: P3)

**Goal**: Add an empty `aitraf-api` package placeholder so future inference work has an explicit ownership location without implementing API behavior in this phase

**Independent Test**: An API developer can inspect the package directory and README, then confirm the package intentionally contains no runtime operations, schemas, app wiring, or smoke command yet

### Validation for User Story 3 ⚠️

- [X] T029 [P] [US3] Add empty API package validation command guidance to `README.md`
- [X] T030 [P] [US3] Add package-boundary validation guidance that `packages/aitraf-api/src/aitraf_api/` contains no runtime API implementation beyond `__init__.py`

### Implementation for User Story 3

- [X] T031 [US3] Keep `packages/aitraf-api/src/aitraf_api/__init__.py` intentionally empty except for package metadata or a short module docstring
- [X] T032 [US3] Document that `packages/aitraf-api/README.md` is a placeholder for future trick recognition and trick assessment work and has no runtime behavior yet
- [X] T033 [US3] Update `README.md` to describe `aitraf-api` as an empty reserved package for future inference API work

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T034 [P] Update package command examples in `README.md`, `packages/aitraf-core/README.md`, `packages/aitraf-train/README.md`, and `packages/aitraf-api/README.md`
- [X] T035 Finalize package/config/script ownership boundaries
- [X] T036 Run and document the final smoke-validation command set in `README.md` and `packages/aitraf-train/README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel if desired
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - establishes the package and command boundaries
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - builds on the package boundaries but should remain independently testable once shared runtime processing is consumed from core
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - reserves the API package without depending on runtime API implementation

### Within Each User Story

- Validation tasks MUST be defined before implementation
- Package scaffolding and imports before deeper module migration
- Shared core helpers before train integration that consumes them
- Command and docs updates after code ownership is in place
- Story complete before moving to the next priority if working sequentially

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel within Phase 2
- Once Foundational completes, User Stories 1-3 can be staffed in parallel
- Validation tasks within a user story marked [P] can run in parallel
- Module migration tasks touching different package trees marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch validation tasks for User Story 1 together:
Task: "Add package-ownership validation commands to README.md"
Task: "Add command-continuity smoke validation commands to README.md"

# Launch train-owned module moves for User Story 1 together:
Task: "Move workflow-oriented modules into packages/aitraf-train/src/aitraf_train/data_ops/, label_ops/, and datasets/"
Task: "Move training support modules into packages/aitraf-train/src/aitraf_train/models/, metrics/, and tracking/"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Confirm package ownership and train command continuity
5. Demo the new package boundaries before migrating shared processing

### Incremental Delivery

1. Complete Setup + Foundational → workspace and package boundaries ready
2. Add User Story 1 → validate package ownership and command continuity
3. Add User Story 2 → validate shared-core reuse in train workflows
4. Add User Story 3 → validate the empty API package placeholder
5. Finish Polish → remove shims and finalize docs

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Merge once each story passes its independent validation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Prefer preserving the current repo structure over renaming modules without a clear boundary benefit
- Prefer extending existing architecture over adding new parallel structure
- Ensure invalid states fail explicitly instead of relying on fallback behavior
- Keep `data_ops` in `aitraf-train`; only reusable processing primitives move to `aitraf-core`
