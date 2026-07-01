# Tasks: Demo Clip Download

**Input**: Design documents from `/specs/005-demo-clip-download/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Validation**: Validation tasks are REQUIRED. Add automated tests where practical, and always add command-level smoke validation for pipeline behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **API package**: `packages/aitraf-api/src/aitraf_api/`, `packages/aitraf-api/tests/`
- **Core runtime package**: `packages/aitraf-core/src/aitraf_core/`, `packages/aitraf-core/tests/`
- **Train package configs**: `packages/aitraf-train/configs/`
- **Train package entrypoints**: `packages/aitraf-train/scripts/`
- **Train package code**: `packages/aitraf-train/src/aitraf_train/`
- **Validation**: package-local `tests/`, `tests/unit/`, `tests/integration/`, or smoke commands documented in `specs/005-demo-clip-download/quickstart.md`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare package metadata and identify current S3/download call sites before changing behavior.

- [X] T001 Inspect current S3 helper call sites in `packages/aitraf-train/src/aitraf_train/` and document import targets in `specs/005-demo-clip-download/research.md`
- [X] T002 Add `boto3` dependency to `packages/aitraf-core/pyproject.toml` and update `uv.lock`
- [X] T003 Remove the direct `boto3` dependency from `packages/aitraf-api/pyproject.toml` if no API module imports boto3 directly, and update `uv.lock`
- [X] T004 [P] Create the shared storage package scaffold in `packages/aitraf-core/src/aitraf_core/storage/__init__.py`
- [X] T005 [P] Create the core storage test scaffold in `packages/aitraf-core/tests/test_storage_s3.py` and `packages/aitraf-core/tests/test_storage_clips.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the shared core storage/download primitive that API and train surfaces will reuse.

**CRITICAL**: No user story work can begin until this phase is complete.

- [X] T006 Move generic S3 settings, client construction, URI parsing, presigning, key iteration, and object-exists helpers from `packages/aitraf-train/src/aitraf_train/utils/s3_utils.py` to `packages/aitraf-core/src/aitraf_core/storage/s3.py`
- [X] T007 Update train imports that use S3 helpers to import from `aitraf_core.storage.s3` in `packages/aitraf-train/src/aitraf_train/preparation/`, `packages/aitraf-train/src/aitraf_train/metrics/`, and `packages/aitraf-train/src/aitraf_train/utils/`
- [X] T008 Add `ClipDownloadRequest` and `ClipDownloadResult` data structures in `packages/aitraf-core/src/aitraf_core/storage/clips.py`
- [X] T009 Implement single-clip download behavior with skip-existing and force semantics in `packages/aitraf-core/src/aitraf_core/storage/clips.py`
- [X] T010 Implement multi-clip download behavior that fails on any unresolved download in `packages/aitraf-core/src/aitraf_core/storage/clips.py`
- [X] T011 Export shared storage helpers from `packages/aitraf-core/src/aitraf_core/storage/__init__.py`
- [X] T012 [P] Add S3 settings and URI parsing tests in `packages/aitraf-core/tests/test_storage_s3.py`
- [X] T013 [P] Add clip download skip-existing, force, destination creation, and failure propagation tests in `packages/aitraf-core/tests/test_storage_clips.py`
- [X] T014 Run `uv run --package aitraf-core pytest packages/aitraf-core/tests` and fix failures in `packages/aitraf-core/src/aitraf_core/storage/`

**Checkpoint**: Shared core storage/download behavior is available and tested.

---

## Phase 3: User Story 1 - Demo API Starts With Required Clips (Priority: P1) - MVP

**Goal**: API startup can explicitly hydrate only selected demo clips into runtime storage.

**Independent Test**: Start the API with empty writable clip storage, valid manifests, and mocked or real object storage credentials; selected demo clips are present before demo inference is considered ready.

### Validation for User Story 1

- [X] T015 [P] [US1] Add API tests for building demo clip download requests from selected manifest rows in `packages/aitraf-api/tests/features/demo_videos/test_download.py`
- [X] T016 [P] [US1] Add API app-factory tests for disabled/enabled startup hydration with a mocked core downloader in `packages/aitraf-api/tests/test_app.py`
- [X] T017 [P] [US1] Add API tests for missing `s3_path`, invalid source URI, and failed download propagation in `packages/aitraf-api/tests/features/demo_videos/test_download.py`

### Implementation for User Story 1

- [X] T018 [US1] Add explicit demo clip download settings to `packages/aitraf-api/src/aitraf_api/config.py`
- [X] T019 [US1] Implement selected demo clip request construction in `packages/aitraf-api/src/aitraf_api/features/demo_videos/download.py`
- [X] T020 [US1] Implement API demo clip hydration by calling core clip downloads from `packages/aitraf-api/src/aitraf_api/features/demo_videos/download.py`
- [X] T021 [US1] Wire opt-in demo clip hydration into API startup before app readiness in `packages/aitraf-api/src/aitraf_api/app.py`
- [X] T022 [US1] Update `packages/aitraf-api/README.md` with the opt-in runtime demo clip download setting and required AWS-compatible environment variables
- [X] T023 [US1] Run `uv run --package aitraf-api --extra dev pytest packages/aitraf-api/tests` and fix failures in `packages/aitraf-api/src/aitraf_api/`

**Checkpoint**: User Story 1 is independently functional and validates the MVP runtime demo clip hydration path.

---

## Phase 4: User Story 2 - Shared Download Behavior Across API And Training (Priority: P2)

**Goal**: Existing train clip downloads reuse the shared core downloader while preserving train data-op behavior.

**Independent Test**: A representative train labels input produces the same destination clip paths and skip/force behavior as before, using the shared core downloader.

### Validation for User Story 2

- [X] T024 [P] [US2] Add train tests for labels-to-core-download-request conversion in `packages/aitraf-train/tests/test_download_clips.py`
- [X] T025 [P] [US2] Add train tests for preserving destination paths and force behavior in `packages/aitraf-train/tests/test_download_clips.py`

### Implementation for User Story 2

- [X] T026 [US2] Refactor `packages/aitraf-train/src/aitraf_train/preparation/data_ops/download_clips.py` to construct `ClipDownloadRequest` values and call `aitraf_core.storage.clips.download_clips`
- [X] T027 [US2] Preserve train progress logging and summary counts in `packages/aitraf-train/src/aitraf_train/preparation/data_ops/download_clips.py`
- [X] T028 [US2] Remove obsolete S3 helper definitions from `packages/aitraf-train/src/aitraf_train/utils/s3_utils.py`
- [X] T029 [US2] Run `uv run --package aitraf-train pytest packages/aitraf-train/tests` if train tests are available, or run the new train download tests directly
- [X] T030 [US2] Run `task train:data_ops -- download_clips.enabled=true` with a representative configuration or document why full S3 smoke validation is unavailable in `specs/005-demo-clip-download/quickstart.md`

**Checkpoint**: User Stories 1 and 2 both use the shared core clip download path without API depending on train.

---

## Phase 5: User Story 3 - Restore Image Build Independence From Local Clips (Priority: P3)

**Goal**: Revert the current API Docker build-time clip dependency so API image builds and GHCR publishing no longer require local clip files or an `aitraf_clips` build context.

**Independent Test**: Remove the current `aitraf_clips` Docker build dependency, build the API image in an environment with no `storage/data/clips` directory, and verify the build succeeds without downloading or bundling clips.

### Validation for User Story 3

- [X] T031 [P] [US3] Add or update Docker validation notes in `specs/005-demo-clip-download/quickstart.md`
- [X] T032 [P] [US3] Inspect `.dockerignore` and document that full `storage/` remains excluded in `packages/aitraf-api/README.md`

### Implementation for User Story 3

- [X] T033 [US3] Remove the demo clip build stage and `aitraf_clips` build-context requirement from `packages/aitraf-api/Dockerfile`
- [X] T034 [US3] Remove `--build-context aitraf_clips=storage/data/clips` from `packages/aitraf-api/Taskfile.yml`
- [X] T035 [US3] Remove `--build-context aitraf_clips=storage/data/clips` from the root `Taskfile.yml`
- [X] T036 [US3] Remove `build-contexts: aitraf_clips=storage/data/clips` and any empty clip context preparation step from `.github/workflows/publish-docker-images.yml`
- [X] T037 [US3] Update `packages/aitraf-api/README.md`, `specs/004-aitraf-api-deployment/quickstart.md`, and `specs/004-aitraf-api-deployment/contracts/docker-image.md` to state that the API image builds without clips and runtime hydration handles demo clips
- [X] T038 [US3] Run `docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .` and verify it does not require local clips
- [X] T039 [US3] Run the built `aitraf-api:local` container with the project env file and a temporary writable storage mount to verify selected demo clips download inside the container

**Checkpoint**: API image build and workflow are independent from local clip storage.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation alignment, and drift cleanup.

- [X] T040 [P] Update `specs/005-demo-clip-download/quickstart.md` with final command outputs or any unavailable smoke-validation notes
- [X] T041 [P] Update `specs/005-demo-clip-download/contracts/core-clip-download.md`, `specs/005-demo-clip-download/contracts/api-runtime-demo-download.md`, and `specs/005-demo-clip-download/contracts/train-data-ops-reuse.md` if implementation changes contract details
- [X] T042 Run `uv run ruff check packages/aitraf-core/src/aitraf_core packages/aitraf-core/tests packages/aitraf-api/src/aitraf_api packages/aitraf-api/tests packages/aitraf-train/src/aitraf_train`
- [X] T043 Run `uv run --package aitraf-core pytest packages/aitraf-core/tests` and `uv run --package aitraf-api --extra dev pytest packages/aitraf-api/tests`
- [X] T044 Verify `rg -n "aitraf_train" packages/aitraf-api/src packages/aitraf-core/src` returns no API/core dependency on train code
- [X] T045 Verify `git diff --stat` and inspect changed files for secrets, generated clips, `.env*`, or unintended storage artifacts

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion and should be validated after US1 if shared API usage changed
- **User Story 3 (Phase 5)**: Depends on US1 design decisions for runtime hydration and can be implemented after Docker cleanup scope is clear
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Requires shared core downloader from Phase 2; no dependency on US2 or US3
- **User Story 2 (P2)**: Requires shared core downloader from Phase 2; no dependency on US3
- **User Story 3 (P3)**: Can be completed after runtime hydration path is defined; does not change core downloader behavior

### Within Each User Story

- Validation tasks come before implementation tasks
- Pure request-construction helpers before startup or pipeline wiring
- Core downloader calls before package-specific orchestration
- Documentation updates after command/config behavior is stable

### Parallel Opportunities

- T004 and T005 can run in parallel after T002/T003 decisions are known
- T012 and T013 can run in parallel after T006-T011 interfaces are defined
- T015, T016, and T017 can run in parallel before US1 implementation
- T024 and T025 can run in parallel before train refactor implementation
- T031 and T032 can run in parallel before Docker cleanup
- T040 and T041 can run in parallel during polish

---

## Parallel Example: User Story 1

```bash
# Launch API validation tasks together:
Task: "T015 Add API tests for building demo clip download requests in packages/aitraf-api/tests/features/demo_videos/test_download.py"
Task: "T016 Add API tests for startup hydration behavior in packages/aitraf-api/tests/test_app.py"
Task: "T017 Add API tests for failure propagation in packages/aitraf-api/tests/features/demo_videos/test_download.py"
```

---

## Parallel Example: User Story 2

```bash
# Launch train validation tasks together:
Task: "T024 Add train tests for labels-to-core-download-request conversion in packages/aitraf-train/tests/test_download_clips.py"
Task: "T025 Add train tests for preserving destination paths and force behavior in packages/aitraf-train/tests/test_download_clips.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch Docker documentation checks together:
Task: "T031 Update Docker validation notes in specs/005-demo-clip-download/quickstart.md"
Task: "T032 Inspect .dockerignore and document full storage exclusion in packages/aitraf-api/README.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational shared core downloader
3. Complete Phase 3: API runtime demo clip hydration
4. Stop and validate with API tests plus a runtime hydration smoke scenario
5. Deploy/demo with runtime demo clip download explicitly enabled

### Incremental Delivery

1. Shared downloader foundation
2. API runtime hydration MVP
3. Train data-op reuse
4. Docker/workflow cleanup so published images stay clip-independent
5. Polish, docs, and full validation

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup and Foundational tasks together
2. Developer A implements US1 API hydration
3. Developer B implements US2 train reuse after core interfaces stabilize
4. Developer C handles US3 Docker/workflow cleanup after API runtime behavior is settled

## Notes

- Keep low-level S3 and clip download mechanics in `aitraf-core`; do not put storage I/O under model pre-processing modules.
- Keep API demo selection and startup policy in `aitraf-api`; do not import `aitraf-train` from API code.
- Keep Hydra, labels parsing, and data-ops orchestration in `aitraf-train`.
- Runtime demo clip download must be opt-in and must fail explicitly when required inputs are missing.
- Do not commit generated clips, secrets, `.env*`, or storage artifacts.
