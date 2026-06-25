# Tasks: AITRAF API Deployment

**Input**: Design documents from `/specs/004-aitraf-api-deployment/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Validation**: Validation tasks are REQUIRED. API tests must gate API publishing, and command-level Docker/workflow validation is required before implementation is complete.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2)
- Include exact file paths in descriptions

## Path Conventions

- **API package**: `packages/aitraf-api/`, `packages/aitraf-api/src/aitraf_api/`, `packages/aitraf-api/tests/`
- **Core runtime package**: `packages/aitraf-core/src/aitraf_core/`
- **Workflow**: `.github/workflows/publish-train-image.yml`
- **Validation docs**: `specs/004-aitraf-api-deployment/quickstart.md`
- **Existing train image precedent**: `packages/aitraf-train/Dockerfile`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the existing image publishing and API runtime surfaces before editing.

- [x] T001 Review existing train image conventions in `packages/aitraf-train/Dockerfile` for base image, uv install, system packages, workspace copy order, and package-scoped `uv sync`
- [x] T002 [P] Review existing publish workflow conventions in `.github/workflows/publish-train-image.yml` for GHCR login, metadata tags, Buildx cache, and `master`/manual triggers
- [x] T003 [P] Review API package runtime command and tests in `packages/aitraf-api/Taskfile.yml` before encoding Docker CMD and workflow test steps
- [x] T004 [P] Verify `.dockerignore` excludes `storage/`, `.env*`, notebooks, local models, virtualenvs, and cache directories for the root Docker build context

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared deployment constraints that must hold for both user stories.

**Critical**: No user story work should start until these constraints are confirmed.

- [x] T005 Confirm the API image dependency boundary from `packages/aitraf-api/pyproject.toml` and `packages/aitraf-core/pyproject.toml`: `aitraf-api` may depend on `aitraf-core`, but must not install `aitraf-train`
- [x] T006 Confirm runtime config requirements in `packages/aitraf-api/src/aitraf_api/config.py`: required env vars remain explicit and no Dockerfile defaults hide missing `AITRAF_*` or `MLFLOW_*` inputs
- [x] T007 [P] Confirm repo data/storage sizes and contents with `data/` and `storage/` so the API image copies only `data/` and keeps `storage/` external
- [x] T008 [P] Confirm current API tests pass locally with `task api:test` for `packages/aitraf-api/tests` before workflow gating is added

**Checkpoint**: Foundation ready. The API Dockerfile and workflow tasks can now proceed.

---

## Phase 3: User Story 1 - Create API Dockerfile (Priority: P1) MVP

**Goal**: Provide a production Dockerfile for `aitraf-api` that builds a serving image from repository sources with only API/core runtime code and small repo data.

**Independent Test**: Build the image with `docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .`, inspect that train/storage are absent, and run the documented smoke command with explicit runtime inputs.

### Validation for User Story 1

- [x] T009 [P] [US1] Add or update the local API image build command and expected outcome in `specs/004-aitraf-api-deployment/quickstart.md`
- [x] T010 [P] [US1] Add or update API Docker runtime smoke instructions and required env vars in `packages/aitraf-api/README.md`
- [x] T011 [US1] Run `docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .` against `packages/aitraf-api/Dockerfile` and record any build blockers in `specs/004-aitraf-api-deployment/quickstart.md`

### Implementation for User Story 1

- [x] T012 [US1] Create `packages/aitraf-api/Dockerfile` by reusing the `packages/aitraf-train/Dockerfile` pattern with CUDA runtime, uv, frozen package-scoped install, and API-specific working directory
- [x] T013 [US1] In `packages/aitraf-api/Dockerfile`, copy root `pyproject.toml`, `uv.lock`, `packages/aitraf-core`, `packages/aitraf-api`, and `data/` while excluding `packages/aitraf-train` and `storage/`
- [x] T014 [US1] In `packages/aitraf-api/Dockerfile`, install only the API package path with `uv sync --frozen --no-dev --no-editable --package aitraf-api`
- [x] T015 [US1] In `packages/aitraf-api/Dockerfile`, set the production command to start `aitraf_api.app:create_app_from_env` through Uvicorn on `0.0.0.0:8000`
- [x] T016 [US1] Validate the built `aitraf-api:local` image does not contain `packages/aitraf-train` or `/workspace/storage` using a container inspection command against `packages/aitraf-api/Dockerfile`
- [x] T017 [US1] Run a missing-env smoke check against `aitraf-api:local` and confirm startup fails explicitly through `packages/aitraf-api/src/aitraf_api/config.py`

**Checkpoint**: User Story 1 is complete when the API image builds locally, starts with explicit runtime inputs, and excludes train/storage contents.

---

## Phase 4: User Story 2 - Publish API Image From GitHub Workflow (Priority: P2)

**Goal**: Extend the existing image publishing workflow so `master` and manual dispatch build both images, with API tests gating only API image publishing.

**Independent Test**: Inspect and validate `.github/workflows/publish-train-image.yml` so train publish is independent, API publish depends on API tests, and the API image target is `ghcr.io/${{ github.repository_owner }}/aitraf-api`.

### Validation for User Story 2

- [x] T018 [P] [US2] Add or update workflow validation expectations in `specs/004-aitraf-api-deployment/quickstart.md` for train independence, API test gating, and API GHCR image naming
- [x] T019 [P] [US2] Run `task api:test` for `packages/aitraf-api/tests` locally before editing `.github/workflows/publish-train-image.yml`
- [x] T020 [US2] Validate `.github/workflows/publish-train-image.yml` preserves `push` to `master` and `workflow_dispatch` triggers after workflow edits

### Implementation for User Story 2

- [x] T021 [US2] Rename or restructure jobs in `.github/workflows/publish-train-image.yml` so the existing train publish path remains an independent `aitraf-train` image job
- [x] T022 [US2] Add an API test job in `.github/workflows/publish-train-image.yml` that checks out the repo, installs locked API test dependencies, and runs `uv run pytest packages/aitraf-api/tests`
- [x] T023 [US2] Add an API publish job in `.github/workflows/publish-train-image.yml` that depends on the API test job and publishes `ghcr.io/${{ github.repository_owner }}/aitraf-api`
- [x] T024 [US2] Configure API image metadata in `.github/workflows/publish-train-image.yml` with `latest` and short-SHA tags matching the existing train image convention
- [x] T025 [US2] Configure API Docker build in `.github/workflows/publish-train-image.yml` to use root context `.` and Dockerfile `packages/aitraf-api/Dockerfile` with GitHub Actions cache
- [x] T026 [US2] Confirm `.github/workflows/publish-train-image.yml` lets train publish continue without depending on API tests while API publish stops when API tests fail

**Checkpoint**: User Story 2 is complete when the workflow builds/publishes train independently and publishes API only after API tests pass.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final documentation, consistency, and end-to-end validation across both stories.

- [x] T027 [P] Update `packages/aitraf-api/README.md` with the final image name, local build command, required runtime env vars, and storage mounting expectations
- [x] T028 [P] Update `specs/004-aitraf-api-deployment/quickstart.md` if implementation changes the final Docker build, run, or workflow validation commands
- [x] T029 Run final API validation with `task api:test` for `packages/aitraf-api/tests`
- [x] T030 Run final Docker validation with `docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .` for `packages/aitraf-api/Dockerfile`
- [x] T031 Run final workflow review on `.github/workflows/publish-train-image.yml` and confirm image names, job dependencies, permissions, tags, and Buildx cache match `specs/004-aitraf-api-deployment/contracts/github-workflow.md`
- [x] T032 Verify no secrets, `.env*`, `storage/`, local models, notebooks, or generated cache directories are introduced into `packages/aitraf-api/Dockerfile`, `.github/workflows/publish-train-image.yml`, or committed docs

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup completion and blocks user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion and needs `packages/aitraf-api/Dockerfile` from US1 before the API publish job can be fully validated
- **Polish (Phase 5)**: Depends on selected user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: MVP. Can start after Foundational. No dependency on US2.
- **User Story 2 (P2)**: Can start workflow restructuring after Foundational, but full API publish validation depends on US1 Dockerfile existing.

### Within Each User Story

- Validation expectations come before implementation.
- Dockerfile content comes before local Docker build validation.
- API tests come before API workflow publish job validation.
- Workflow dependency review comes after all workflow job edits.

### Parallel Opportunities

- T002, T003, and T004 can run in parallel during Setup.
- T007 and T008 can run in parallel during Foundational.
- T009 and T010 can run in parallel for US1 documentation/validation preparation.
- T018 and T019 can run in parallel for US2 validation preparation.
- T027 and T028 can run in parallel during Polish.

---

## Parallel Example: User Story 1

```bash
# Documentation/validation preparation can run together:
Task: "T009 update specs/004-aitraf-api-deployment/quickstart.md"
Task: "T010 update packages/aitraf-api/README.md"
```

---

## Parallel Example: User Story 2

```bash
# Pre-work can run together before workflow implementation:
Task: "T018 update specs/004-aitraf-api-deployment/quickstart.md"
Task: "T019 run task api:test for packages/aitraf-api/tests"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: Create API Dockerfile.
4. Stop and validate the API image independently with local Docker build and runtime smoke checks.

### Incremental Delivery

1. Deliver US1 so a local production API image can be built and inspected.
2. Deliver US2 so the existing workflow publishes train and API images on `master`, with API tests gating API publish only.
3. Complete Polish validation and documentation updates.

### Parallel Team Strategy

After Foundational:

- Developer A can complete US1 Dockerfile implementation and Docker validation.
- Developer B can prepare US2 workflow test-gate edits, but final API publish validation waits for the US1 Dockerfile.

## Notes

- [P] tasks use different files or are command-only validations that do not depend on incomplete edits.
- [US1] maps to Create API Dockerfile.
- [US2] maps to Publish API Image From GitHub Workflow.
- Keep `storage/` external and do not commit secrets.
- API test failure must block only API image publishing; train publishing must remain independent.
