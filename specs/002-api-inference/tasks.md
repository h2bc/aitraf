# Tasks: API Inference Surface

**Input**: Design documents from `/workspace/specs/002-api-inference/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/openapi.yaml`

**Validation**: Basic API tests are required. Use plain pytest Arrange/Act/Assert structure with descriptive test names, fixtures for setup, one obvious action, and direct assertions. Do not add BDD step comments. Do not test model quality, model performance, metric values, or exhaustive edge cases.

**Organization**: Tasks are grouped by user story so each story can be implemented and verified independently after shared foundations are complete.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the reserved API package for FastAPI implementation.

- [X] T001 Add FastAPI, Uvicorn, python-dotenv, MLflow, Pydantic, pytest, and test-client dependencies plus the local `aitraf-core` dependency in `packages/aitraf-api/pyproject.toml`
- [X] T002 [P] Create the API module scaffold files under `packages/aitraf-api/src/aitraf_api/`, including feature modules for health, demo videos, trick classification, and trick assessment
- [X] T003 [P] Create API test scaffolding in `packages/aitraf-api/tests/__init__.py` and `packages/aitraf-api/tests/conftest.py`
- [X] T004 [P] Replace the placeholder package note with FastAPI run and test entrypoints in `packages/aitraf-api/README.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared runtime pieces that must exist before endpoint user stories.

**Critical**: No user story work should begin until this phase is complete.

- [X] T005 Implement settings loading from `.env` and environment for `AITRAF_DATA_PATH`, `AITRAF_STORAGE_PATH`, `MLFLOW_TRACKING_URI`, API auth token, MLflow credentials, and registered model references in `packages/aitraf-api/src/aitraf_api/config.py`
- [X] T006 Ensure `packages/aitraf-api/src/aitraf_api/config.py` keeps API-owned config limited to registered model URIs, compact model kind, and derived manifest/clip path suffixes
- [X] T007 Implement bearer-token validation dependency for protected routes in `packages/aitraf-api/src/aitraf_api/auth.py`
- [X] T008 [P] Define typed response schemas for health, demo videos, inference results, model info, and errors in `packages/aitraf-api/src/aitraf_api/schemas.py`
- [X] T009 [P] Use FastAPI `HTTPException` responses for unauthorized, invalid selection, not found, and service-unavailable states in route, auth, manifest, and service modules
- [X] T010 Implement JSONL manifest loading, required-column validation, flat matching demo-video selection, and per-endpoint `video_id` resolution in `packages/aitraf-api/src/aitraf_api/manifests.py`
- [X] T011 Implement registered-model loading, caching, prediction calls, and response formatting in `packages/aitraf-api/src/aitraf_api/prediction.py`
- [X] T012 Ensure `packages/aitraf-api/src/aitraf_api/prediction.py` uses the registered MLflow model package for model and processor loading instead of API-local model constants
- [X] T013 Ensure video/frame preparation reuses `aitraf-core` helpers through shared core prediction functions and does not import `aitraf-train`
- [X] T014 Implement `create_app()` wiring, dependency injection hooks, and route registration in `packages/aitraf-api/src/aitraf_api/app.py`
- [X] T015 Implement reusable pytest fixtures with temporary manifests, configured token, FastAPI test client, and stubbed model loading/prediction in `packages/aitraf-api/tests/conftest.py`

**Checkpoint**: Foundation ready. Endpoint stories can now be implemented and tested.

---

## Phase 3: User Story 1 - Verify API Availability (Priority: P1)

**Goal**: Let the frontend or operator confirm the API process is reachable.

**Independent Test**: Call `GET /health` with no parameters and verify a successful service status.

### Validation for User Story 1

- [X] T016 [P] [US1] Add `test_health_returns_ok` in `packages/aitraf-api/tests/features/health/test_health.py`

### Implementation for User Story 1

- [X] T017 [US1] Implement `GET /health` with no request parameters and response `{"status": "ok"}` in `packages/aitraf-api/src/aitraf_api/features/health/route.py` and `packages/aitraf-api/src/aitraf_api/features/health/service.py`

**Checkpoint**: Health endpoint works independently.

---

## Phase 4: User Story 2 - List Demo Video Choices (Priority: P1)

**Goal**: Return a flat display list of matching current test-manifest videos for the demo UI.

**Independent Test**: Call `GET /demo-videos` with a valid token and no parameters, then verify a flat `videos` list with display metadata from both required manifests.

### Validation for User Story 2

- [X] T018 [P] [US2] Add `test_demo_videos_returns_matching_rows` in `packages/aitraf-api/tests/features/demo_videos/test_demo_videos.py`
- [X] T019 [P] [US2] Add `test_demo_videos_returns_explicit_error_for_missing_required_manifest` in `packages/aitraf-api/tests/features/demo_videos/test_demo_videos.py`

### Implementation for User Story 2

- [X] T020 [US2] Implement demo-video row shaping with `id`, `video_id`, and optional display fields in `packages/aitraf-api/src/aitraf_api/features/demo_videos/service.py`
- [X] T021 [US2] Implement `GET /demo-videos` with no request parameters, token protection, and `DemoVideosResponse` output in `packages/aitraf-api/src/aitraf_api/features/demo_videos/route.py`
- [X] T022 [US2] Return explicit service errors for missing, unreadable, or empty required manifests through the demo-videos feature and manifest helpers

**Checkpoint**: Demo video listing works independently with stub data.

---

## Phase 5: User Story 3 - Run Trick Classification Prediction (Priority: P1)

**Goal**: Run trick classification inference for a selected demo video id.

**Independent Test**: Call `POST /inference/trick-classification/{id}` with a valid token and a stubbed predictor, then verify decoded prediction label, confidence, ground truth label, video id, and compact model metadata.

### Validation for User Story 3

- [X] T023 [P] [US3] Add `test_classification_returns_prediction_for_valid_video_id` in `packages/aitraf-api/tests/features/trick_classification/test_trick_classification.py`
- [X] T024 [P] [US3] Add `test_classification_returns_invalid_selection_error_for_unknown_video_id` in `packages/aitraf-api/tests/features/trick_classification/test_trick_classification.py`

### Implementation for User Story 3

- [X] T025 [US3] Implement classification id resolution against `data/manifests/trick_classification/test.jsonl` in `packages/aitraf-api/src/aitraf_api/manifests.py`
- [X] T026 [US3] Implement classification prediction orchestration that loads the selected clip from the derived storage clips directory in `packages/aitraf-api/src/aitraf_api/features/trick_classification/service.py`
- [X] T027 [US3] Implement classification result formatting with user-facing trick label, required confidence, ground truth label, and compact model metadata via `packages/aitraf-api/src/aitraf_api/prediction.py`
- [X] T028 [US3] Implement `POST /inference/trick-classification/{id}` with only the `id` path parameter and token protection in `packages/aitraf-api/src/aitraf_api/features/trick_classification/route.py`
- [X] T029 [US3] Return explicit errors for missing classification rows, missing clips, missing registered model metadata, and unavailable model artifacts through the functional service helpers

**Checkpoint**: Trick classification inference works independently with stubbed model loading and prediction.

---

## Phase 6: User Story 4 - Run Trick AQA Prediction (Priority: P1)

**Goal**: Run trick AQA inference for a selected demo video id using the ordinal score prediction surface.

**Independent Test**: Call `POST /inference/trick-aqa/{id}` with a valid token and a stubbed predictor, then verify decoded AQA label or value, confidence, ground truth label, video id, and compact model metadata.

### Validation for User Story 4

- [X] T030 [P] [US4] Add `test_aqa_returns_prediction_for_valid_video_id` in `packages/aitraf-api/tests/features/trick_assessment/test_trick_assessment.py`
- [X] T031 [P] [US4] Add `test_aqa_returns_invalid_selection_error_for_unknown_video_id` in `packages/aitraf-api/tests/features/trick_assessment/test_trick_assessment.py`

### Implementation for User Story 4

- [X] T032 [US4] Implement AQA id resolution against `data/manifests/score_prediction_ordinal/test.jsonl` in `packages/aitraf-api/src/aitraf_api/manifests.py`
- [X] T033 [US4] Implement AQA prediction orchestration that loads the selected clip from the derived storage clips directory in `packages/aitraf-api/src/aitraf_api/features/trick_assessment/service.py`
- [X] T034 [US4] Implement AQA result formatting with user-facing score label or value, required confidence, ground truth label, and compact model metadata via `packages/aitraf-api/src/aitraf_api/prediction.py`
- [X] T035 [US4] Implement `POST /inference/trick-aqa/{id}` with only the `id` path parameter and token protection in `packages/aitraf-api/src/aitraf_api/features/trick_assessment/route.py`
- [X] T036 [US4] Return explicit errors for missing AQA rows, missing clips, missing registered model metadata, and unavailable model artifacts through the functional service helpers

**Checkpoint**: Trick AQA inference works independently with stubbed model loading and prediction.

---

## Phase 7: User Story 5 - Restrict API Access (Priority: P1)

**Goal**: Reject unapproved callers for protected API capabilities.

**Independent Test**: Call one protected endpoint without a token, with an invalid token, and with the configured token.

### Validation for User Story 5

- [X] T037 [P] [US5] Add `test_demo_videos_rejects_missing_token` in `packages/aitraf-api/tests/test_auth.py`
- [X] T038 [P] [US5] Add `test_demo_videos_rejects_invalid_token` in `packages/aitraf-api/tests/test_auth.py`
- [X] T039 [P] [US5] Add `test_demo_videos_allows_valid_token` in `packages/aitraf-api/tests/features/demo_videos/test_demo_videos.py`

### Implementation for User Story 5

- [X] T040 [US5] Apply the token validation dependency to `GET /demo-videos`, `POST /inference/trick-classification/{id}`, and `POST /inference/trick-aqa/{id}` in the endpoint-specific route modules
- [X] T041 [US5] Keep `GET /health` unauthenticated for service monitoring in `packages/aitraf-api/src/aitraf_api/features/health/route.py`
- [X] T042 [US5] Return consistent missing-token and invalid-token errors from `packages/aitraf-api/src/aitraf_api/auth.py`

**Checkpoint**: Protected endpoint access is token-gated.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final alignment, smoke validation, and cleanup.

- [X] T043 [P] Align implemented response schemas and status codes with `specs/002-api-inference/contracts/openapi.yaml`
- [X] T044 [P] Document required `.env` variables, model registry assumptions, and local run command in `packages/aitraf-api/README.md`
- [X] T045 Verify API package has no `aitraf-train` imports by running `rg "aitraf_train|aitraf-train" packages/aitraf-api/src packages/aitraf-api/tests` and recording the result in `packages/aitraf-api/README.md`
- [X] T046 Run basic API tests with `pytest packages/aitraf-api/tests` and record the command in `packages/aitraf-api/README.md`
- [X] T047 Run package lint or import smoke validation for `packages/aitraf-api/src/aitraf_api` using the repository's available Python tooling and record the command in `packages/aitraf-api/README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup and blocks all endpoint stories.
- **User Stories (Phases 3-7)**: Depend on Foundational. All are P1 and can proceed in parallel after shared foundations if staffed.
- **Polish (Phase 8)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 Health**: No dependency on other user stories after foundation.
- **US2 Demo Videos**: No dependency on other user stories after foundation.
- **US3 Trick Classification**: No dependency on US4 after foundation.
- **US4 Trick AQA**: No dependency on US3 after foundation.
- **US5 Auth**: Can be tested independently after foundation, but final route protection must be applied before shipping protected endpoints.

### Within Each User Story

- Write the focused pytest test first using fixtures, a single action call, and direct assertions.
- Implement only the endpoint behavior needed for that story.
- Verify the story independently with stubbed model loading/prediction where inference is involved.

---

## Parallel Opportunities

- T002, T003, and T004 can run in parallel after T001 is understood.
- T008 and T009 can run in parallel with T005-T007 because they touch different modules.
- T016, T018, T019, T023, T024, T030, T031, and T037-T039 can be drafted in parallel after test fixtures exist.
- US3 and US4 implementation can run in parallel because they use separate endpoint mappings and manifest paths.
- T043 and T044 can run in parallel during polish.

---

## Parallel Example: User Story 3

```text
Task: T023 Add the valid classification inference API test in packages/aitraf-api/tests/features/trick_classification/test_trick_classification.py
Task: T024 Add the unknown-id classification API test in packages/aitraf-api/tests/features/trick_classification/test_trick_classification.py
```

---

## Parallel Example: User Story 4

```text
Task: T030 Add the valid AQA inference API test in packages/aitraf-api/tests/features/trick_assessment/test_trick_assessment.py
Task: T031 Add the unknown-id AQA API test in packages/aitraf-api/tests/features/trick_assessment/test_trick_assessment.py
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: US1 health endpoint.

### Demo-Usable Increment

1. Complete MVP First.
2. Complete Phase 7: US5 auth.
3. Complete Phase 4: US2 demo-video listing.

### Full API Increment

1. Complete Demo-Usable Increment.
2. Complete Phase 5: US3 trick classification inference.
3. Complete Phase 6: US4 trick AQA inference.
4. Complete Phase 8: Polish and validation.
